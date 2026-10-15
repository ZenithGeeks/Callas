import shutil
import wave

import dspy
import numpy as np
import pytest
import yaml
from dspy.utils import DummyLM

from data_pipeline import simulate
from data_pipeline.agents import SIGNATURES, Agents
from data_pipeline.config import DEFAULT_DIR, ConfigError, load_config
from data_pipeline.simulate import record_id, run_call
from data_pipeline.store import Store

CONFIG = load_config()


class Script:
    """Stands in for one ChainOfThought program: returns the next scripted answer."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return dspy.Prediction(reasoning="thinking", **self.answers.pop(0))


def judge(holds=True, start=-1, giveaway=False):
    return dict(
        label_holds=holds,
        scam_start_turn=start,
        natural_thai=4,
        giveaway=giveaway,
        summary="สรุป",
        issues="none",
    )


def make_agents(caller, receiver, verdict, config=CONFIG):
    a = Agents(config, lms={name: DummyLM([]) for name in SIGNATURES})
    a.scam_caller = a.legit_caller = Script(caller)
    a.receiver = Script(receiver)
    a.judge = Script([verdict])
    a.planner = Script([dict(plan=["build trust", "ask for the OTP", "push for a transfer"])])
    return a


SCAM = [
    dict(line="สวัสดีครับ ผมจาก {ORG} ครับ", asks_now=False, end_call=False),
    dict(line="บัญชีคุณมีปัญหา ขอรหัส OTP ด้วยครับ", asks_now=True, end_call=False),
    dict(line="โอนมาที่ {BANK_ACCOUNT} นะครับ", asks_now=True, end_call=True),
]
REPLIES = [
    dict(line="ค่ะ มีอะไรคะ", end_call=False),
    dict(line="รหัส {OTP} ค่ะ", end_call=False),
]


def test_scam_call_record(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)

    assert rec["generation"]["status"] == "accepted", rec["generation"]["reasons"]
    assert rec["label_scam"] and rec["label_type"] == "phishing_otp"
    assert rec["scam_start_turn"] == 2 and rec["generation"]["scam_start_from"] == "caller"
    assert [t["speaker"] for t in rec["speaker_turns"]] == ["A", "B", "A", "B", "A"]
    assert rec["text"].splitlines()[0].startswith("A: สวัสดีครับ")
    for e in rec["entities"]:  # spans index into the final text
        assert rec["text"][e["start"] : e["end"]] == e["text"]
    assert {e["label"] for e in rec["entities"]} == {"ORG", "OTP", "BANK_ACCOUNT"}
    assert rec["reference_summary"] == "สรุป"
    assert rec["generation"]["config_hash"] == CONFIG.hash
    assert store.load(record_id("phishing_otp", 1)) == rec


def test_each_turn_sees_the_log_so_far(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    run_call("phishing_otp", 1, store, agents, CONFIG)
    caller_views = [c["conversation"] for c in agents.scam_caller.calls]
    receiver_views = [c["conversation"] for c in agents.receiver.calls]
    assert caller_views[0] == CONFIG.agents.speakers.empty_call
    # the caller sees slots, the receiver sees filled values
    assert "{ORG}" in caller_views[1]
    assert "{ORG}" not in receiver_views[0]
    assert caller_views[2].count("\n") == 3  # four earlier turns


def test_judge_supplies_start_turn_when_caller_does_not_flag(tmp_path):
    caller = [dict(d, asks_now=False) for d in SCAM]
    agents = make_agents(caller, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    assert rec["scam_start_turn"] == 2
    assert rec["generation"]["scam_start_from"] == "judge"


def test_scam_without_any_ask_is_rejected(tmp_path):
    caller = [dict(d, asks_now=False) for d in SCAM]
    agents = make_agents(caller, REPLIES, judge(holds=False, start=-1))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    assert rec["generation"]["status"] == "rejected"
    assert {"no_scam_move", "judge_label_does_not_hold"} <= set(rec["generation"]["reasons"])


def test_normal_call_the_judge_calls_a_scam_is_rejected(tmp_path):
    legit = [dict(line="สวัสดีครับ", end_call=False), dict(line="ขอบคุณครับ", end_call=True)]
    agents = make_agents(legit, [dict(line="ค่ะ", end_call=False)], judge(start=0))
    store = Store(tmp_path / "calls.db")
    rec = run_call("benign", 1, store, agents, CONFIG)
    assert rec["label_scam"] is False and rec["scam_start_turn"] is None
    assert "judge_saw_scam_in_normal_call" in rec["generation"]["reasons"]
    assert "too_short" in rec["generation"]["reasons"]


def test_resume_continues_from_saved_turns(tmp_path, monkeypatch):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    real_speak = simulate._speak
    calls = {"n": 0}

    def crash_on_third(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_speak(*a, **kw)

    monkeypatch.setattr(simulate, "_speak", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        run_call("phishing_otp", 1, store, agents, CONFIG)
    saved = store.load(record_id("phishing_otp", 1))
    assert len(saved["generation"]["turns"]) == 2

    monkeypatch.setattr(simulate, "_speak", real_speak)
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    assert len(rec["generation"]["turns"]) == 5
    assert rec["generation"]["status"] == "accepted"


def test_resume_with_another_config_is_refused(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = simulate.new_record(simulate.sample_config("phishing_otp", 1, CONFIG), CONFIG, agents)
    store.save(rec)

    other = tmp_path / "other_cfg"
    shutil.copytree(DEFAULT_DIR, other)
    path = other / "details.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["banks"].append("ธนาคารใหม่")
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="was started with config"):
        run_call("phishing_otp", 1, store, agents, load_config(other))


class FakeVoices:
    def __init__(self):
        self.spoken = []

    def speak(self, text, voice, path):
        self.spoken.append((text, voice))
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.zeros(16000, dtype="<i2").tobytes())  # 1 s
        return 1.0


def test_each_turn_is_spoken_and_joined(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    voices = FakeVoices()
    rec = run_call("phishing_otp", 1, store, agents, CONFIG, voices=voices)

    caller_voice, receiver_voice = (rec["generation"]["voices"][s] for s in "AB")
    assert caller_voice != receiver_voice
    assert [v for _, v in voices.spoken] == [caller_voice, receiver_voice] * 2 + [caller_voice]
    assert voices.spoken[1][0] == rec["speaker_turns"][1]["text"]  # filled text is spoken
    assert rec["audio_path"] == f"audio/{rec['id']}.wav"
    gap = CONFIG.agents.tts.gap_s
    assert rec["speaker_turns"][1]["start"] == pytest.approx(1.0 + gap)
    with wave.open(str(tmp_path / rec["audio_path"])) as w:
        assert w.getnframes() / 16000 == pytest.approx(5 + 4 * gap, abs=1e-3)


def test_scam_call_is_planned_once(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    plan = ["build trust", "ask for the OTP", "push for a transfer"]
    assert rec["generation"]["briefing"]["plan"] == plan
    assert len(agents.planner.calls) == 1
    assert all(kw["plan"] == plan for kw in agents.scam_caller.calls)


def test_normal_call_has_no_plan(tmp_path):
    legit = [dict(line="สวัสดีครับ", end_call=False), dict(line="ขอบคุณครับ", end_call=True)]
    agents = make_agents(legit, [dict(line="ค่ะ", end_call=False)], judge(start=-1))
    store = Store(tmp_path / "calls.db")
    rec = run_call("benign", 1, store, agents, CONFIG)
    assert rec["generation"]["briefing"] is None
    assert len(agents.planner.calls) == 0


def test_receiver_gets_structured_persona(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    run_call("phishing_otp", 1, store, agents, CONFIG)
    for kw in agents.receiver.calls:
        assert isinstance(kw["persona"].occupation, str) and kw["persona"].occupation
        assert isinstance(kw["persona"].region, str) and kw["persona"].region
        assert isinstance(kw["persona"].age, int)
        assert isinstance(kw["voice"], str) and kw["voice"]


def test_planner_reads_pattern_notes(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.add_note(
        "phishing_otp",
        "Callers pose as bank security",
        ["say an account was hacked", "ask for the OTP"],
        ["มีคนพยายามเข้าบัญชีของคุณ"],
    )
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    assert "Callers pose as bank security" in agents.planner.calls[0]["pattern_notes"]
    assert rec["generation"]["briefing"]["notes_used"] == ["Callers pose as bank security"]


def test_receiver_hears_its_own_name_as_the_slot(tmp_path):
    store = Store(tmp_path / "calls.db")
    greet = [dict(line="สวัสดีค่ะ คุณ{RECEIVER_NAME} ผมจาก {ORG}", asks_now=False, end_call=False)]
    agents = make_agents(greet + SCAM[1:], REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, store, agents, CONFIG)
    name = rec["generation"]["turns"][0]["entities"][0]["text"]  # the filled receiver name
    heard = agents.receiver.calls[0]["conversation"]
    assert "{RECEIVER_NAME}" in heard and name not in heard
    assert name in rec["text"]  # the stored call keeps the real name
