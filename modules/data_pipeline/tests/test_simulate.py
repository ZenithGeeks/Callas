import shutil
import wave

import dspy
import numpy as np
import pytest
import yaml
from dspy.utils import DummyLM

from data_pipeline import graph as graph_module
from data_pipeline import simulate
from data_pipeline import turns as turn_module
from data_pipeline.agents import SIGNATURES, Agents
from data_pipeline.budget import prompt_tokens
from data_pipeline.config import DEFAULT_DIR, ConfigError, load_config
from data_pipeline.finish import _judge_score, drop_false_spelling, scam_start
from data_pipeline.records import Judge, Turn
from data_pipeline.scenarios import sample_config
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


def judge(
    holds=True,
    start=-1,
    giveaway=False,
    scam_clear=True,
    language_realistic=True,
    setup_holds=True,
):
    return dict(
        scam_clear=scam_clear,
        scam_clarity_reason="clear evidence",
        label_holds=holds,
        label_reason="matches the intended type" if holds else "does not match the intended type",
        language_realistic=language_realistic,
        language_realism_reason="natural dialogue" if language_realistic else "repetitive dialogue",
        setup_holds=setup_holds,
        setup_reason="follows its persona and scenario" if setup_holds else "ignores its persona",
        scam_start_turn=start,
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
    # relationship calls work out the footing once; strangers never call it
    a.footing = Script([dict(footing="พี่สาวที่ไว้ใจ คุยกันเองไม่เป็นทางการ เรียกตัวเองว่าหนู เรียกเขาว่าพี่")] * 5)
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
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()

    assert rec["generation"]["status"] == "accepted", rec["generation"]["reasons"]
    assert rec["label_scam"] and rec["label_type"] == "phishing_otp"
    assert rec["scam_start_turn"] == 2 and rec["generation"]["scam_start_from"] == "caller"
    assert [t["speaker"] for t in rec["speaker_turns"]] == ["A", "B", "A", "B", "A"]
    assert rec["text"].splitlines()[0].startswith("A: สวัสดีครับ")
    for e in rec["entities"]:  # spans index into the final text
        assert rec["text"][e["start"] : e["end"]] == e["text"]
    assert {e["label"] for e in rec["entities"]} == {"ORG", "OTP", "BANK_ACCOUNT"}
    assert rec["reference_summary"] == "สรุป"
    assert rec["generation"]["judge"]["confidence"] == 1.0
    assert rec["generation"]["config_hash"] == CONFIG.hash
    assert store.load(record_id("phishing_otp", 1)) == rec


def test_turn_text_is_cleaned_and_raw_keeps_the_model_line(tmp_path, monkeypatch):
    from nlp_core import text as nlp_text

    monkeypatch.setattr(nlp_text, "_mlm", lambda: None)  # no sentence repair: it would fix เกิด่
    messy = "เกิด่่ขึ้น   กับ  {ORG} ครับ"  # doubled tone mark and extra spaces
    caller = [dict(SCAM[0], line=messy), *SCAM[1:]]
    agents = make_agents(caller, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    first = rec["generation"]["turns"][0]
    assert first["raw"] == messy
    assert first["text"].startswith("เกิด่ขึ้น กับ ")
    assert "  " not in first["text"]
    assert "[0]" in agents.judge.calls[0]["transcript"]
    assert first["text"] in agents.judge.calls[0]["transcript"]
    for e in rec["entities"]:  # spans index into the cleaned text
        assert rec["text"][e["start"] : e["end"]] == e["text"]


def test_a_misspelled_word_is_corrected_in_text_whatever_the_setting(tmp_path, monkeypatch):
    from nlp_core import text as nlp_text

    monkeypatch.setattr(nlp_text, "_fix_word", lambda w: "ขอบคุณ" if w == "ขอบคนุ" else w)
    replies = [dict(REPLIES[0], line="ขอบคนุ ค่ะ มีอะไรคะ"), *REPLIES[1:]]
    agents = make_agents(SCAM, replies, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    reply = rec["generation"]["turns"][1]
    assert reply["raw"] == "ขอบคนุ ค่ะ มีอะไรคะ"
    assert reply["text"] == "ขอบคุณ ค่ะ มีอะไรคะ"
    assert "ขอบคุณ ค่ะ" in agents.judge.calls[0]["transcript"]


def test_a_fixed_tone_mark_is_listed_in_fixes_not_flagged_or_rejected(tmp_path):
    garbled = "ผมไม่สะดวกโอนเงินเข้าบัญชีอื่่นๆ แบบนั้้นนะ"  # doubled tone marks (studio, #romance)
    replies = [dict(REPLIES[0], line=garbled), *REPLIES[1:]]
    agents = make_agents(SCAM, replies, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    reply = rec["generation"]["turns"][1]
    assert reply["raw"] == garbled
    assert reply["text"] == "ผมไม่สะดวกโอนเงินเข้าบัญชีอื่นๆ แบบนั้นนะ"
    assert "tone_mark_order" not in reply["spelling"]  # the spell check reads the cleaned line
    assert reply["fixes"] == [
        {"before": "อื่่นๆ", "after": "อื่นๆ"},
        {"before": "นั้้น", "after": "นั้น"},
    ]
    assert "tone_mark_order" not in rec["generation"]["reasons"]


@pytest.mark.parametrize(
    ("before", "after", "fixes"),
    [
        ("สวัสดี   ครับ", "สวัสดี ครับ", []),  # spacing alone is not listed
        ("เยอะมากเลยท่ีรัก", "เยอะมากเลยที่รัก", [{"before": "ท่ีรัก", "after": "ที่รัก"}]),
        ("ให้คณุนะคะ", "ให้คุณนะคะ", [{"before": "คณุ", "after": "คุณ"}]),
    ],
)
def test_text_fixes_lists_each_changed_word(before, after, fixes):
    assert turn_module.text_fixes(before, after) == fixes


@pytest.mark.parametrize(
    ("scam_clear", "label_holds", "language_realistic", "setup_holds", "expected"),
    [
        (True, True, True, True, 1.0),
        (True, False, True, True, 0.75),
        (True, True, False, True, 0.75),
        (True, True, True, False, 0.75),
        (False, True, False, True, 0.5),
        (False, False, False, False, 0.0),
    ],
)
def test_judge_score_gives_each_of_the_four_checks_25_points(
    scam_clear, label_holds, language_realistic, setup_holds, expected
):
    assert _judge_score(scam_clear, label_holds, language_realistic, setup_holds) == expected


def test_the_judge_sees_the_selected_scenario_and_persona(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    asked = agents.judge.calls[0]
    call = rec["generation"]["call"]
    assert asked["scenario"]["role"] == call["scenario"]["role"]
    assert asked["scenario"]["goal"] == call["scenario"]["goal"]
    assert "key" not in asked["scenario"] and "key" not in asked["persona"]
    assert asked["persona"]["occupation"] == call["receiver"]["occupation"]
    assert rec["generation"]["judge"]["setup_holds"] is True


@pytest.mark.parametrize(
    "verdict",
    [
        judge(start=2, language_realistic=False, setup_holds=False),  # 50: exactly the pass mark
        judge(start=2, scam_clear=False, holds=False),  # 50
        judge(start=2, holds=False, setup_holds=False),  # 50
        judge(
            start=-1,
            holds=False,
            giveaway=True,
            scam_clear=False,
            language_realistic=False,
            setup_holds=False,
        ),
    ],
)
def test_a_score_of_50_or_less_rejects_with_only_the_score(tmp_path, verdict):
    agents = make_agents(SCAM, REPLIES, verdict)
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["status"] == "rejected"
    assert rec["generation"]["reasons"] == ["judge_low_confidence"]


NO_ASKS = [dict(d, asks_now=False) for d in SCAM]


@pytest.mark.parametrize(
    ("caller", "verdict"),
    [
        (SCAM, judge(start=2, holds=False)),  # 75: does not show the intended call
        (SCAM, judge(start=2, setup_holds=False)),  # 75: strays from its persona or scenario
        (SCAM, judge(start=2, giveaway=True)),  # 100: names the label outright
        (NO_ASKS, judge(start=-1)),  # 100: no scam move from the caller or the judge
    ],
)
def test_a_score_above_50_accepts_whatever_the_other_judge_answers_say(tmp_path, caller, verdict):
    agents = make_agents(caller, REPLIES, verdict)
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["status"] == "accepted", rec["generation"]["reasons"]


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
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    assert rec["scam_start_turn"] == 2
    assert rec["generation"]["scam_start_from"] == "judge"


def test_a_scam_start_in_a_normal_call_no_longer_rejects_it(tmp_path):
    legit = [dict(line="สวัสดีครับ", end_call=False), dict(line="ขอบคุณครับ", end_call=True)]
    agents = make_agents(legit, [dict(line="ค่ะ", end_call=False)], judge(start=0))
    store = Store(tmp_path / "calls.db")
    rec = run_call("benign", 1, store, agents, CONFIG).to_dict()
    assert rec["label_scam"] is False and rec["scam_start_turn"] is None
    # only the code check holds: the judge's score of 100 passes the call
    assert rec["generation"]["reasons"] == ["too_short"]


def test_resume_continues_from_saved_turns(tmp_path, monkeypatch):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    real_speak = graph_module.write_turn
    calls = {"n": 0}

    def crash_on_third(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_speak(*a, **kw)

    monkeypatch.setattr(graph_module, "write_turn", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        run_call("phishing_otp", 1, store, agents, CONFIG)
    saved = store.load(record_id("phishing_otp", 1))
    assert len(saved["generation"]["turns"]) == 2

    monkeypatch.setattr(graph_module, "write_turn", real_speak)
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    assert len(rec["generation"]["turns"]) == 5
    assert rec["generation"]["status"] == "accepted"


def test_resume_with_another_config_is_refused(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = simulate.new_record(simulate.sample_config("phishing_otp", 1, CONFIG), CONFIG, agents)
    store.save_call(rec)

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
    rec = run_call("phishing_otp", 1, store, agents, CONFIG, voices=voices).to_dict()

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
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    plan = ["build trust", "ask for the OTP", "push for a transfer"]
    assert rec["generation"]["briefing"]["plan"] == plan
    assert len(agents.planner.calls) == 1
    assert all(kw["plan"] == plan for kw in agents.scam_caller.calls)


def test_normal_call_has_no_plan(tmp_path):
    legit = [dict(line="สวัสดีครับ", end_call=False), dict(line="ขอบคุณครับ", end_call=True)]
    agents = make_agents(legit, [dict(line="ค่ะ", end_call=False)], judge(start=-1))
    store = Store(tmp_path / "calls.db")
    rec = run_call("benign", 1, store, agents, CONFIG).to_dict()
    assert rec["generation"]["briefing"] is None
    assert len(agents.planner.calls) == 0


def test_receiver_gets_structured_persona(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    run_call("phishing_otp", 1, store, agents, CONFIG)
    for kw in agents.receiver.calls:
        assert set(kw) == {"persona", "voice", "stage", "footing", "red_flags", "conversation"}
        assert kw["footing"] == ""  # a stranger call sets no prior relationship, so no footing
        assert "key" not in kw["persona"]  # the profile key is not for the model
        assert isinstance(kw["persona"]["occupation"], str) and kw["persona"]["occupation"]
        assert isinstance(kw["persona"]["region"], str) and kw["persona"]["region"]
        assert isinstance(kw["persona"]["age"], int)
        assert isinstance(kw["voice"], str) and kw["voice"]
        # the same general scam warning signs reach every receiver, whatever the call is about
        assert kw["red_flags"] == CONFIG.red_flags and kw["red_flags"]


def test_a_romance_call_gives_the_receiver_and_footing_the_scenario_backstory(tmp_path):
    # the footing step and the receiver both get the scenario's prior relationship, so the ground
    # truth ("met on a dating app", a family role) reaches them, not just the footing model's guess
    call = sample_config("romance", 2, CONFIG)
    context = call.scenario.receiver_context
    assert context  # romance scenarios set a prior relationship the receiver believes in
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    run_call("romance", 2, Store(tmp_path / "calls.db"), agents, CONFIG)
    assert agents.footing.calls and agents.footing.calls[0]["context"] == context
    assert agents.receiver.calls and all(context in c["stage"] for c in agents.receiver.calls)


def test_the_footing_is_stored_and_a_resume_reuses_it(tmp_path, monkeypatch):
    # The footing is saved with the call, so a dataset reviewer can see what the receiver was told
    # and a resume reuses it instead of asking the footing model again.
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    real_speak = graph_module.write_turn
    calls = {"n": 0}

    def crash_on_third(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real_speak(*a, **kw)

    monkeypatch.setattr(graph_module, "write_turn", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        run_call("romance", 2, store, agents, CONFIG)
    footing = store.load(record_id("romance", 2))["generation"]["footing"]
    assert footing and len(agents.footing.calls) == 1

    monkeypatch.setattr(graph_module, "write_turn", real_speak)
    rec = run_call("romance", 2, store, agents, CONFIG).to_dict()
    assert rec["generation"]["status"] == "accepted"
    assert rec["generation"]["footing"] == footing
    assert len(agents.footing.calls) == 1  # not asked again on resume
    assert all(c["footing"] == footing for c in agents.receiver.calls)


def test_a_stranger_call_stores_an_empty_footing(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["footing"] == ""  # read, and there is no prior relationship
    assert not agents.footing.calls


def test_planner_reads_pattern_notes(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.add_note(
        "phishing_otp",
        "Callers pose as bank security",
        ["say an account was hacked", "ask for the OTP"],
        ["มีคนพยายามเข้าบัญชีของคุณ"],
    )
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    assert "Callers pose as bank security" in agents.planner.calls[0]["pattern_notes"]
    assert rec["generation"]["briefing"]["notes_used"] == ["Callers pose as bank security"]


def test_receiver_hears_its_own_name_as_the_slot(tmp_path):
    store = Store(tmp_path / "calls.db")
    greet = [dict(line="สวัสดีค่ะ คุณ{RECEIVER_NAME} ผมจาก {ORG}", asks_now=False, end_call=False)]
    agents = make_agents(greet + SCAM[1:], REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    name = rec["generation"]["turns"][0]["entities"][0]["text"]  # the filled receiver name
    heard = agents.receiver.calls[0]["conversation"]
    assert "{RECEIVER_NAME}" in heard and name not in heard
    assert name in rec["text"]  # the stored call keeps the real name


def test_receiver_persona_has_no_key_even_for_a_named_profile(tmp_path):
    profile = CONFIG.scenarios.receivers.profiles[0]
    assert profile.key
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG, receiver=profile)
    persona = agents.receiver.calls[0]["persona"]
    assert "key" not in persona and persona["occupation"] == profile.occupation


# ---- resume, retries, ids ----------------------------------------------------------------


class RateLimitError(Exception):
    """Named like the errors DSPy raises for a rate limit, so the retry policy accepts it."""


@pytest.fixture
def fast_graph(monkeypatch):
    """The call graph with retry waits of milliseconds instead of 5 s."""
    graph = simulate.build_graph(simulate.retry_policy(initial_interval=0.001))
    monkeypatch.setattr(simulate, "CALL_GRAPH", graph)


@pytest.mark.parametrize("saved", [1, 3])
def test_resume_after_an_odd_number_of_turns(tmp_path, monkeypatch, fast_graph, saved):
    """After 1 or 3 saved turns the receiver is next; `plan` must route there."""
    store = Store(tmp_path / "calls.db")
    real_speak = graph_module.write_turn
    seen = {"n": 0}

    def stop_after_saved(*a, **kw):
        seen["n"] += 1
        if seen["n"] == saved + 1:
            raise KeyboardInterrupt
        return real_speak(*a, **kw)

    monkeypatch.setattr(graph_module, "write_turn", stop_after_saved)
    first = make_agents(SCAM, REPLIES, judge(start=2))
    with pytest.raises(KeyboardInterrupt):
        run_call("phishing_otp", 1, store, first, CONFIG)
    assert len(store.load(record_id("phishing_otp", 1))["generation"]["turns"]) == saved

    monkeypatch.setattr(graph_module, "write_turn", real_speak)
    callers_done, receivers_done = (saved + 1) // 2, saved // 2
    second = make_agents(SCAM[callers_done:], REPLIES[receivers_done:], judge(start=2))
    rec = run_call("phishing_otp", 1, store, second, CONFIG).to_dict()
    turns = rec["generation"]["turns"]
    assert [t["speaker"] for t in turns] == ["A", "B", "A", "B", "A"]
    assert rec["generation"]["status"] == "accepted", rec["generation"]["reasons"]
    assert len(second.planner.calls) == 0  # the saved plan is kept


def test_a_failed_save_inside_a_turn_does_not_add_the_turn_twice(tmp_path, fast_graph):
    store = Store(tmp_path / "calls.db")
    real_save = store.save
    failures = []

    def save_fails_once_on_the_receiver_turn(record):
        if len(record["generation"]["turns"]) == 2 and not failures:
            failures.append(1)
            raise RateLimitError("try again")
        real_save(record)

    store.save = save_fails_once_on_the_receiver_turn
    # the receiver is asked twice for its first answer: once before the failed save, once after
    receiver = [REPLIES[0], REPLIES[0], REPLIES[1]]
    agents = make_agents(SCAM, receiver, judge(start=2))
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()

    assert failures == [1]
    saved = store.load(record_id("phishing_otp", 1))
    assert [t["speaker"] for t in saved["generation"]["turns"]] == ["A", "B", "A", "B", "A"]
    assert saved["generation"]["turns"] == rec["generation"]["turns"]


def test_a_key_error_in_a_node_is_not_retried(tmp_path, fast_graph):
    class Boom:
        calls = 0

        def __call__(self, **kw):
            Boom.calls += 1
            raise KeyError("not a model problem")

    agents = make_agents(SCAM, REPLIES, judge(start=2))
    agents.receiver = Boom()
    with pytest.raises(KeyError):
        run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG)
    assert Boom.calls == 1


def test_rate_limit_is_retried(tmp_path, fast_graph):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    real = agents.receiver

    class Flaky:
        calls = 0

        def __call__(self, **kw):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RateLimitError("slow down")
            return real(**kw)

    agents.receiver = Flaky()
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert Flaky.calls == 3 and len(rec["generation"]["turns"]) == 5


def test_which_errors_count_as_model_errors():
    import httpx
    import openai
    from dspy.utils.exceptions import AdapterParseError

    request = httpx.Request("POST", "http://llm.test")

    def status(code):
        response = httpx.Response(code, request=request)
        return openai.APIStatusError("x", response=response, body=None)

    parse = AdapterParseError(adapter_name="chat", signature=SIGNATURES["judge"], lm_response="?")
    assert simulate.is_model_error(parse)
    assert simulate.is_model_error(RateLimitError())
    assert simulate.is_model_error(status(429)) and simulate.is_model_error(status(503))
    assert not simulate.is_model_error(status(400))
    assert simulate.is_model_error(dspy.LMServerError("502 from the gateway"))
    assert not simulate.is_model_error(dspy.LMTimeoutError("timed out"))
    for other in (KeyError("k"), ValueError("v"), RuntimeError("r"), ConfigError("c")):
        assert not simulate.is_model_error(other)


def sampled_with_a_named_profile(call_type="phishing_otp"):
    for seed in range(200):
        if simulate.sample_config(call_type, seed, CONFIG).receiver.key:
            return seed
    raise AssertionError("no seed drew a hand-written profile")


def test_a_sampled_hand_written_profile_keeps_the_plain_id(tmp_path):
    seed = sampled_with_a_named_profile()
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", seed, store, agents, CONFIG).to_dict()
    assert rec["id"] == f"phishing_otp_{seed:05d}"
    assert store.exists(rec["id"])
    assert rec["generation"]["call"]["receiver_override"] == ""


def test_a_chosen_receiver_gets_its_own_id_and_a_resume_must_match(tmp_path):
    store = Store(tmp_path / "calls.db")
    mine, other = CONFIG.scenarios.receivers.profiles[:2]
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    agents.scam_caller = Script([])  # no answers: the first caller turn fails, call stays open
    with pytest.raises(IndexError):
        run_call("phishing_otp", 1, store, agents, CONFIG, receiver=mine)
    stored = store.load(f"phishing_otp_{mine.key}_00001")
    assert stored["generation"]["status"] == "in_progress"
    assert stored["generation"]["call"]["receiver_override"] == simulate.receiver_hash(mine)

    again = make_agents(SCAM, REPLIES, judge(start=2))
    changed = mine.model_copy(update={"age": 99})
    with pytest.raises(ConfigError, match="different receiver"):
        run_call("phishing_otp", 1, store, again, CONFIG, receiver=changed)
    rec = run_call("phishing_otp", 1, store, again, CONFIG, receiver=mine).to_dict()
    assert rec["generation"]["status"] == "accepted", rec["generation"]["reasons"]


# ---- where the scam starts ---------------------------------------------------------------


def scam_record(asks, judge_at, reveal=2, holds=True):
    """Turns A,B,A,B,A,B, the reveal turn and the judge's answer; `asks` are the turn indexes
    with asks_now set."""
    turns = [Turn(speaker="AB"[i % 2], asks_now=i in asks) for i in range(6)]
    verdict = Judge(
        label_holds=holds,
        scam_start_turn=judge_at,
        giveaway=False,
        summary="",
        issues="",
        reasoning="",
    )
    return turns, reveal, verdict


def test_an_ask_before_the_reveal_turn_is_not_a_scam_move():
    # reveal turn 2 = speaker turn 4: an ask on turn 0 or 2 is too early, so the judge decides
    assert scam_start(*scam_record({0, 2}, judge_at=4)) == (4, "judge")
    assert scam_start(*scam_record({0, 2}, judge_at=-1)) == (None, None)
    assert scam_start(*scam_record({0, 4}, judge_at=-1)) == (4, "caller")
    assert scam_start(*scam_record({0}, judge_at=-1, reveal=0)) == (0, "caller")


def test_the_judge_fallback_must_point_at_a_caller_turn():
    assert scam_start(*scam_record(set(), judge_at=3)) == (None, None)  # turn 3 is the receiver
    assert scam_start(*scam_record(set(), judge_at=2)) == (2, "judge")
    assert scam_start(*scam_record(set(), judge_at=2, holds=False)) == (None, None)


SPELLED = [
    Turn(speaker="A", text="สวัสดีครับ ผม ธนพล จากฝ่ายสนับสนุน", spelling=["ธนพล"]),
    Turn(speaker="B", text="ค่ะ มีอะไรคะ"),
]


def test_spelling_claims_the_spell_check_does_not_back_are_dropped():
    issues = "Severe spelling errors (e.g., สวัสดี, ฝ่าย). The call ends early."
    assert drop_false_spelling(issues, SPELLED) == "The call ends early."
    assert drop_false_spelling('Typo in [0] "สวัสดี".', SPELLED) == "none"  # spelled correctly
    assert drop_false_spelling('Typo in [1] "ธนพล".', SPELLED) == "none"  # not in turn 1
    assert drop_false_spelling('Typo in [9] "ธนพล".', SPELLED) == "none"  # no turn 9


def test_a_spelling_claim_the_spell_check_backs_is_kept():
    assert drop_false_spelling('Typo in [0] "ธนพล".', SPELLED) == 'Typo in [0] "ธนพล".'
    assert drop_false_spelling("", SPELLED) == ""
    assert drop_false_spelling("none", SPELLED) == "none"


def test_a_line_with_several_misspellings_is_rejected_as_garbled():
    from data_pipeline.finish import reasons_to_reject

    turns = [Turn(speaker="AB"[i % 2]) for i in range(6)]
    turns[1] = Turn(speaker="B", spelling=["จาย", "เทาไร"])  # two different non-words: garbled
    verdict = Judge(
        label_holds=True, scam_start_turn=2, giveaway=False, summary="", issues="", reasoning=""
    )
    assert "garbled_spelling" in reasons_to_reject(turns, verdict, CONFIG)


def test_one_odd_word_is_not_garbled():
    from data_pipeline.finish import reasons_to_reject

    turns = [Turn(speaker="AB"[i % 2]) for i in range(6)]
    turns[1] = Turn(speaker="B", spelling=["ลิ้งก์"])  # one loanword the dictionary lacks: tolerated
    verdict = Judge(
        label_holds=True, scam_start_turn=2, giveaway=False, summary="", issues="", reasoning=""
    )
    assert "garbled_spelling" not in reasons_to_reject(turns, verdict, CONFIG)


def test_a_garbled_caller_line_is_not_counted():
    from data_pipeline.finish import reasons_to_reject

    turns = [Turn(speaker="AB"[i % 2]) for i in range(6)]
    turns[0] = Turn(speaker="A", spelling=["จาย", "เทาไร"])  # caller line: this is a receiver check
    verdict = Judge(
        label_holds=True, scam_start_turn=2, giveaway=False, summary="", issues="", reasoning=""
    )
    assert "garbled_spelling" not in reasons_to_reject(turns, verdict, CONFIG)


def test_the_judge_gives_no_natural_thai_score(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert "natural_thai" not in rec["generation"]["judge"]
    assert "natural_thai" not in Agents(CONFIG, lms={}).signatures["judge"].output_fields


def test_judge_prompt_stays_small():
    """The scored verdict fields, editable realism rubric, and selected scenario and persona
    leave room for the transcript."""
    call = sample_config("romance", 1, CONFIG).to_dict()  # the longest scenario text
    tokens = prompt_tokens(
        Agents(CONFIG, lms={}).signatures["judge"],
        dict(
            intended="x",
            transcript="[0] ผู้โทร: สวัสดีครับ\n[1] ผู้รับ: ค่ะ",
            realism_criteria=CONFIG.agents.judge.realism_criteria,
            scenario=call["scenario"],
            persona=call["receiver"],
        ),
    )
    # Measured at 2,442 with dspy 3.4.1 (13 output fields, 9 criteria, a romance scenario and
    # persona). The prompt budget is about 96K, so this only guards against large growth.
    assert tokens <= 2_600


# ---- the receiver gives away its own details ---------------------------------------------

LEAKY = [
    dict(line="ค่ะ มีอะไรคะ", end_call=False),
    dict(line="รหัส {OTP} ค่ะ บัตรประชาชน {ID_CARD} ค่ะ", end_call=False),
]


def test_receiver_prompt_stays_small():
    """The receiver's prompt without the call. It grew from 776 as behaviour was added (#64, #99),
    then two short general few-shot examples replaced some prose (greeting-yourself, re-asking a
    given detail, concern-not-excitement) for ~1070, and then a short red_flags digest from the
    scam wiki (red_flags.py: one general warning sign per scam type, same for every call so the
    receiver stays blind to the label) was added to every receiver prompt for ~1420, and then a
    register-matching line (mirror the caller's level: formal, affectionate pet names and เค้า, or
    casual) for ~1540, and then that enumerated register prose was replaced by a `footing` field (a
    one-time read decides the relationship and register, so the receiver no longer re-derives it per
    turn) for ~1610, and then that footing clause was tightened to bind the receiver to footing's
    exact self-word and way of addressing the caller every turn (a later turn had drifted to a
    different kin term) for ~1665, and then a line that the home region is a background, not an
    accent to perform (answer in standard central Thai unless the caller uses dialect first) for
    ~1725. Content comes partly from the wiki, so expect some wording variance; this guard only
    catches runaway growth. Still tiny against the 100k window. A stranger call sets footing to "",
    measured here."""
    from data_pipeline.budget import prompt_tokens

    inputs = dict(
        persona="{}",
        voice="v",
        stage="s",
        footing="",
        red_flags=CONFIG.red_flags,
        conversation="[0] ผู้โทร: สวัสดีครับ",
    )
    assert prompt_tokens(Agents(CONFIG, lms={}).signatures["receiver"], inputs) <= 1750


def test_receiver_leaks_are_recorded_by_code(tmp_path):
    agents = make_agents(SCAM, LEAKY, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    turns = rec["generation"]["turns"]
    assert [t["leaked"] for t in turns] == [[], [], [], ["OTP", "ID_CARD"], []]
    assert rec["generation"]["receiver_leaked"] == ["ID_CARD", "OTP"]
    assert rec["generation"]["first_leak_turn"] == 3
    labels = {e["label"] for t in turns[3:4] for e in t["entities"]}
    assert labels == {"OTP", "ID_CARD"}
    assert Store(tmp_path / "calls.db").load(rec["id"]) == rec


def test_a_call_without_leaks_records_none(tmp_path):
    quiet = [dict(line="ค่ะ มีอะไรคะ", end_call=False), dict(line="ไม่ให้ค่ะ", end_call=False)]
    agents = make_agents(SCAM, quiet, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["receiver_leaked"] == []
    assert rec["generation"]["first_leak_turn"] is None


def test_only_receiver_turns_count_as_leaks(tmp_path):
    caller = [
        dict(line="เบอร์คุณ {RECEIVER_PHONE} ใช่ไหมครับ {OTP}", asks_now=False, end_call=False),
        dict(line="บัญชี {RECEIVER_ACCOUNT} ใช่ไหมครับ", asks_now=True, end_call=False),
        dict(line="ขอบคุณครับ", asks_now=False, end_call=True),
    ]
    quiet = [dict(line="ค่ะ {RECEIVER_NAME} พูดค่ะ {PHONE}", end_call=False)] * 2
    agents = make_agents(caller, quiet, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["receiver_leaked"] == []
    assert rec["generation"]["first_leak_turn"] is None
    assert all(t["leaked"] == [] for t in rec["generation"]["turns"])


def test_the_receiver_sees_its_own_details_as_slots(tmp_path):
    agents = make_agents(SCAM, LEAKY, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    call = simulate.sample_config("phishing_otp", 1, CONFIG)
    id_card = simulate.SlotFiller(call, CONFIG.details).values["ID_CARD"]
    turns = rec["generation"]["turns"]
    assert id_card in turns[3]["text"]
    typed = [Turn.from_dict(turn) for turn in turns]
    view = turn_module.receiver_view(typed, simulate.SlotFiller(call, CONFIG.details), CONFIG)
    assert "{ID_CARD}" in view and id_card not in view


def test_the_receiver_hears_the_callers_numbers_as_slots(tmp_path):
    echo = [dict(line="โอนไป {BANK_ACCOUNT} ใช่ไหมคะ โทรกลับ {PHONE} ได้ไหมคะ", end_call=False)] * 2
    agents = make_agents(SCAM, echo, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    call = simulate.sample_config("phishing_otp", 1, CONFIG)
    values = simulate.SlotFiller(call, CONFIG.details).values
    last_view = agents.receiver.calls[-1]["conversation"]
    assert "{BANK_ACCOUNT}" in last_view and values["BANK_ACCOUNT"] not in last_view
    receiver_turns = [t for t in rec["generation"]["turns"] if t["speaker"] == "B"]
    # repeating the caller's numbers is neither a leak nor a number without a slot
    assert all(t["leaked"] == [] and t["problems"] == [] for t in receiver_turns)
    assert values["PHONE"] in receiver_turns[0]["text"]


def test_the_receiver_hears_a_name_chosen_in_the_studio_as_its_own_slot(tmp_path):
    import threading

    from data_pipeline import live

    greet = dict(line="สวัสดีครับ คุณ{RECEIVER_NAME}", asks_now=False, end_call=False)
    agents = make_agents([greet, *SCAM[1:]], REPLIES, judge(start=2))
    spec = live.CallSpec("phishing_otp", 1, None, None, "สมหญิง")
    store = Store(tmp_path / "calls.db")
    for _ in live.stream_call(spec, store, agents, CONFIG, None, live.Capture(), threading.Event()):
        pass
    assert "คุณสมหญิง" in store.load(spec.record_id)["generation"]["turns"][0]["text"]
    heard = agents.receiver.calls[0]["conversation"]
    assert "คุณ{RECEIVER_NAME}" in heard and "สมหญิง" not in heard


# ---- a caller does not know the receiver's own details -------------------------------------


def test_a_caller_using_a_receiver_only_slot_is_flagged_and_rejected(tmp_path):
    caller = [dict(d) for d in SCAM]
    caller[1] = dict(line="บัญชี {RECEIVER_ACCOUNT} ใช่ไหมครับ", asks_now=True, end_call=False)
    agents = make_agents(caller, REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    turns = rec["generation"]["turns"]
    assert "caller_used_receiver_slot" not in turns[0]["problems"]
    assert "caller_used_receiver_slot" in turns[2]["problems"]
    assert rec["generation"]["status"] == "rejected"
    assert "caller_used_receiver_slot" in rec["generation"]["reasons"]


def test_a_caller_may_use_the_receivers_name(tmp_path):
    greet = dict(line="สวัสดีครับ คุณ{RECEIVER_NAME}", asks_now=False, end_call=False)
    agents = make_agents([greet, *SCAM[1:]], REPLIES, judge(start=2))
    rec = run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG).to_dict()
    assert rec["generation"]["turns"][0]["problems"] == []


def test_the_prompts_list_the_receiver_slots_only_for_the_receiver():
    from data_pipeline.fill import RECEIVER_ONLY_SLOTS

    agents = CONFIG.agents
    for slot in RECEIVER_ONLY_SLOTS:
        for caller in (agents.scam_caller, agents.legit_caller):
            assert "{" + slot + "}" not in caller.fields["line"]
        assert "{" + slot + "}" in agents.receiver.instructions
    assert "{RECEIVER_NAME}" in agents.scam_caller.fields["line"]


def test_each_scam_type_gets_the_shared_rules_and_only_its_own():
    signatures = Agents(CONFIG, lms={}).scam_caller_signatures
    assert set(signatures) == set(CONFIG.agents.scam_caller.by_type)
    shared = CONFIG.agents.scam_caller.instructions.strip()
    text = {scam_type: s.instructions for scam_type, s in signatures.items()}
    assert all(t.startswith(shared) for t in text.values())
    office = ("เรียนสายคุณ {RECEIVER_NAME}", "never ฉัน", "ทางเราส่งลิงก์", "{ORG}")
    for scam_type in ("fake_authority", "investment", "lottery", "phishing_otp"):
        assert all(rule in text[scam_type] for rule in office)
    # romance talks like someone the receiver knows: no office words, org or slip form
    banned = ("ทางเรา", "{ORG}", "สลิป", "form in the link")
    assert not any(word in text["romance"] for word in banned)
    assert "แอดไลน์ไอดี" in text["fake_authority"] and "แอดไลน์ไอดี" in text["investment"]
    assert "แอดไลน์ไอดี" not in text["phishing_otp"] and "สลิป" not in text["phishing_otp"]


@pytest.mark.parametrize("call_type", ["phishing_otp", "romance"])
def test_the_scam_caller_is_asked_with_its_call_type(tmp_path, call_type):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    run_call(call_type, 1, Store(tmp_path / "calls.db"), agents, CONFIG)
    assert {kw["scam_type"] for kw in agents.scam_caller.calls} == {call_type}


def test_an_office_scam_caller_without_a_scenario_org_still_gets_the_org_slot(tmp_path):
    seed = next(s for s in range(200) if not sample_config("phishing_otp", s, CONFIG).scenario.org)
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    run_call("phishing_otp", seed, Store(tmp_path / "calls.db"), agents, CONFIG)
    assert {kw["org"] for kw in agents.scam_caller.calls} == {"{ORG}"}  # its opener says {ORG}
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    run_call("romance", 1, Store(tmp_path / "calls.db"), agents, CONFIG)
    assert {kw["org"] for kw in agents.scam_caller.calls} == {""}


def test_a_command_line_resume_keeps_the_studio_scenario_and_name(tmp_path):
    import threading

    from data_pipeline import live

    sampled = simulate.sample_config("romance", 3, CONFIG).scenario
    chosen = next(s for s in CONFIG.scenarios.scams["romance"] if s.key != sampled.key)
    store = Store(tmp_path / "calls.db")
    agents = live.fake_agents(CONFIG, "romance")
    capture, cancel = live.Capture(agents), threading.Event()
    spec = live.CallSpec("romance", 3, chosen, None, "สมหญิง")
    for event in live.stream_call(spec, store, agents, CONFIG, None, capture, cancel):
        if event.kind == "turn" and event.index == 1:
            cancel.set()  # stopped in the studio after two turns
    assert len(store.load("romance_00003")["generation"]["turns"]) == 2

    caller = [
        dict(line="คุณ{RECEIVER_NAME}ครับ ขอรหัส {OTP} ด้วยครับ", asks_now=True, end_call=False),
        dict(line="ขอบคุณครับ", asks_now=False, end_call=True),
    ]
    resumed = make_agents(caller, [dict(line="ค่ะ", end_call=False)], judge(start=2))
    rec = run_call("romance", 3, store, resumed, CONFIG).to_dict()
    assert all(kw["goal"] == chosen.goal for kw in resumed.scam_caller.calls)
    assert "คุณสมหญิงครับ" in rec["generation"]["turns"][2]["text"]
    assert rec["generation"]["call"]["scenario"]["key"] == chosen.key


def test_a_command_line_resume_keeps_the_studio_forced_outcome(tmp_path):
    import threading

    from data_pipeline import live

    # The studio can force the scam outcome; forcing skips the profile draw, so it changes the whole
    # sampling path. A command line resume passes no outcome, so it must read the forced one back
    # from the stored call and rebuild the same receiver, not a fresh one.
    store = Store(tmp_path / "calls.db")
    agents = live.fake_agents(CONFIG, "romance")
    capture, cancel = live.Capture(agents), threading.Event()
    spec = live.CallSpec("romance", 3, outcome="scammed")
    for event in live.stream_call(spec, store, agents, CONFIG, None, capture, cancel):
        if event.kind == "turn" and event.index == 1:
            cancel.set()  # stopped in the studio after two turns
    started = store.load("romance_00003")["generation"]["call"]["receiver"]
    assert started["scam_outcome"] == "scammed"
    assert simulate.sample_config("romance", 3, CONFIG).receiver.scam_outcome != "scammed"

    caller = [
        dict(line="ขอรหัส {OTP} ด้วยครับ", asks_now=True, end_call=False),
        dict(line="ขอบคุณครับ", asks_now=False, end_call=True),
    ]
    resumed = make_agents(caller, [dict(line="ค่ะ", end_call=False)], judge(start=2))
    rec = run_call("romance", 3, store, resumed, CONFIG).to_dict()  # --resume, no outcome
    assert rec["generation"]["call"]["receiver"] == started  # the same person, not a fresh draw
    assert rec["generation"]["status"] == "accepted"
