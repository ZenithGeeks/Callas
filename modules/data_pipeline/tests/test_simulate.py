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
from data_pipeline.finish import drop_false_spelling, scam_start
from data_pipeline.records import Judge, Turn
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
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    assert rec["scam_start_turn"] == 2
    assert rec["generation"]["scam_start_from"] == "judge"


def test_scam_without_any_ask_is_rejected(tmp_path):
    caller = [dict(d, asks_now=False) for d in SCAM]
    agents = make_agents(caller, REPLIES, judge(holds=False, start=-1))
    store = Store(tmp_path / "calls.db")
    rec = run_call("phishing_otp", 1, store, agents, CONFIG).to_dict()
    assert rec["generation"]["status"] == "rejected"
    assert {"no_scam_move", "judge_label_does_not_hold"} <= set(rec["generation"]["reasons"])


def test_normal_call_the_judge_calls_a_scam_is_rejected(tmp_path):
    legit = [dict(line="สวัสดีครับ", end_call=False), dict(line="ขอบคุณครับ", end_call=True)]
    agents = make_agents(legit, [dict(line="ค่ะ", end_call=False)], judge(start=0))
    store = Store(tmp_path / "calls.db")
    rec = run_call("benign", 1, store, agents, CONFIG).to_dict()
    assert rec["label_scam"] is False and rec["scam_start_turn"] is None
    assert "judge_saw_scam_in_normal_call" in rec["generation"]["reasons"]
    assert "too_short" in rec["generation"]["reasons"]


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
        assert set(kw) == {"persona", "voice", "stage", "conversation"}
        assert "key" not in kw["persona"]  # the profile key is not for the model
        assert isinstance(kw["persona"]["occupation"], str) and kw["persona"]["occupation"]
        assert isinstance(kw["persona"]["region"], str) and kw["persona"]["region"]
        assert isinstance(kw["persona"]["age"], int)
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
        natural_thai=4,
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


def test_judge_prompt_stays_small():
    """The judge's prompt without the call (#58 added about 65 tokens to 841)."""
    tokens = prompt_tokens(
        Agents(CONFIG, lms={}).signatures["judge"],
        dict(intended="x", transcript="[0] ผู้โทร: สวัสดีครับ\n[1] ผู้รับ: ค่ะ"),
    )
    assert tokens <= 950


# ---- the receiver gives away its own details ---------------------------------------------

LEAKY = [
    dict(line="ค่ะ มีอะไรคะ", end_call=False),
    dict(line="รหัส {OTP} ค่ะ บัตรประชาชน {ID_CARD} ค่ะ", end_call=False),
]


def test_receiver_prompt_stays_small():
    """The receiver's prompt without the call (#64 added about 30 tokens to 776, #99 about 45
    more)."""
    from data_pipeline.budget import prompt_tokens

    inputs = dict(persona="{}", voice="v", stage="s", conversation="[0] ผู้โทร: สวัสดีครับ")
    assert prompt_tokens(Agents(CONFIG, lms={}).signatures["receiver"], inputs) <= 880


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
