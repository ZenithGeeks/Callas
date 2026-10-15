import threading

import dspy
import pytest
from dspy.utils import DummyLM

from data_pipeline import live
from data_pipeline.agents import SIGNATURES, Agents
from data_pipeline.config import load_config
from data_pipeline.scenarios import sample_config
from data_pipeline.store import Store

CONFIG = load_config()


# ---- the DSPy callback ------------------------------------------------------------------------


class Boom(DummyLM):
    def __call__(self, *args, **kwargs):
        raise RuntimeError("model down")


def agents_with(lm):
    return Agents(CONFIG, lms={name: lm for name in SIGNATURES})


def test_capture_records_the_agent_call_with_prompt_answer_model_and_time():
    lm = DummyLM([{"reasoning": "because", "line": "สวัสดี", "end_call": False}])
    lm.model = "my-model"
    agents = agents_with(lm)
    capture = live.Capture(agents)
    with dspy.context(callbacks=[capture], lm=lm):
        agents.receiver(persona={"age": 30}, voice="v", stage="s", conversation="c")
    (run,) = capture.take()
    assert run.agent == "receiver" and run.module == "ChainOfThought"  # not the inner Predict
    assert run.model == "my-model" and run.seconds > 0 and run.error == ""
    (call,) = run.lm_calls
    sent = "\n".join(m["content"] for m in call.messages)
    assert "[[ ## conversation ## ]]\nc" in sent and "my-model" not in sent
    assert {m["role"] for m in call.messages} == {"system", "user"}
    assert "[[ ## line ## ]]\nสวัสดี" in call.answer
    assert capture.take() == []  # taken once


def test_capture_keeps_one_run_per_call_in_order():
    lm = DummyLM([{"reasoning": "r", "line": f"l{i}", "end_call": False} for i in range(2)])
    agents = agents_with(lm)
    capture = live.Capture(agents)
    with dspy.context(callbacks=[capture], lm=lm):
        for _ in range(2):
            agents.receiver(persona={}, voice="v", stage="s", conversation="c")
    first, second = capture.take()
    assert "l0" in first.lm_calls[0].answer and "l1" in second.lm_calls[0].answer


def test_capture_records_a_failing_model():
    lm = Boom([])
    agents = agents_with(lm)
    capture = live.Capture(agents)
    with dspy.context(callbacks=[capture], lm=lm), pytest.raises(RuntimeError):
        agents.receiver(persona={}, voice="v", stage="s", conversation="c")
    (run,) = capture.take()
    assert run.error == "RuntimeError"


def test_capture_ignores_lm_calls_outside_an_agent():
    lm = DummyLM([{"answer": "x"}])
    capture = live.Capture()
    with dspy.context(callbacks=[capture], lm=lm):
        lm("hello")  # an LM call with no module around it
    assert capture.take() == []


# ---- the step-by-step run ---------------------------------------------------------------------


def collect(spec, tmp_path, cancel=None, delay=0.0, voices=True):
    store = Store(tmp_path / "calls.db")
    agents = live.fake_agents(CONFIG, spec.call_type, delay)
    capture = live.Capture(agents)
    events = []
    with dspy.context(callbacks=[capture]):
        for event in live.stream_call(
            spec, store, agents, CONFIG, live.SilentVoices() if voices else None, capture, cancel
        ):
            events.append(event)
            if cancel is not None and event.kind == "turn" and event.index == 1:
                cancel.set()
    return store, events


def test_a_scam_call_streams_thinking_then_plan_turns_and_verdict(tmp_path):
    store, events = collect(live.CallSpec("romance", 2), tmp_path)
    kinds = [e.kind for e in events]
    assert kinds[:4] == ["thinking", "plan", "thinking", "turn"]
    assert kinds[-2:] == ["thinking", "verdict"]
    assert events[0].agent == "planner" and events[-2].agent == "judge"
    turns = [e for e in events if e.kind == "turn"]
    assert [t.index for t in turns] == list(range(len(turns)))
    assert [t.agent for t in turns[:2]] == ["scam_caller", "receiver"]
    # each thinking names the agent that works next, and a step shows what it captured
    assert [e.agent for e in events if e.kind == "thinking"][1:3] == ["scam_caller", "receiver"]
    for step in (events[1], turns[0], events[-1]):
        (module,) = step.modules
        assert module.lm_calls and module.model == live.FAKE_MODEL
    record = store.load("romance_00002")
    assert record["generation"]["status"] == "accepted"
    assert all(t["audio"] for t in record["generation"]["turns"])  # voices ran


def test_a_normal_call_has_no_plan_step(tmp_path):
    _, events = collect(live.CallSpec("benign", 0), tmp_path)
    assert "plan" not in {e.kind for e in events}
    assert events[0].kind == "thinking" and events[0].agent == "legit_caller"


def test_cancel_stops_after_the_step_and_leaves_the_call_unfinished(tmp_path):
    cancel = threading.Event()
    store, events = collect(live.CallSpec("romance", 0), tmp_path, cancel)
    assert events[-1].kind == "turn" and events[-1].index == 1
    assert "verdict" not in {e.kind for e in events}
    record = store.load("romance_00000")
    assert record["generation"]["status"] == "in_progress"
    assert len(record["generation"]["turns"]) == 2


def test_a_stopped_call_resumes_and_shows_its_earlier_turns(tmp_path):
    cancel = threading.Event()
    collect(live.CallSpec("romance", 0), tmp_path, cancel)
    store, events = collect(live.CallSpec("romance", 0), tmp_path)
    first = [e for e in events if e.kind == "turn"][:2]
    assert [e.index for e in first] == [0, 1] and first[0].modules == []  # replayed, not rerun
    assert events[-1].kind == "verdict"
    assert store.load("romance_00000")["generation"]["status"] == "accepted"


def test_a_call_stopped_after_its_plan_resumes_with_the_plan_first(tmp_path):
    store = Store(tmp_path / "calls.db")
    agents = live.fake_agents(CONFIG, "romance", 0.0)
    capture = live.Capture(agents)
    cancel = threading.Event()
    with dspy.context(callbacks=[capture]):
        for event in live.stream_call(
            live.CallSpec("romance", 0), store, agents, CONFIG, None, capture, cancel
        ):
            if event.kind == "plan":
                cancel.set()
    assert store.load("romance_00000")["generation"]["turns"] == []  # stopped after the plan

    _, events = collect(live.CallSpec("romance", 0), tmp_path)
    kinds = [e.kind for e in events]
    assert kinds[0] == "plan" and kinds.count("plan") == 1
    assert kinds.index("plan") < kinds.index("turn")


def test_a_finished_call_is_not_run_again(tmp_path):
    collect(live.CallSpec("benign", 0), tmp_path)
    with pytest.raises(ValueError, match="already finished"):
        collect(live.CallSpec("benign", 0), tmp_path)


def test_scenario_receiver_and_name_are_applied(tmp_path):
    scenario = CONFIG.scenarios.scams["romance"][-1]
    profile = CONFIG.scenarios.receivers.profiles[0]
    spec = live.CallSpec("romance", 5, scenario, profile, "สมหญิง")
    store, _ = collect(spec, tmp_path)
    record = store.load(spec.record_id)
    call = record["generation"]["call"]
    assert call["scenario"]["key"] == scenario.key and call["receiver"]["key"] == profile.key
    assert call["receiver_name"] == "สมหญิง" and "สมหญิง" in record["text"]
    assert spec.record_id == f"romance_{profile.key}_00005"


def test_next_free_seed_skips_taken_ids():
    taken = {"romance_00000", "romance_00001", "romance_00003", "benign_00000"}
    assert live.next_free_seed("romance", taken, None) == 2
    assert live.next_free_seed("benign", taken, None) == 1
    assert live.next_free_seed("investment", taken, None) == 0


def test_a_model_error_rejects_the_call_like_the_command_line(tmp_path):
    import openai

    store, _ = collect(live.CallSpec("benign", 0), tmp_path)  # a finished call: left alone
    live.mark_model_error(
        store, "benign_00000", openai.RateLimitError.__new__(openai.RateLimitError)
    )
    assert store.load("benign_00000")["generation"]["status"] == "accepted"
    cancel = threading.Event()
    store, _ = collect(live.CallSpec("romance", 0), tmp_path, cancel)
    live.mark_model_error(store, "romance_00000", RuntimeError("a bug"))  # not the model's fault
    assert store.load("romance_00000")["generation"]["status"] == "in_progress"
    live.mark_model_error(
        store, "romance_00000", openai.RateLimitError.__new__(openai.RateLimitError)
    )
    record = store.load("romance_00000")
    assert record["generation"]["status"] == "rejected"
    assert record["generation"]["reasons"] == ["llm_error:RateLimitError"]


def test_a_resumed_call_keeps_its_scenario_and_name_and_refuses_others(tmp_path):
    sampled = sample_config("romance", 3, CONFIG).scenario  # what an empty form would pick
    second = sampled
    first = next(s for s in CONFIG.scenarios.scams["romance"] if s.key != sampled.key)
    cancel = threading.Event()
    spec = live.CallSpec("romance", 3, first, None, "สมหญิง")
    store, _ = collect(spec, tmp_path, cancel)
    stored = store.load("romance_00003")["generation"]["call"]
    assert stored["scenario"]["key"] == first.key and stored["receiver_name"] == "สมหญิง"
    with pytest.raises(live.CallConflict, match="scenario"):
        collect(live.CallSpec("romance", 3, second), tmp_path)
    with pytest.raises(live.CallConflict, match="name"):
        collect(live.CallSpec("romance", 3, None, None, "อื่น"), tmp_path)
    store, events = collect(live.CallSpec("romance", 3), tmp_path)  # empty form: keeps both
    prompts = " ".join(
        m["content"] for e in events for md in e.modules for c in md.lm_calls for m in c.messages
    )
    assert (
        first.goal in prompts and sampled.goal not in prompts
    )  # the callers used the stored scenario
    record = store.load("romance_00003")
    assert record["generation"]["status"] == "accepted"
    assert record["generation"]["call"]["scenario"]["key"] == first.key
    assert "สมหญิง" in record["text"]


def test_a_call_that_ends_early_yields_an_ended_step_then_a_verdict(tmp_path, monkeypatch):
    from data_pipeline import graph
    from data_pipeline.budget import ContextTooLong

    real = graph.write_turn

    def next_turn(agents, call, config, written, *rest):
        if len(written) >= 2:
            raise ContextTooLong("scam_caller", 9000, 8000)
        return real(agents, call, config, written, *rest)

    monkeypatch.setattr(graph, "write_turn", next_turn)
    store, events = collect(live.CallSpec("romance", 0), tmp_path)
    kinds = [e.kind for e in events]
    assert kinds[-3:] == ["thinking", "ended", "verdict"]
    ended = events[-2]
    assert ended.index == 2 and ended.agent == "scam_caller"  # the turn that was not written
    assert [e.index for e in events if e.kind == "turn"] == [0, 1]
    record = store.load("romance_00000")["generation"]
    assert record["status"] == "rejected" and record["judge"] is None
    assert events[-1].kind == "verdict"
