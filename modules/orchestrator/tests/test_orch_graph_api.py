"""Graph API use: runtime context, retry on analyze only, mermaid drawing."""

import pytest

from asr_diarization.types import Turn
from conftest import FakeSpeaker
from llm_serving.client import LLMUnavailable
from llm_serving.fake import FakeLLM
from orchestrator import CallSessions, Runtime, build_message_graph, build_turn_graph
from tts_voice_agent.memory import CallState


def turn(text):
    return Turn("A", 0.0, 1.0, text)


def flaky(fakes, exc, fail_times=1):
    """Make the fake classifier raise `exc` the first `fail_times` calls, then answer normally."""
    real = fakes["classifier"].predict
    box = {"n": 0}

    def predict(text):
        box["n"] += 1
        if box["n"] <= fail_times:
            raise exc
        return real(text)

    fakes["classifier"].predict = predict
    return box


def test_analyze_retries_once_on_llm_unavailable(fakes):
    box = flaky(fakes, LLMUnavailable("down"))
    s = CallSessions(Runtime(llm=FakeLLM({"reply": "ค่ะ"}), speaker=FakeSpeaker()))
    out = s.on_turn(s.start(), turn("hello"))
    assert box["n"] == 2
    assert out["decision"].auto_reply and out["reply_text"] == "ค่ะ"


def test_analyze_gives_up_after_one_retry(fakes):
    box = flaky(fakes, LLMUnavailable("down"), fail_times=5)
    s = CallSessions(Runtime(llm=FakeLLM()))
    with pytest.raises(LLMUnavailable):
        s.on_turn(s.start(), turn("hello"))
    assert box["n"] == 2


def test_other_errors_are_not_retried(fakes):
    box = flaky(fakes, RuntimeError("boom"))
    s = CallSessions(Runtime(llm=FakeLLM()))
    with pytest.raises(RuntimeError, match="boom"):
        s.on_turn(s.start(), turn("hello"))
    assert box["n"] == 1


def test_message_graph_also_retries_analyze(fakes):
    box = flaky(fakes, LLMUnavailable("down"))
    s = CallSessions(Runtime(llm=FakeLLM()))
    assert s.analyze_message("hello")["verdict"]["level"] == "safe"
    assert box["n"] == 2


def test_retry_does_not_repeat_memory_writes(fakes):
    flaky(fakes, LLMUnavailable("down"))
    s = CallSessions(Runtime(llm=FakeLLM()))
    cid = s.start()
    s.on_turn(cid, turn("hello"))
    assert [t.speaker for t in s.runtime.states[cid].turns].count("A") == 1


def test_graphs_read_dependencies_from_context(fakes):
    """One compiled graph serves different runtimes: dependencies come from context=."""
    rt1 = Runtime(llm=FakeLLM({"reply": "หนึ่ง"}))
    rt2 = Runtime(llm=FakeLLM({"reply": "สอง"}))
    graph = build_turn_graph(rt1)
    for rt, want in ((rt1, "หนึ่ง"), (rt2, "สอง")):
        rt.states["c"] = CallState()
        out = graph.invoke(
            {"call_id": "c", "turn": turn("hello"), "timings": {}}, context=rt.context()
        )
        assert out["reply_text"] == want


def test_context_is_the_runtime(fakes):
    rt = Runtime(llm=FakeLLM(), ner="a", classifier="b", summarizer="c")
    ctx = rt.context()
    assert ctx is rt
    assert (ctx.ner, ctx.classifier, ctx.summarizer) == ("a", "b", "c")
    assert ctx.states is rt.states and ctx.llm is rt.llm


def test_graph_drawing_includes_speak_only_with_speaker(fakes):
    with_spk = build_turn_graph(Runtime(llm=FakeLLM(), speaker=FakeSpeaker()))
    without = build_turn_graph(Runtime(llm=FakeLLM()))
    assert "speak" in with_spk.get_graph().nodes
    assert "speak" not in without.get_graph().nodes
    assert "escalate" in with_spk.get_graph().draw_mermaid()
    assert "verdict" in build_message_graph().get_graph().nodes


def test_graph_has_no_cache_and_no_checkpointer(fakes):
    g = build_turn_graph(Runtime(llm=FakeLLM(), speaker=FakeSpeaker()))
    assert g.checkpointer is None and g.cache is None
