import json

import pytest

from asr_diarization import pipeline
from asr_diarization.types import Transcript, Turn
from conftest import FakeSpeaker
from llm_serving.fake import FakeLLM
from orchestrator import CallSessions, Runtime, to_jsonable


def make():
    return CallSessions(Runtime(llm=FakeLLM(), speaker=FakeSpeaker()))


def test_end_deletes_call_memory(fakes):
    s = make()
    cid = s.start()
    s.on_turn(cid, Turn("A", 0, 1, "PHONE"))
    state = s.runtime.states[cid]
    assert state.turns and state.entities_seen
    s.end(cid)
    assert cid not in s.runtime.states
    assert state.turns == [] and state.entities_seen == [] and state.escalated is False
    s.end(cid)  # twice is fine
    with pytest.raises(KeyError):
        s.on_turn(cid, Turn("A", 0, 1, "hi"))


def test_start_custom_and_duplicate_id():
    s = make()
    assert s.start("c1") == "c1"
    with pytest.raises(ValueError):
        s.start("c1")
    assert s.start() != s.start()


def test_stream_turn_yields_nodes_in_order(fakes):
    s = make()
    nodes = [u["node"] for u in s.stream_turn(s.start(), Turn("A", 0, 1, "hello"))]
    assert nodes == ["remember", "analyze", "decide", "reply", "speak"]
    nodes = [u["node"] for u in s.stream_turn(s.start(), Turn("A", 0, 1, "SCAM"))]
    assert nodes == ["remember", "analyze", "decide", "escalate"]


def test_stream_updates_are_json_safe(fakes):
    s = make()
    for u in s.stream_turn(s.start(), Turn("A", 0, 1, "PHONE")):
        json.dumps(to_jsonable(u))


def test_simulate_call_feeds_turns_in_order_and_ends(fakes, monkeypatch):
    turns = [Turn("A", 0, 1, "one"), Turn("B", 1, 2, "two"), Turn("A", 2, 3, "three")]
    seen = []
    monkeypatch.setattr(pipeline, "run", lambda path: seen.append(path) or Transcript(turns=turns))
    s = make()
    items = list(s.simulate_call("call.wav"))
    assert seen == ["call.wav"]
    analyzed = [i for i in items if i["node"] == "analyze"]
    assert [i["turn_index"] for i in analyzed] == [0, 1, 2]
    ctx = analyzed[-1]["update"]["transcript"]  # the agent's replies sit between the turns
    assert ctx.startswith("A: one\nagent: ")
    assert ctx.endswith("A: three") and "B: two" in ctx
    assert s.runtime.states == {}  # session ended


def test_simulate_call_ends_session_when_abandoned(fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda p: Transcript(turns=[Turn("A", 0, 1, "x")]))
    s = make()
    gen = s.simulate_call("c.wav")
    next(gen)
    assert len(s.runtime.states) == 1
    gen.close()
    assert s.runtime.states == {}
