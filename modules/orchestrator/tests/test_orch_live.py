import base64
import json

from conftest import FakeSpeaker
from llm_serving.fake import FakeLLM
from orchestrator import CallSessions, Runtime
from orchestrator.live import LiveCall

REPLY = "สวัสดีค่ะ มีอะไรให้ช่วยคะ"


class FakeTranscriber:
    """feed() returns the next scripted batch of events; flush() returns the queued tail."""

    def __init__(self, batches=()):
        self.batches = list(batches)
        self.tail = []
        self.fed = []

    def feed(self, pcm):
        self.fed.append(pcm)
        return self.batches.pop(0) if self.batches else []

    def flush(self):
        out, self.tail = self.tail, []
        return out


def turn_ev(text, start=0.0, end=1.0, speaker="Caller"):
    return {"type": "turn", "speaker": speaker, "start": start, "end": end, "text": text}


def make(batches=(), speaker=True):
    sessions = CallSessions(Runtime(llm=FakeLLM({"reply": REPLY}), speaker=FakeSpeaker()))
    tr = FakeTranscriber(batches)
    return LiveCall(sessions, tr), sessions, tr


def drain_until_ended(call):
    call.end()
    return call.poll_events(10_000)


def types_of(events):
    return [e["type"] for e in events]


def test_benign_turn_event_order_and_contract(fakes):
    call, _, _ = make([[{"type": "partial", "text": "hel"}, turn_ev("hello")]])
    call.feed(b"\0\0")
    events = drain_until_ended(call)
    assert types_of(events)[:3] == ["ready", "partial", "turn"]
    kinds = types_of(events)
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert nodes == ["remember", "analyze", "decide", "reply", "speak"]
    assert kinds.index("turn") < kinds.index("node") < kinds.index("verdict") < kinds.index("reply")
    verdict = next(e for e in events if e["type"] == "verdict")
    assert verdict["turn"] == 1 and verdict["escalated"] is False and verdict["label_scam"] is False
    assert set(verdict) == {
        "type",
        "turn",
        "label_scam",
        "label_type",
        "confidence",
        "escalated",
        "reason",
        "entities",
        "summary",
    }
    reply = next(e for e in events if e["type"] == "reply")
    assert reply["text"] == REPLY and reply["template_used"] is False
    assert base64.b64decode(reply["audio_wav_b64"]) == b"RIFFfake"
    assert kinds[-1] == "ended" and events[0]["type"] == "ready"
    json.dumps(events)


def test_scam_turn_escalates_and_later_benign_stays_escalated(fakes):
    call, _, _ = make([[turn_ev("SCAM PHONE")], [turn_ev("hello again", speaker="Other")]])
    call.feed(b"1")
    call.feed(b"2")
    events = drain_until_ended(call)
    verdicts = [e for e in events if e["type"] == "verdict"]
    assert [v["turn"] for v in verdicts] == [1, 2]
    assert verdicts[0]["escalated"] is True and verdicts[0]["label_scam"] is True
    assert verdicts[0]["entities"] == [
        {"text": "0812345678", "label": "PHONE", "start": 0, "end": 10}
    ]
    assert verdicts[1]["escalated"] is True
    assert verdicts[1]["reason"] == "call already escalated"
    assert "reply" not in types_of(events)


def test_turns_are_processed_in_order(fakes):
    batches = [[turn_ev(f"hello {i}", speaker=f"S{i}")] for i in range(5)]
    call, _, _ = make(batches)
    for _ in batches:
        call.feed(b"x")
    events = drain_until_ended(call)
    assert [e["turn"] for e in events if e["type"] == "verdict"] == [1, 2, 3, 4, 5]
    assert [e["turn"] for e in events if e["type"] == "reply"] == [1, 2, 3, 4, 5]


def test_pause_drops_audio_and_flushes_open_turn(fakes):
    call, _, tr = make()
    tr.tail = [turn_ev("hello")]
    call.pause()
    call.feed(b"dropped")
    assert tr.fed == []
    call.resume()
    call.feed(b"kept")
    assert tr.fed == [b"kept"]
    events = drain_until_ended(call)
    assert types_of(events).count("turn") == 1 and "reply" in types_of(events)


def test_end_deletes_memory_and_emits_ended_last(fakes):
    call, sessions, tr = make([[turn_ev("hello")]])
    tr.tail = [turn_ev("PHONE", 2.0, 3.0, speaker="Other")]
    call.feed(b"x")
    cid = call.call_id
    assert cid in sessions.runtime.states
    events = drain_until_ended(call)
    assert cid not in sessions.runtime.states
    assert events[-1] == {"type": "ended", "call_id": cid}
    assert types_of(events).count("verdict") == 2  # flush at end processed the open turn
    call.end()  # idempotent
    assert call.poll_events() == []
    call.feed(b"late")
    assert tr.fed == [b"x"]


def test_exception_in_turn_gives_error_event_and_call_continues(fakes, monkeypatch):
    call, sessions, _ = make([[turn_ev("secret words")], [turn_ev("hello", speaker="Other")]])
    real = sessions.stream_turn
    state = {"n": 0}

    def flaky(call_id, turn):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom secret words")
        yield from real(call_id, turn)

    monkeypatch.setattr(sessions, "stream_turn", flaky)
    call.feed(b"1")
    call.feed(b"2")
    events = drain_until_ended(call)
    err = next(e for e in events if e["type"] == "error")
    assert err["turn"] == 1 and err["message"].startswith("RuntimeError")
    assert "secret" not in err["message"]
    assert [e["turn"] for e in events if e["type"] == "reply"] == [2]
    assert [e["turn"] for e in events if e["type"] == "done"] == [1, 2]  # a failed turn too


def test_no_speaker_gives_reply_with_null_audio(fakes):
    sessions = CallSessions(Runtime(llm=FakeLLM({"reply": REPLY}), speaker=None))
    call = LiveCall(sessions, FakeTranscriber([[turn_ev("hello")]]))
    call.feed(b"x")
    events = drain_until_ended(call)
    reply = next(e for e in events if e["type"] == "reply")
    assert reply["audio_wav_b64"] is None


def test_reply_is_added_to_memory(fakes, monkeypatch):
    call, sessions, _ = make([[turn_ev("hello")]])
    seen = []
    real_end = sessions.end

    def spy(call_id):
        seen.append([t.text for t in sessions.runtime.states[call_id].turns])
        real_end(call_id)

    monkeypatch.setattr(sessions, "end", spy)
    call.feed(b"x")
    drain_until_ended(call)
    assert seen == [["hello", REPLY]]


def test_turns_close_together_are_merged_into_one_reply(fakes):
    call, _, _ = make([[turn_ev("hello", 0, 1)], [turn_ev("again", 1.5, 2.5)]])
    call.feed(b"1")
    call.feed(b"2")
    events = drain_until_ended(call)
    assert types_of(events).count("turn") == 2  # the screen still shows both
    assert [e["turn"] for e in events if e["type"] == "reply"] == [2]
    analyze = next(e for e in events if e["type"] == "node" and e["node"] == "analyze")
    assert analyze["data"]["transcript"] == "Caller: hello again"
    assert [e["turn"] for e in events if e["type"] == "verdict"] == [1, 2]  # no row left pending


def test_each_turn_ends_with_done_after_its_reply(fakes):
    batches = [[turn_ev(f"hello {i}", speaker=f"S{i}")] for i in range(3)]
    call, _, _ = make(batches)
    for _ in batches:
        call.feed(b"x")
    events = drain_until_ended(call)
    done = [e for e in events if e["type"] == "done"]
    assert [e["turn"] for e in done] == [1, 2, 3]
    for e in done:
        reply = next(r for r in events if r["type"] == "reply" and r["turn"] == e["turn"])
        assert events.index(reply) < events.index(e)
