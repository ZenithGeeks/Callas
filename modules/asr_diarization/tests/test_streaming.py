import numpy as np

from asr_diarization import streaming
from asr_diarization.streaming import (
    MAX_TURN_S,
    SAMPLE_RATE,
    SILENCE_CLOSE_S,
    LiveTranscriber,
    pcm16_to_float,
)


class FakeEngine:
    """Returns one scripted word per `every` fed chunks (20 ms each), and a tail on flush."""

    def __init__(self, words=("a", "b", "c"), every=25, tail="end"):
        self.words, self.every, self.tail = list(words), every, tail
        self.n = 0
        self.resets = 0

    def feed_chunk(self, chunk):
        self.n += 1
        if self.words and self.n % self.every == 0:
            return self.words[(self.n // self.every - 1) % len(self.words)]
        return ""

    def flush(self):
        return self.tail

    def reset(self):
        self.n = 0
        self.resets += 1


def pcm(seconds, loud):
    n = int(SAMPLE_RATE * seconds)
    if loud:
        t = np.arange(n) / SAMPLE_RATE
        x = 0.3 * np.sin(2 * np.pi * 220 * t)
    else:
        x = np.zeros(n)
    return (x * 32767).astype("<i2").tobytes()


def run(tr, *parts, chunk_s=0.1):
    """Feed (seconds, loud) parts in chunk_s pieces and collect events."""
    events = []
    data = b"".join(pcm(s, loud) for s, loud in parts)
    step = int(SAMPLE_RATE * chunk_s) * 2
    for i in range(0, len(data), step):
        events += tr.feed(data[i : i + step])
    return events


def make(**kw):
    eng = FakeEngine(**kw)
    return LiveTranscriber(lambda: eng), eng


def turns(events):
    return [e for e in events if e["type"] == "turn"]


def test_pcm16_round_trip():
    samples = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
    out = pcm16_to_float(samples.tobytes())
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, samples / 32768.0, atol=1e-6)
    assert pcm16_to_float(b"").shape == (0,)


def test_pcm16_odd_bytes_are_buffered_by_transcriber():
    tr, _ = make()
    data = pcm(1.0, True) + pcm(1.0, False)
    a = tr.feed(data[:1001])  # odd split
    b = tr.feed(data[1001:])
    split = a + b
    tr2, _ = make()
    assert split == tr2.feed(data)


def test_speech_then_silence_gives_partials_then_one_turn():
    tr, eng = make()
    events = run(tr, (2.0, True), (1.0, False))
    kinds = [e["type"] for e in events]
    assert kinds.count("turn") == 1 and kinds[-1] == "turn"
    assert kinds.count("partial") >= 1
    t = turns(events)[0]
    assert t["speaker"] == "Caller" and t["text"].startswith("abc") and t["text"].endswith("end")
    assert abs(t["start"] - 0.0) < 0.05 and abs(t["end"] - 2.0) < 0.05
    assert eng.resets == 1


def test_partials_are_rate_limited():
    tr, _ = make(every=5)
    events = run(tr, (3.0, True))
    assert 1 <= len([e for e in events if e["type"] == "partial"]) <= 6


def test_leading_silence_gives_no_events():
    tr, _ = make()
    assert run(tr, (3.0, False)) == []


def test_two_bursts_two_turns_with_increasing_times():
    tr, _ = make()
    ev = turns(run(tr, (1.0, False), (1.5, True), (1.2, False), (1.5, True), (1.2, False)))
    assert len(ev) == 2
    assert ev[0]["end"] <= ev[1]["start"]
    assert ev[0]["start"] < ev[1]["start"] and ev[0]["end"] < ev[1]["end"]
    assert abs(ev[0]["start"] - 0.7) < 0.05  # 1.0 s silence minus 0.3 s pre-roll


def test_max_turn_forces_close():
    tr, _ = make()
    ev = turns(run(tr, (MAX_TURN_S + 3.0, True)))
    assert len(ev) == 1
    assert abs(ev[0]["end"] - MAX_TURN_S) < 0.1


def test_empty_text_gives_no_turn():
    tr, _ = make(words=(), tail="  ")
    assert turns(run(tr, (2.0, True), (SILENCE_CLOSE_S + 0.3, False))) == []


def test_short_blip_is_dropped():
    tr, _ = make()
    assert turns(run(tr, (0.2, True), (1.5, False))) == []


def test_flush_closes_open_turn():
    tr, _ = make()
    events = run(tr, (1.5, True))
    assert turns(events) == []
    ev = turns(tr.flush())
    assert len(ev) == 1 and ev[0]["text"].endswith("end")
    assert tr.flush() == []


def test_model_loaded_once(monkeypatch):
    calls = []

    class Model:
        def create_streaming_session(self):
            return FakeEngine()

    monkeypatch.setattr(streaming, "_model", None)
    import types

    fake = types.SimpleNamespace(FastConformerRNNT=lambda device: calls.append(device) or Model())
    monkeypatch.setitem(__import__("sys").modules, "pythaiasr", fake)
    LiveTranscriber()
    LiveTranscriber()
    assert calls == ["auto"]
    monkeypatch.setattr(streaming, "_model", None)
