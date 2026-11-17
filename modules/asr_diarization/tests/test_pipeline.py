import json
from pathlib import Path

import numpy as np
import pytest

from asr_diarization import pipeline
from asr_diarization.diarize import Segment, label_for, relabel_and_merge

SR = 16_000


class FakeBackend:
    name = "fake_asr"
    version = "9.9"

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def transcribe(self, audio_path):
        return "whole"

    def transcribe_array(self, audio, sample_rate=SR):
        self.calls.append(len(audio))
        return self.texts.pop(0)


def loader(seconds=10):
    return lambda _p: np.zeros(int(seconds * SR), dtype=np.float32)


def test_label_for():
    assert [label_for(i) for i in (0, 1, 25, 26, 27)] == ["A", "B", "Z", "AA", "AB"]


def test_relabel_first_appearance_and_merge():
    raw = [
        {"speaker": "SPEAKER_01", "start": 3.0, "end": 4.0},
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
        {"speaker": "SPEAKER_00", "start": 1.2, "end": 2.0},  # gap 0.2 -> merged
        {"speaker": "SPEAKER_01", "start": 2.5, "end": 2.9},
    ]
    segs = relabel_and_merge(raw)
    assert segs == [Segment("A", 0.0, 2.0), Segment("B", 2.5, 4.0)]


def test_no_merge_over_large_gap():
    raw = [
        {"speaker": "x", "start": 0, "end": 1},
        {"speaker": "x", "start": 5, "end": 6},
    ]
    assert len(relabel_and_merge(raw)) == 2


def test_pipeline_shape_and_empty_dropped():
    backend = FakeBackend(["สวัสดี", "  ", "ครับ"])
    segs = [Segment("A", 0, 2), Segment("B", 2, 4), Segment("A", 4, 6)]
    t = pipeline.run("x.wav", backend, lambda _p: segs, loader())
    assert t.model == "fake_asr" and t.model_version == "9.9"
    assert [(x.speaker, x.start, x.end, x.text) for x in t.turns] == [
        ("A", 0, 2, "สวัสดี"),
        ("A", 4, 6, "ครับ"),
    ]
    assert backend.calls == [2 * SR] * 3


def test_no_segments_fallback(caplog):
    backend = FakeBackend(["ทั้งไฟล์"])
    with caplog.at_level("WARNING"):
        t = pipeline.run("x.wav", backend, lambda _p: [], loader(3))
    assert [(x.speaker, x.start, x.end, x.text) for x in t.turns] == [("A", 0.0, 3.0, "ทั้งไฟล์")]
    assert "no segments" in caplog.text
    assert t.model == "fake_asr"


def test_fallback_turn_rounds_end_like_segment_turns():
    backend = FakeBackend(["ทั้งไฟล์"])

    def audio_loader(_path):
        return np.zeros(16_001, dtype=np.float32)  # 1.0000625 s

    t = pipeline.run("x.wav", backend, lambda _p: [], audio_loader)
    assert t.turns[0].end == 1.0


def test_only_the_missing_dependency_is_built(monkeypatch):
    def no_real_model():
        raise AssertionError("the real ASR model must not be built")

    monkeypatch.setattr(pipeline, "get_default_backend", no_real_model)
    monkeypatch.setattr(pipeline, "diarize", lambda _p: [Segment("A", 0, 1)])
    backend = FakeBackend(["ครับ"])
    t = pipeline.run("x.wav", backend=backend, audio_loader=loader(2))
    assert [x.text for x in t.turns] == ["ครับ"]


def test_missing_backend_gets_the_default_one(monkeypatch):
    backend = FakeBackend(["ค่ะ"])
    monkeypatch.setattr(pipeline, "get_default_backend", lambda: backend)
    t = pipeline.run("x.wav", diarizer=lambda _p: [Segment("A", 0, 1)], audio_loader=loader(2))
    assert t.model == "fake_asr"


def test_zero_length_segment_skipped():
    backend = FakeBackend([])
    t = pipeline.run("x.wav", backend, lambda _p: [Segment("A", 1, 1)], loader())
    assert t.turns == []


def test_json_output(tmp_path):
    backend = FakeBackend(["สวัสดีค่ะ"])
    t = pipeline.run("x.wav", backend, lambda _p: [Segment("A", 0, 1)], loader(2))
    out = tmp_path / "t.json"
    pipeline.write_json(t, str(out))
    raw = out.read_bytes().decode("utf-8")
    assert "สวัสดีค่ะ" in raw  # ensure_ascii=False
    assert json.loads(raw) == {
        "model": "fake_asr",
        "model_version": "9.9",
        "speaker_turns": [{"speaker": "A", "start": 0, "end": 1, "text": "สวัสดีค่ะ"}],
    }


def write_m4a(path, seconds=1.5, rate=44_100):
    """A real AAC .m4a with a 440 Hz tone, the format phones record."""
    av = pytest.importorskip("av")
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("aac", rate=rate, layout="mono")
        for start in range(0, len(samples), 1024):
            frame = av.AudioFrame.from_ndarray(
                samples[None, start : start + 1024], format="flt", layout="mono"
            )
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def test_m4a_is_decoded_to_a_wav_both_steps_read(tmp_path):
    sf = pytest.importorskip("soundfile")
    m4a = tmp_path / "Test 1.m4a"
    write_m4a(m4a)
    seen = []

    def diarizer(path):
        seen.append(path)
        return [Segment("A", 0, 1)]

    def audio_loader(path):
        seen.append(path)
        audio, rate = sf.read(path, dtype="float32")  # libsndfile alone cannot open the .m4a
        assert rate == SR
        return audio

    backend = FakeBackend(["ฮัลโหล"])
    t = pipeline.run(str(m4a), backend, diarizer, audio_loader)
    assert [x.text for x in t.turns] == ["ฮัลโหล"]
    assert len(set(seen)) == 1 and seen[0].endswith(".wav")
    assert abs(backend.calls[0] - SR) < SR // 10  # the first second of 16 kHz audio
    assert not Path(seen[0]).exists()  # the decoded copy is deleted afterwards
    assert m4a.exists()  # the upload itself is left to its owner


def test_other_formats_are_read_as_they_are():
    seen = []
    diarizer = lambda p: seen.append(p) or [Segment("A", 0, 1)]  # noqa: E731
    pipeline.run("x.mp3", FakeBackend(["ครับ"]), diarizer, loader(1))
    assert seen == ["x.mp3"]


def test_main_writes_json(tmp_path, monkeypatch, capsys):
    backend = FakeBackend(["ทดสอบ"])
    monkeypatch.setattr(pipeline, "get_default_backend", lambda: backend)
    monkeypatch.setattr(pipeline, "diarize", lambda _p: [Segment("A", 0, 1)])
    import asr_diarization.backends as b

    monkeypatch.setattr(b, "load_audio", loader(2))
    out = tmp_path / "o.json"
    assert pipeline.main(["--audio", "x.wav", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["speaker_turns"][0]["text"] == "ทดสอบ"
    assert "A: ทดสอบ" in capsys.readouterr().out
