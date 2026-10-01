import json

import numpy as np
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


def test_main_writes_json(tmp_path, monkeypatch, capsys):
    backend = FakeBackend(["ทดสอบ"])
    monkeypatch.setattr(
        pipeline, "get_default_pipeline", lambda: (backend, lambda _p: [Segment("A", 0, 1)])
    )
    import asr_diarization.backends as b

    monkeypatch.setattr(b, "load_audio", loader(2))
    out = tmp_path / "o.json"
    assert pipeline.main(["--audio", "x.wav", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["speaker_turns"][0]["text"] == "ทดสอบ"
    assert "A: ทดสอบ" in capsys.readouterr().out
