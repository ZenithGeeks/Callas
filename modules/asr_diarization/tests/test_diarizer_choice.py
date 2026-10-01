import pytest
from pydantic import ValidationError

import callas_settings
from asr_diarization import diarize as dz


class FakeDiarization:
    loads: list[str] = []

    def __init__(self, model="nemotron-3-diarization", **kwargs):
        self.model = model
        FakeDiarization.loads.append(model)

    def diarize(self, data, **kwargs):
        return [
            {"speaker": "S1", "start": 0.0, "end": 1.0},
            {"speaker": "S0", "start": 1.2, "end": 2.0},
            {"speaker": "S0", "start": 2.3, "end": 3.0},
        ]


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    import pythaiasr

    FakeDiarization.loads = []
    dz._cache.clear()
    monkeypatch.setattr(pythaiasr, "Diarization", FakeDiarization)
    yield
    dz._cache.clear()
    callas_settings.get_settings.cache_clear()


def use_setting(monkeypatch, value):
    monkeypatch.setenv("DIARIZER", value)
    callas_settings.get_settings.cache_clear()


def test_setting_default_and_validation(monkeypatch):
    monkeypatch.delenv("DIARIZER", raising=False)
    assert callas_settings.Settings(_env_file=None).diarizer == "nemotron"
    assert callas_settings.Settings(_env_file=None, diarizer=" Nemotron ").diarizer == "nemotron"
    with pytest.raises(ValidationError, match="DIARIZER must be one of"):
        callas_settings.Settings(_env_file=None, diarizer="whisper")


@pytest.mark.parametrize(
    ("choice", "expected"),
    [("nemotron", "nemotron-3-diarization"), ("pyannote_segmentation", "pyannote_segmentation")],
)
def test_choice_reaches_pythaiasr(monkeypatch, choice, expected):
    use_setting(monkeypatch, choice)
    dz.diarize("x.wav")
    assert FakeDiarization.loads == [expected]


def test_model_argument_overrides_setting(monkeypatch):
    use_setting(monkeypatch, "nemotron")
    dz.diarize("x.wav", model="pyannote_segmentation")
    assert FakeDiarization.loads == ["pyannote_segmentation"]


def test_model_loaded_once_across_calls(monkeypatch):
    use_setting(monkeypatch, "nemotron")
    dz.diarize("a.wav")
    dz.diarize("b.wav")
    assert FakeDiarization.loads == ["nemotron-3-diarization"]
    dz.diarize("c.wav", model="pyannote_segmentation")
    dz.diarize("d.wav")
    assert FakeDiarization.loads == ["nemotron-3-diarization", "pyannote_segmentation"]


def test_output_contract_relabel_and_merge(monkeypatch):
    use_setting(monkeypatch, "nemotron")
    segs = dz.diarize("x.wav")
    assert [(s.speaker, s.start, s.end) for s in segs] == [("A", 0.0, 1.0), ("B", 1.2, 3.0)]
