import pytest

from tts_voice_agent.speech import Speaker


class Boom(Speaker):
    def _load(self):
        raise AssertionError("model must not load before the leak check")


@pytest.mark.parametrize("text", ["โทร 0812345678", "ศูนย์แปดหนึ่ง", "รหัส OTP"])
def test_speaker_refuses_leaky_text_without_loading_model(text, tmp_path):
    sp = Boom()
    with pytest.raises(ValueError):
        sp.synthesize(text, str(tmp_path / "x.wav"))
    with pytest.raises(ValueError):
        sp.synthesize_bytes(text)


def test_model_loaded_lazily():
    assert Speaker(pretrained="fastthaig2p", speaker="thai_som")._tts is None
