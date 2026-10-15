"""Thai speech with PyThaiTTS (FastThaiG2P engine). Each Speaker loads the model on first use."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .facts import fact_numbers
from .leak import leak_reasons


class Speaker:
    """Reads a reply aloud. Voice settings default to the TTS_* settings."""

    def __init__(self, pretrained: str | None = None, speaker: str | None = None) -> None:
        self._pretrained = pretrained
        self._speaker = speaker
        self._tts = None

    def _settings(self) -> tuple[str, str]:
        """(pretrained model, speaker): the constructor values, else the app settings."""
        if self._pretrained and self._speaker:
            return self._pretrained, self._speaker
        from callas_settings import get_settings

        s = get_settings()
        return self._pretrained or s.tts_pretrained, self._speaker or s.tts_speaker

    def _load(self):
        if self._tts is None:
            from pythaitts import TTS

            pretrained, _ = self._settings()
            self._tts = TTS(pretrained=pretrained)
        return self._tts

    @staticmethod
    def _refuse_leaky_text(text: str) -> None:
        # Defense in depth: replies are already checked, but never speak a number or secret.
        if leak_reasons(text, [], fact_numbers()):
            raise ValueError("refusing to synthesize text that fails the leak check")

    def synthesize(self, text: str, out_path: str) -> str:
        """Write a WAV file and return its path."""
        self._refuse_leaky_text(text)
        tts = self._load()
        _, speaker = self._settings()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tts.tts(text, speaker_idx=speaker, return_type="file", filename=str(out_path))
        return str(out_path)

    def synthesize_bytes(self, text: str) -> bytes:
        """Return WAV bytes."""
        self._refuse_leaky_text(text)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self.synthesize(text, path)
            return Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)
