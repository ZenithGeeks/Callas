"""Thai speech with PyThaiTTS (FastThaiG2P engine). Each Speaker loads the model on first use."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from .facts import fact_numbers
from .leak import leak_reasons


class Speaker:
    """Reads a reply aloud. Voice settings default to the TTS_* settings."""

    def __init__(self, pretrained: str | None = None, speaker: str | None = None) -> None:
        self._pretrained = pretrained
        self._speaker = speaker
        self._tts = None
        # The live page speaks the caller's lines while a turn may be speaking Callas's reply.
        self._lock = threading.Lock()

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

    def synthesize(self, text: str, out_path: str, check_leaks: bool = True) -> str:
        """Write a WAV file and return its path.

        Pass check_leaks=False only for text that is not Callas's own reply, such as a test
        call's caller lines, which are full of numbers on purpose.
        """
        if check_leaks:
            self._refuse_leaky_text(text)
        _, speaker = self._settings()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tts = self._load()
            tts.tts(text, speaker_idx=speaker, return_type="file", filename=str(out_path))
        return str(out_path)

    def synthesize_bytes(self, text: str, check_leaks: bool = True) -> bytes:
        """Return WAV bytes."""
        # synthesize() checks again with a later clock, and the allowed date and time numbers
        # can change in between (minute or day rollover), so neither check covers the other.
        if check_leaks:
            self._refuse_leaky_text(text)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self.synthesize(text, path, check_leaks)
            return Path(path).read_bytes()
        finally:
            Path(path).unlink(missing_ok=True)
