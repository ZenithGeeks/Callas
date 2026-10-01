"""ASR backends. Models load once in __init__, never per request."""

from __future__ import annotations

import logging
import time
from importlib import metadata
from typing import Protocol

import numpy as np
from callas_settings import get_settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class ASRBackend(Protocol):
    name: str
    version: str

    def transcribe(self, audio_path: str) -> str: ...

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str: ...


class PyThaiASR:
    """pythaiasr ASR (Typhoon FastConformer RNN-T ONNX by default; offline after download)."""

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        from pythaiasr import ASR  # heavy import, kept lazy so tests can use fakes

        self.name = model or get_settings().asr_model
        self.version = metadata.version("pythaiasr")
        t0 = time.perf_counter()
        self._asr = ASR(model=self.name, device=device)
        log.info(
            "loaded ASR model=%s pythaiasr=%s in %.2fs",
            self.name,
            self.version,
            time.perf_counter() - t0,
        )

    def transcribe(self, audio_path: str) -> str:
        return str(self._asr(audio_path)).strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        return str(self._asr(audio, sampling_rate=sample_rate)).strip()


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 array at sample_rate (pythaiasr handles mono mixdown and resampling)."""
    from pythaiasr import load_audio as _load

    return _load(audio_path, target_sr=sample_rate)
