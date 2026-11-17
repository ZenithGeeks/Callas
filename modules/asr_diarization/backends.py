"""ASR backends. Models load once in __init__, never per request."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Protocol

import numpy as np

from callas_settings import get_settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class ASRBackend(Protocol):
    """What the pipeline needs from an ASR engine: names for the transcript and two ways in."""

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
        started = time.perf_counter()
        self._asr = ASR(model=self.name, device=device)
        log.info(
            "loaded ASR model=%s pythaiasr=%s in %.2fs",
            self.name,
            self.version,
            time.perf_counter() - started,
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a whole audio file."""
        return str(self._asr(audio_path)).strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Transcribe mono samples that are already in memory."""
        audio = np.asarray(audio, dtype=np.float32)
        return str(self._asr(audio, sampling_rate=sample_rate)).strip()


# Containers libsndfile cannot open (AAC audio, as phones record .m4a). They are decoded once
# with PyAV, which bundles its own FFmpeg, into a WAV that every later step can read.
NEEDS_DECODING = (".m4a", ".mp4", ".aac")


@contextmanager
def readable(audio_path: str, sample_rate: int = SAMPLE_RATE) -> Iterator[str]:
    """A path every step can read: `audio_path` itself, or for a NEEDS_DECODING file a temporary
    16 kHz mono WAV of it, deleted when the block ends."""
    if Path(audio_path).suffix.lower() not in NEEDS_DECODING:
        yield audio_path
        return
    import soundfile as sf

    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="callas-decoded-")
    os.close(fd)
    try:
        sf.write(wav_path, decode(audio_path, sample_rate), sample_rate, subtype="PCM_16")
        yield wav_path
    finally:
        Path(wav_path).unlink(missing_ok=True)


def decode(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Every audio stream frame of the file as mono float32 at sample_rate, with PyAV."""
    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            f"Reading {Path(audio_path).suffix} files needs PyAV: pip install -e .[asr]"
        ) from exc
    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    with av.open(audio_path) as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]  # flush
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 array at sample_rate (pythaiasr handles mono mixdown and resampling)."""
    from pythaiasr import load_audio as load_with_pythaiasr

    return load_with_pythaiasr(audio_path, target_sr=sample_rate)
