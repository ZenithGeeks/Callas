"""Speak each simulated turn with PyThaiTTS (VachanaTTS2 voices) and join them into one call.

The demo's voice reply uses FastThaiG2P, which has one voice. A call needs two different
voices, so the dataset uses VachanaTTS2: two male and two female voices, picked to match the
sampled genders. The caller and the receiver never get the same voice.

This path does not use tts_voice_agent.Speaker on purpose: Speaker refuses any text with a
number (a leak guard for replies), and dataset calls are full of numbers.
"""

from __future__ import annotations

import random
import threading
import wave
from math import gcd
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # what the ASR and diarizer read


def pick_voices(
    voices: dict[str, list[str]], caller_gender: str, receiver_gender: str, seed_key: str
) -> tuple[str, str]:
    """Voices come from agents.yaml tts.voices; caller and receiver never share one."""
    rng = random.Random(f"voice:{seed_key}")
    caller = rng.choice(voices[caller_gender])
    receiver = rng.choice([v for v in voices[receiver_gender] if v != caller])
    return caller, receiver


def _read_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{path.name}: expected 16-bit audio, got {8 * width}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, rate // divisor)
    return np.clip(samples, -32768, 32767).astype("<i2")


def _write(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


class Voices:
    """speak(text, voice, path) -> seconds of audio. One model, shared by all threads."""

    def __init__(self) -> None:
        self._tts = None
        self._lock = threading.Lock()

    def speak(self, text: str, voice: str, path: Path) -> float:
        with self._lock:  # the wrapper writes temp files and is not documented as thread-safe
            if self._tts is None:
                from pythaitts import TTS

                self._tts = TTS(pretrained="vachana")
            path.parent.mkdir(parents=True, exist_ok=True)
            self._tts.tts(text, speaker_idx=voice, return_type="file", filename=str(path))
        pcm = _read_mono16k(path)
        _write(path, pcm)  # store every turn as 16 kHz mono, like the joined call
        return len(pcm) / SAMPLE_RATE


def join_call(turn_paths: list[Path], out: Path, gap_s: float) -> list[tuple[float, float]]:
    """Join turn WAVs with gap_s of silence between; return (start, end) seconds of each turn."""
    gap = np.zeros(int(gap_s * SAMPLE_RATE), dtype="<i2")
    parts, times, now = [], [], 0.0
    for i, turn_path in enumerate(turn_paths):
        if i:
            parts.append(gap)
            now += gap_s
        pcm = _read_mono16k(turn_path)
        parts.append(pcm)
        times.append((round(now, 3), round(now + len(pcm) / SAMPLE_RATE, 3)))
        now += len(pcm) / SAMPLE_RATE
    _write(out, np.concatenate(parts) if parts else np.zeros(0, dtype="<i2"))
    return times
