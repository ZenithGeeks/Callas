"""Live (microphone) transcription: raw PCM chunks in, partial text and closed turns out.

The pythaiasr StreamingTranscriber.feed_chunk(float32 at 16 kHz) returns only the NEW text decoded
since the last call (committed, never revised), and flush() returns the tail. So the running
text of a turn is the concatenation of those pieces. The ONNX model is loaded once per process
(`_get_model`); each LiveTranscriber only creates a cheap streaming session on top of it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SILENCE_CLOSE_S = 0.8  # this much silence after speech closes the turn
MIN_TURN_S = 0.5  # less speech than this is noise, dropped without a turn event
MAX_TURN_S = 15.0  # a turn this long is closed even while the caller keeps talking
PARTIAL_EVERY_S = 0.5  # at most one partial event per this much audio

# Energy VAD: a frame is speech when its RMS (float32 in [-1, 1]) is at or above this.
# Tune it: print the RMS of room noise and of normal speech from your mic; pick a value about
# 2-3x the noise level and well below speech. Too low: noise opens turns. Too high: quiet
# words are cut. Typical laptop mic: noise ~0.003-0.01, speech ~0.03-0.2.
VAD_RMS_THRESHOLD = 0.015

FRAME_S = 0.02  # VAD frame length
PREROLL_S = 0.3  # silence kept before speech starts so the first syllable is not clipped


class Engine(Protocol):
    def feed_chunk(self, chunk: np.ndarray) -> str: ...
    def flush(self) -> str: ...
    def reset(self) -> None: ...


def pcm16_to_float(b: bytes) -> np.ndarray:
    """Little-endian int16 mono bytes -> float32 in [-1, 1). A trailing odd byte is dropped."""
    n = len(b) // 2 * 2
    return np.frombuffer(b[:n], dtype="<i2").astype(np.float32) / 32768.0


_model = None
_model_lock = threading.Lock()


def _get_model():
    """Load the pythaiasr ONNX model once and share it across all calls."""
    global _model
    with _model_lock:
        if _model is None:
            from pythaiasr import FastConformerRNNT  # heavy import, lazy

            _model = FastConformerRNNT(device="auto")
            log.info("loaded streaming ASR model")
        return _model


def default_engine_factory() -> Engine:
    return _get_model().create_streaming_session()


class LiveTranscriber:
    """Not thread-safe: feed from one thread."""

    def __init__(self, engine_factory: Callable[[], Engine] | None = None) -> None:
        self._engine = (engine_factory or default_engine_factory)()
        self._frame = int(SAMPLE_RATE * FRAME_S)
        self._byte_left = b""  # odd byte carried to the next feed
        self._samples = np.zeros(0, dtype=np.float32)  # less than one frame, carried over
        self._t = 0.0  # seconds of audio consumed (start of the next frame)
        self._preroll: list[np.ndarray] = []
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._open = False
        self._start = 0.0
        self._last_speech_end = 0.0
        self._speech_s = 0.0
        self._silence_s = 0.0
        self._since_partial = 0.0
        self._text = ""
        self._last_partial = ""

    def feed(self, pcm: bytes) -> list[dict]:
        data = self._byte_left + pcm
        self._byte_left = data[len(data) // 2 * 2 :]
        audio = pcm16_to_float(data)
        if len(self._samples):
            audio = np.concatenate([self._samples, audio])
        events: list[dict] = []
        n = len(audio) // self._frame * self._frame
        for i in range(0, n, self._frame):
            self._on_frame(audio[i : i + self._frame], events)
        self._samples = audio[n:]
        return events

    def flush(self) -> list[dict]:
        """Close any open turn (call on end or pause)."""
        events: list[dict] = []
        self._samples = np.zeros(0, dtype=np.float32)
        self._byte_left = b""
        self._preroll.clear()
        if self._open:
            self._close(events)
        return events

    def _on_frame(self, frame: np.ndarray, events: list[dict]) -> None:
        t0, t1 = self._t, self._t + FRAME_S
        self._t = t1
        speech = float(np.sqrt(np.mean(frame * frame))) >= VAD_RMS_THRESHOLD
        if not self._open:
            if not speech:
                self._preroll.append(frame)
                del self._preroll[: -int(PREROLL_S / FRAME_S)]
                return
            self._open = True
            self._start = max(0.0, t0 - len(self._preroll) * FRAME_S)
            for f in self._preroll:
                self._feed_engine(f)
            self._preroll.clear()
        self._feed_engine(frame)
        self._since_partial += FRAME_S
        if speech:
            self._speech_s += FRAME_S
            self._silence_s = 0.0
            self._last_speech_end = t1
        else:
            self._silence_s += FRAME_S
        if self._silence_s >= SILENCE_CLOSE_S or t1 - self._start >= MAX_TURN_S:
            self._close(events)
        elif self._since_partial >= PARTIAL_EVERY_S:
            self._since_partial = 0.0
            if self._text.strip() and self._text != self._last_partial:
                self._last_partial = self._text
                events.append({"type": "partial", "text": self._text.strip()})

    def _feed_engine(self, frame: np.ndarray) -> None:
        piece = self._engine.feed_chunk(frame)
        if piece:  # Thai has no spaces between words, so pieces are joined as they come
            self._text += piece

    def _close(self, events: list[dict]) -> None:
        tail = self._engine.flush()
        if tail:
            self._text += tail
        text = self._text.strip()
        start, end, speech_s = self._start, self._last_speech_end, self._speech_s
        self._engine.reset()
        self._reset_turn()
        if speech_s >= MIN_TURN_S and text:
            events.append(
                {"type": "turn", "speaker": "Caller", "start": start, "end": end, "text": text}
            )
