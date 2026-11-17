"""Live (microphone) transcription: raw PCM chunks in, partial text and closed turns out.

Read LiveTranscriber first. An engine's feed_chunk(float32 at 16 kHz) returns only the NEW text
since the last call (never revised), and flush() returns the tail, so the text of a turn is the
pieces joined. The ONNX model loads once per process (_get_model); each LiveTranscriber creates
a cheap streaming session on top of it. ASR_MODEL must name a Typhoon FastConformer or a Typhoon
Nemotron model; any other is refused with a message that says so.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Literal, Protocol, TypedDict

import numpy as np

from asr_diarization.backends import SAMPLE_RATE

log = logging.getLogger(__name__)

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
PREROLL_FRAMES = int(PREROLL_S / FRAME_S)
# Silence the engine gets before a turn's first speech, the pre-roll included. Typhoon Nemotron
# drops the first syllable of speech that starts within its first chunks after reset(): with no
# lead-in it heard "ฮัลโหล" as "โหล" and lost "เป็น" and "แต่"; 0.5 s brought them back. A turn
# right after resume() has no pre-roll at all, and a TTS line has no silence of its own.
ENGINE_LEAD_IN_FRAMES = int(0.5 / FRAME_S)

# Every pythaiasr model that can stream is a Typhoon FastConformer or a Typhoon Nemotron, and all
# their names contain one of these. Checking the name first means a Whisper or Wav2Vec2 setting
# fails with a clear message instead of trying to load torch (device="auto" is not a torch device).
STREAMING_NAME_PARTS = ("typhoon", "nemotron")
CANNOT_STREAM = "ASR_MODEL {name!r} cannot stream. Use typhoon_asr or typhoon_nemotron_asr."

_model = None
_model_lock = threading.Lock()


class PartialEvent(TypedDict):
    """Text so far of the turn that is still open."""

    type: Literal["partial"]
    text: str


class TurnEvent(TypedDict):
    """A finished turn. start and end are seconds from the first sample fed."""

    type: Literal["turn"]
    speaker: str
    start: float
    end: float
    text: str


class Engine(Protocol):
    """A streaming ASR session: feed audio, get new text back."""

    def feed_chunk(self, chunk: np.ndarray) -> str: ...
    def flush(self) -> str: ...
    def reset(self) -> None: ...


class LiveTranscriber:
    """Turns a stream of 16 kHz int16 PCM into partial and turn events. Feed from one thread."""

    def __init__(self, engine_factory: Callable[[], Engine] | None = None) -> None:
        self._engine = (engine_factory or default_engine_factory)()
        self._frame_len = int(SAMPLE_RATE * FRAME_S)
        self._odd_byte = b""  # odd byte carried to the next feed
        self._leftover = np.zeros(0, dtype=np.float32)  # less than one frame, carried over
        self._audio_s = 0.0  # seconds of audio consumed (start of the next frame)
        self._preroll: list[np.ndarray] = []
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._turn_open = False
        self._turn_start = 0.0
        self._last_speech_end = 0.0
        self._speech_s = 0.0
        self._silence_s = 0.0
        self._since_partial_s = 0.0
        self._turn_text = ""
        self._last_partial_text = ""

    def feed(self, pcm: bytes) -> list[PartialEvent | TurnEvent]:
        """Take the next bytes of audio; return the events they caused, in order."""
        data = self._odd_byte + pcm
        self._odd_byte = data[len(data) // 2 * 2 :]
        audio = pcm16_to_float(data)
        if len(self._leftover):
            audio = np.concatenate([self._leftover, audio])
        events: list[PartialEvent | TurnEvent] = []
        whole_frames_end = len(audio) // self._frame_len * self._frame_len
        for i in range(0, whole_frames_end, self._frame_len):
            self._on_frame(audio[i : i + self._frame_len], events)
        self._leftover = audio[whole_frames_end:]
        return events

    def flush(self) -> list[PartialEvent | TurnEvent]:
        """Close any open turn (call on end or pause)."""
        events: list[PartialEvent | TurnEvent] = []
        self._leftover = np.zeros(0, dtype=np.float32)
        self._odd_byte = b""
        self._preroll.clear()
        if self._turn_open:
            self._close(events)
        return events

    def _on_frame(self, frame: np.ndarray, events: list[PartialEvent | TurnEvent]) -> None:
        """Handle one 20 ms frame: open a turn on speech, close it after silence or at the
        maximum length, and add a partial event every PARTIAL_EVERY_S."""
        frame_start, frame_end = self._audio_s, self._audio_s + FRAME_S
        self._audio_s = frame_end
        is_speech = float(np.sqrt(np.mean(frame * frame))) >= VAD_RMS_THRESHOLD
        if not self._turn_open:
            if not is_speech:
                self._keep_preroll(frame)
                return
            self._open_turn(frame_start)
        self._feed_engine(frame)
        self._since_partial_s += FRAME_S
        if is_speech:
            self._speech_s += FRAME_S
            self._silence_s = 0.0
            self._last_speech_end = frame_end
        else:
            self._silence_s += FRAME_S
        if self._silence_s >= SILENCE_CLOSE_S or frame_end - self._turn_start >= MAX_TURN_S:
            self._close(events)
        elif self._since_partial_s >= PARTIAL_EVERY_S:
            self._emit_partial(events)

    def _keep_preroll(self, silent_frame: np.ndarray) -> None:
        self._preroll.append(silent_frame)
        del self._preroll[:-PREROLL_FRAMES]

    def _open_turn(self, frame_start: float) -> None:
        """Start a turn at the first speech frame, replaying the kept silence into the engine."""
        self._turn_open = True
        self._turn_start = max(0.0, frame_start - len(self._preroll) * FRAME_S)
        silence = np.zeros(self._frame_len, dtype=np.float32)
        for _ in range(ENGINE_LEAD_IN_FRAMES - len(self._preroll)):
            self._feed_engine(silence)
        for frame in self._preroll:
            self._feed_engine(frame)
        self._preroll.clear()

    def _emit_partial(self, events: list[PartialEvent | TurnEvent]) -> None:
        """Add a partial event unless the text is empty or the same as the last partial."""
        self._since_partial_s = 0.0
        text = self._turn_text.strip()
        if text and self._turn_text != self._last_partial_text:
            self._last_partial_text = self._turn_text
            events.append({"type": "partial", "text": text})

    def _feed_engine(self, frame: np.ndarray) -> None:
        piece = self._engine.feed_chunk(frame)
        if piece:  # Thai has no spaces between words, so pieces are joined as they come
            self._turn_text += piece

    def _close(self, events: list[PartialEvent | TurnEvent]) -> None:
        tail = self._engine.flush()
        if tail:
            self._turn_text += tail
        text = self._turn_text.strip()
        start, end, speech_s = self._turn_start, self._last_speech_end, self._speech_s
        self._engine.reset()
        self._reset_turn()
        if speech_s >= MIN_TURN_S and text:
            events.append(
                {"type": "turn", "speaker": "Caller", "start": start, "end": end, "text": text}
            )


def pcm16_to_float(pcm: bytes) -> np.ndarray:
    """Little-endian int16 mono bytes -> float32 in [-1, 1). A trailing odd byte is dropped."""
    n = len(pcm) // 2 * 2
    return np.frombuffer(pcm[:n], dtype="<i2").astype(np.float32) / 32768.0


def default_engine_factory() -> Engine:
    """A new streaming session on the shared model, whichever kind ASR_MODEL names."""
    model = _get_model()
    if hasattr(model, "create_streaming_session"):  # Typhoon FastConformer
        return model.create_streaming_session()
    from pythaiasr.typhoon_nemotron_asr import NemotronStreamingASR

    if isinstance(model, NemotronStreamingASR):
        return _NemotronEngine(model)
    raise RuntimeError(CANNOT_STREAM.format(name=type(model).__name__))


def _get_model():
    """Load the ASR_MODEL pythaiasr ONNX model once and share it across all calls."""
    global _model
    with _model_lock:
        if _model is None:
            _model = _load_streaming_model()
        return _model


def _load_streaming_model():
    from callas_settings import get_settings

    name = get_settings().asr_model
    if not any(part in name.lower() for part in STREAMING_NAME_PARTS):
        raise RuntimeError(CANNOT_STREAM.format(name=name))
    from pythaiasr import ASR  # heavy import, lazy

    try:
        model = ASR(model=name, device="auto").model
    except (AssertionError, ImportError) as exc:  # unknown name, missing extras
        raise RuntimeError(f"{CANNOT_STREAM.format(name=name)} ({exc})") from exc
    log.info("loaded streaming ASR model=%s", name)
    return model


class _NemotronEngine:
    """Engine view of pythaiasr's Nemotron 3.5 stream, which names its feed process_chunk."""

    def __init__(self, model) -> None:
        from pythaiasr.typhoon_nemotron_asr import NemotronStreamASR

        self._stream = NemotronStreamASR(model, sample_rate=SAMPLE_RATE)

    def feed_chunk(self, chunk: np.ndarray) -> str:
        return self._stream.process_chunk(chunk)

    def flush(self) -> str:
        return self._stream.flush()

    def reset(self) -> None:
        self._stream.reset()
