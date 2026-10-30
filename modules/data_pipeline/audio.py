"""Speak each turn as it is written, and join the turns into one WAV for the whole call.

Paths are relative to the store's folder: audio/<id>/NN_<speaker>.wav per turn and
audio/<id>.wav for the call.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .records import CallRecord, Turn
from .tts import Voices, join_call

log = logging.getLogger("data_pipeline")


def speak_turn(voices: Voices, turn: Turn, index: int, record: CallRecord, folder: Path) -> Turn:
    """The turn with its audio file and length. A turn with no text, or one whose speech
    failed, is returned as it is: the call is kept for its text either way.

    `record` is the stored call, for its id and the voice of each speaker.
    """
    if not turn.text:
        return turn
    said = turn.spoken_text or turn.text
    rel = f"audio/{record.id}/{index:02d}_{turn.speaker}.wav"
    voice = record.generation.voices[turn.speaker]
    try:
        seconds = round(voices.speak(said, voice, folder / rel), 3)
    except Exception as exc:  # the dataset is the text: a turn without audio does not reject it
        log.warning("tts failed for %s turn %d: %s", record.id, index, type(exc).__name__)
        return turn
    return replace(turn, audio=rel, audio_s=seconds)


def join_audio(
    turns: list[Turn], record_id: str, folder: Path, gap_s: float
) -> tuple[str, list[tuple[float, float]]] | None:
    """Join the spoken turns into one file. Returns its path and each turn's (start, end) in
    it, or None when a turn has no audio."""
    if not turns or any(turn.audio is None for turn in turns):
        return None
    rel = f"audio/{record_id}.wav"
    times = join_call([folder / turn.audio for turn in turns], folder / rel, gap_s)
    return rel, times
