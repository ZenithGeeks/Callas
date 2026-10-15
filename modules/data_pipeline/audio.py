"""Speak each turn as it is written, and join the turns into one WAV for the whole call."""

from __future__ import annotations

import logging
from pathlib import Path

from .tts import Voices, join_call

log = logging.getLogger("data_pipeline")


def speak_turn(voices: Voices, record: dict, index: int, out_dir: Path) -> None:
    turn = record["generation"]["turns"][index]
    if not turn["text"]:
        return
    said = turn.get("spoken_text") or turn["text"]
    rel = f"audio/{record['id']}/{index:02d}_{turn['speaker']}.wav"
    voice = record["generation"]["voices"][turn["speaker"]]
    try:
        turn["audio_s"] = round(voices.speak(said, voice, out_dir / rel), 3)
        turn["audio"] = rel
    except Exception as exc:  # a turn without audio is flagged, the call text is still kept
        log.warning("tts failed for %s turn %d: %s", record["id"], index, type(exc).__name__)
        turn["problems"].append("tts_failed")


def join_audio(record: dict, out_dir: Path, gap_s: float) -> None:
    turns = record["generation"]["turns"]
    if not turns or any(turn["audio"] is None for turn in turns):
        return
    rel = f"audio/{record['id']}.wav"
    times = join_call([out_dir / turn["audio"] for turn in turns], out_dir / rel, gap_s)
    record["audio_path"] = rel
    for speaker_turn, (start, end) in zip(record["speaker_turns"], times, strict=True):
        speaker_turn["start"], speaker_turn["end"] = start, end
