"""Shared types for speech-to-text output. Matches data-pipeline's speaker_turns schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    speaker: str  # "A", "B", ...
    start: float  # seconds
    end: float  # seconds
    text: str


@dataclass
class Transcript:
    turns: list[Turn] = field(default_factory=list)
    model: str = ""  # which ASR model made this transcript
    model_version: str = ""

    @property
    def text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)
