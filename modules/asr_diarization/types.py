"""Shared types for speech-to-text output. Matches data-pipeline's speaker_turns schema."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    """One speaker's text with its start and end."""

    speaker: str  # "A", "B", ...
    start: float  # seconds
    end: float  # seconds
    text: str
    raw: str | None = None  # the ASR text before normalize and autocorrect, if they changed it


@dataclass
class Transcript:
    """The turns of one call and the ASR model that made them."""

    turns: list[Turn] = field(default_factory=list)
    model: str = ""  # which ASR model made this transcript
    model_version: str = ""

    @property
    def text(self) -> str:
        """One line per turn, as "A: text"."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)
