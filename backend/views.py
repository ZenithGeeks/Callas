"""Presentation helpers: turn pipeline output into what the templates draw.

Nothing here decides anything about a call or message. It only formats and names things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callas_settings import get_settings

DIARIZER_LABELS = {
    "nemotron": "Nemotron-3 diarization",
    "pyannote_segmentation": "pyannote segmentation 3.0",
}

SENSITIVE_LABELS = frozenset(
    {"PHONE", "BANK_ACCOUNT", "OTP", "MONEY", "URL", "ID_CARD", "BIRTH_DATE", "ADDRESS"}
)
# In a recording, speaker A is the person being called. Generated calls in the dataset use
# the opposite letters: see dataset.GENERATED_CALLER.
CALLED_PERSON = "A"
HIGH_CONFIDENCE = 0.7  # same number as tts_voice_agent.constants.ESCALATE_THRESHOLD

TYPE_NAMES: dict[str | None, str] = {
    "fake_authority": "Fake authority / call center",
    "phishing_otp": "Phishing / OTP",
    "romance": "Romance scam",
    "investment": "Investment scam",
    "lottery": "Lottery scam",
    None: "Type not known",
}

MESSAGE_INSTRUCTIONS: dict[str | None, str] = {
    "phishing_otp": "Do not open the link. Do not send the code.",
    "fake_authority": "Do not transfer money. Call the agency yourself on its public number.",
    "romance": "Do not send money to someone you have not met.",
    "investment": "Do not transfer money. Check the company with the regulator first.",
    "lottery": "Do not pay any fee to claim a prize.",
    None: "Do not reply, open links or send anything.",
}

LABEL_TEXT = {"BANK_ACCOUNT": "BANK ACCOUNT", "ID_CARD": "ID CARD", "BIRTH_DATE": "BIRTH DATE"}


def type_name(label_type: str | None) -> str:
    """The scam type in words, "Type not known" for None or a type we have no name for."""
    return TYPE_NAMES.get(label_type, TYPE_NAMES[None])


def confidence_word(confidence: float) -> str:
    """High from HIGH_CONFIDENCE up, Low for any other positive score, else Not known."""
    if confidence >= HIGH_CONFIDENCE:
        return "High"
    if confidence > 0:
        return "Low"
    return "Not known"


def fmt_time(seconds: float) -> str:
    """Minutes and seconds, e.g. 1:05."""
    whole = max(0, int(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def fmt_seconds(seconds: float) -> str:
    """A duration with one decimal, e.g. 0.4 s."""
    return f"{seconds:.1f} s"


def diarizer_label() -> str:
    """Name of the speaker-split model the DIARIZER setting picks."""
    return DIARIZER_LABELS[get_settings().diarizer]


def who(speaker: str) -> str:
    """Name a speaker of a recording: "You" for the person called, "Caller" for the other."""
    return "You" if speaker == CALLED_PERSON else "Caller"


@dataclass
class Seg:
    """A piece of text, plain (label None) or marked as one entity."""

    text: str
    label: str | None = None

    @property
    def tone(self) -> str:
        """Colour class for an entity piece: alarm for sensitive labels, ink for the rest."""
        if self.label is None:
            return ""
        return "alarm" if self.label in SENSITIVE_LABELS else "ink"


def _field(entity: Any, name: str) -> Any:
    """Read a field from an entity that is either a dict or an object."""
    return entity[name] if isinstance(entity, dict) else getattr(entity, name)


def segments(text: str, entities: list[Any]) -> list[Seg]:
    """Split text into plain and entity pieces.

    An overlapping entity is dropped (the earlier piece wins).
    """
    pieces: list[Seg] = []
    pos = 0
    for start, end, label in _entity_spans(text, entities):
        if start < pos:
            continue
        if start > pos:
            pieces.append(Seg(text[pos:start]))
        pieces.append(Seg(text[start:end], label))
        pos = end
    if pos < len(text):
        pieces.append(Seg(text[pos:]))
    return pieces or [Seg(text)]


def _entity_spans(text: str, entities: list[Any]) -> list[tuple[int, int, str]]:
    """(start, end, label) of each entity, sorted by start.

    An entity is placed by its offsets when they still match its text, otherwise by searching
    for its text. An entity whose text is not found is left out.
    """
    spans = []
    for entity in entities:
        value, label = _field(entity, "text"), _field(entity, "label")
        if not value:
            continue
        start, end = _field(entity, "start"), _field(entity, "end")
        if text[start:end] != value:
            start = text.find(value)
            end = start + len(value)
        if start < 0:
            continue
        spans.append((start, end, label))
    return sorted(spans)


def entity_rows(entities: list[Any]) -> list[dict]:
    """One row per distinct (text, label), in order of first appearance."""
    rows, seen = [], set()
    for e in entities:
        key = (_field(e, "text"), _field(e, "label"))
        if key in seen:
            continue
        seen.add(key)
        label = key[1]
        rows.append(
            {
                "text": key[0],
                "tag": LABEL_TEXT.get(label, label),
                "sensitive": label in SENSITIVE_LABELS,
            }
        )
    return rows


@dataclass
class Band:
    """Everything a verdict band shows. kind: scam | safe | unsure."""

    kind: str
    word: str
    subtype: str
    instruction: str
    fact_label: str
    fact_big: str
    fact_note: str
    fact_extra: str = ""
    inline_fact: str = ""
    fresh: bool = False


def message_band(verdict: dict) -> Band:
    """The verdict band for a checked message. `verdict` has level, label_type, confidence."""
    level, label_type = verdict["level"], verdict["label_type"]
    confidence = confidence_word(verdict["confidence"])
    if level == "scam":
        kind, word = "scam", "SCAM"
        subtype = type_name(label_type)
        instruction = MESSAGE_INSTRUCTIONS.get(label_type, MESSAGE_INSTRUCTIONS[None])
    elif level == "suspicious":
        kind, word = "unsure", "UNSURE"
        subtype = "Might be a scam. Type not known"
        instruction = "Treat it as a scam until you have checked. Do not reply or send anything."
    else:
        kind, word = "safe", "NORMAL"
        subtype = "No scam signs found"
        instruction = "Still check links and codes you did not ask for."
    return Band(
        kind=kind,
        word=word,
        subtype=subtype,
        instruction=instruction,
        fact_label="Confidence",
        fact_big=confidence,
        fact_note="The LLM gives this label. A PhayaThaiBERT model is planned to replace it.",
        inline_fact=f"Confidence: {confidence.lower()}",
    )
