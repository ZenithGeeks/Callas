"""Presentation helpers: turn pipeline output into what the templates draw.

Nothing here decides anything about a call or message. It only formats and names things.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SENSITIVE_LABELS = frozenset(
    {"PHONE", "BANK_ACCOUNT", "OTP", "MONEY", "URL", "ID_CARD", "BIRTH_DATE", "ADDRESS"}
)
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
    return TYPE_NAMES.get(label_type, TYPE_NAMES[None])


def confidence_word(confidence: float) -> str:
    if confidence >= HIGH_CONFIDENCE:
        return "High"
    if confidence > 0:
        return "Low"
    return "Not known"


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def fmt_seconds(seconds: float) -> str:
    return f"{seconds:.1f} s"


def who(speaker: str) -> str:
    """Speaker A is the person being called; every other speaker is the caller."""
    return "You" if speaker == "A" else "Caller"


@dataclass
class Seg:
    text: str
    label: str | None = None

    @property
    def tone(self) -> str:
        if self.label is None:
            return ""
        return "alarm" if self.label in SENSITIVE_LABELS else "ink"


def _get(entity: Any, name: str) -> Any:
    return entity[name] if isinstance(entity, dict) else getattr(entity, name)


def segments(text: str, entities: list[Any]) -> list[Seg]:
    """Split text into plain and entity pieces.

    An entity is placed by its offsets when they still match its text, otherwise by searching
    for its text. Overlaps are dropped (the earlier piece wins).
    """
    spans: list[tuple[int, int, str]] = []
    for e in entities:
        value, label = _get(e, "text"), _get(e, "label")
        if not value:
            continue
        start, end = _get(e, "start"), _get(e, "end")
        if text[start:end] != value:
            start = text.find(value)
            end = start + len(value)
        if start < 0:
            continue
        spans.append((start, end, label))
    spans.sort()
    out: list[Seg] = []
    pos = 0
    for start, end, label in spans:
        if start < pos:
            continue
        if start > pos:
            out.append(Seg(text[pos:start]))
        out.append(Seg(text[start:end], label))
        pos = end
    if pos < len(text):
        out.append(Seg(text[pos:]))
    return out or [Seg(text)]


def entity_rows(entities: list[Any]) -> list[dict]:
    """One row per distinct (text, label), in order of first appearance."""
    rows, seen = [], set()
    for e in entities:
        key = (_get(e, "text"), _get(e, "label"))
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
    level, kind = verdict["level"], verdict["label_type"]
    conf = confidence_word(verdict["confidence"])
    note = "Interim LLM classifier. WangchanBERTa replaces it after training."
    if level == "scam":
        return Band(
            "scam",
            "SCAM",
            type_name(kind),
            MESSAGE_INSTRUCTIONS.get(kind, MESSAGE_INSTRUCTIONS[None]),
            "Confidence",
            conf,
            note,
            inline_fact=f"Confidence: {conf.lower()}",
        )
    if level == "suspicious":
        return Band(
            "unsure",
            "UNSURE",
            "Might be a scam. Type not known",
            "Treat it as a scam until you have checked. Do not reply or send anything.",
            "Confidence",
            conf,
            note,
            inline_fact=f"Confidence: {conf.lower()}",
        )
    return Band(
        "safe",
        "NORMAL",
        "No scam signs found",
        "Still check links and codes you did not ask for.",
        "Confidence",
        conf,
        note,
        inline_fact=f"Confidence: {conf.lower()}",
    )
