"""Shared types for nlp-core output."""

from __future__ import annotations

from dataclasses import dataclass, field

# label_type values. None means not a scam.
SCAM_TYPES: tuple[str, ...] = (
    "fake_authority",
    "romance",
    "investment",
    "phishing_otp",
    "lottery",
)

# Entity labels used across the project.
ENTITY_LABELS: tuple[str, ...] = (
    "PHONE",
    "BANK_ACCOUNT",
    "OTP",
    "PERSON",
    "ORG",
    "MONEY",
    "URL",
    "ID_CARD",
    "BIRTH_DATE",
    "ADDRESS",
)


@dataclass
class Entity:
    """A span of the text with its label; start and end index the original text."""

    text: str
    label: str
    start: int
    end: int


@dataclass
class NLPResult:
    """What analyze() returns. label_type is None for normal calls and for unsure answers."""

    entities: list[Entity] = field(default_factory=list)
    summary: str = ""
    label_scam: bool = False
    label_type: str | None = None
    confidence: float = 0.0
