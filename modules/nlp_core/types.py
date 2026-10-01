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
ENTITY_LABELS: tuple[str, ...] = ("PHONE", "BANK_ACCOUNT", "OTP", "PERSON", "ORG", "MONEY", "URL")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


@dataclass
class NLPResult:
    entities: list[Entity] = field(default_factory=list)
    summary: str = ""
    label_scam: bool = False
    label_type: str | None = None
    confidence: float = 0.0
