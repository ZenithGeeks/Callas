"""Protocols for the three swappable nlp-core parts.

A fine-tuned model only has to implement one of these and be registered in
`nlp_core.registry`. Nothing else in the project needs to change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Entity


@runtime_checkable
class NERModel(Protocol):
    def extract(self, text: str) -> list[Entity]: ...


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, text: str) -> str: ...


@runtime_checkable
class ScamClassifier(Protocol):
    def predict(self, text: str) -> tuple[bool, str | None, float]:
        """Return (label_scam, label_type, confidence).

        label_type is one of nlp_core.types.SCAM_TYPES or None. When the model is unsure it
        must return (True, None, 0.0): an unsure answer is never treated as safe.
        """
        ...
