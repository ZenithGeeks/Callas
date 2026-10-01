"""nlp-core: entities, summary and scam classification behind one analyze() call."""

from __future__ import annotations

from .analyzer import Analyzer
from .types import NLPResult

_default: Analyzer | None = None


def analyze(text: str) -> NLPResult:
    """Analyze text with a lazily built default Analyzer(LLMClient())."""
    global _default
    if _default is None:
        from llm_serving.client import LLMClient

        _default = Analyzer(LLMClient())
    return _default.analyze(text)
