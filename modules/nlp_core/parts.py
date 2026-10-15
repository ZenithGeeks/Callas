"""The built-in nlp-core parts: rule NER, LLM scam classifier, LLM summarizer.

registry.py registers them under the default names; analyzer.py builds them directly.
"""

from __future__ import annotations

from llm_serving.base import LLM
from llm_serving.client import parse_label

from .entities import extract_entities
from .types import SCAM_TYPES, Entity


class RulesNER:
    """Rule-based entities (regex), see entities.py."""

    def extract(self, text: str) -> list[Entity]:
        return extract_entities(text)


class _LLMPart:
    """Base for the parts that call the LLM. The default client is built on first use."""

    def __init__(self, llm: LLM | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            from llm_serving.client import LLMClient

            self._llm = LLMClient()
        return self._llm


class LLMClassifier(_LLMPart):
    """Scam label from the vLLM "classify" job.

    Unknown or unparseable answers map to (True, None, 0.0): never treated as safe.
    """

    def predict(self, text: str) -> tuple[bool, str | None, float]:
        label = parse_label(self.llm.complete("classify", text))
        if label == "normal":
            return False, None, 1.0
        if label in SCAM_TYPES:
            return True, label, 1.0
        return True, None, 0.0


class LLMSummarizer(_LLMPart):
    """Summary from the vLLM "summary" job."""

    def summarize(self, text: str) -> str:
        return self.llm.complete("summary", text).strip()
