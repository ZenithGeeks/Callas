"""nlp-core analyzer, interim LLM version.

Entities come from rules (entities.py); summary and scam label come from the LLM via
llm-serving. A fine-tuned PhayaThaiBERT classifier is planned to replace the LLM scam
check behind the same interface (`analyze(text) -> NLPResult`).

An unparseable classifier answer is never treated as safe: it becomes label_scam=True
with label_type=None and confidence 0.0, so the call is escalated for review and never
auto-replied.
"""

from __future__ import annotations

from llm_serving.base import LLM

from .entities import extract_entities
from .parts import LLMClassifier, LLMSummarizer
from .types import NLPResult


class Analyzer:
    """Run the rule NER and one LLM for the summary and the scam label."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def analyze(self, text: str) -> NLPResult:
        """Entities from rules, summary and scam label from two LLM calls.

        This path does not read NLP_NER, NLP_CLASSIFIER or NLP_SUMMARIZER; use
        registry.analyze_with_tools for that.
        """
        entities = extract_entities(text)
        summary = LLMSummarizer(self.llm).summarize(text)
        label_scam, label_type, confidence = LLMClassifier(self.llm).predict(text)
        return NLPResult(
            entities=entities,
            summary=summary,
            label_scam=label_scam,
            label_type=label_type,
            confidence=confidence,
        )
