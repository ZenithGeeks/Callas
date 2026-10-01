"""nlp-core analyzer, interim LLM version.

Entities come from rules (entities.py); summary and scam label come from the LLM via
llm-serving. A fine-tuned WangchanBERTa will replace this behind the same interface
(`analyze(text) -> NLPResult`).

An unparseable classifier answer is never treated as safe: it becomes label_scam=True
with label_type=None and confidence 0.0, so the call is escalated for review and never
auto-replied.
"""

from __future__ import annotations

from llm_serving.base import LLM
from llm_serving.client import parse_label

from .entities import extract_entities
from .types import SCAM_TYPES, NLPResult


class Analyzer:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def analyze(self, text: str) -> NLPResult:
        entities = extract_entities(text)
        summary = self.llm.complete("summary", text).strip()
        label = parse_label(self.llm.complete("classify", text))

        if label == "normal":
            scam, kind, conf = False, None, 1.0
        elif label in SCAM_TYPES:
            scam, kind, conf = True, label, 1.0
        else:
            scam, kind, conf = True, None, 0.0
        return NLPResult(
            entities=entities,
            summary=summary,
            label_scam=scam,
            label_type=kind,
            confidence=conf,
        )
