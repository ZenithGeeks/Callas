"""LangChain tools over the nlp-core registry. Sub-agents call these.

Each tool returns a JSON-serialisable dict. Which model sits behind a tool is decided by the
registry (env vars NLP_NER / NLP_CLASSIFIER / NLP_SUMMARIZER), so replacing a model needs
no change here.
"""

from __future__ import annotations

from dataclasses import asdict

from langchain_core.tools import tool

from nlp_core import registry


@tool
def extract_entities(text: str) -> dict:
    """Find phone numbers, bank accounts, OTPs, URLs and amounts in Thai text."""
    entities = registry.get("ner").extract(text)
    return {"entities": [asdict(e) for e in entities]}


@tool
def classify_scam(text: str) -> dict:
    """Say whether Thai text looks like a scam, with the scam type and a confidence 0..1."""
    scam, kind, conf = registry.get("classifier").predict(text)
    return {"label_scam": bool(scam), "label_type": kind, "confidence": float(conf)}


@tool
def summarize_text(text: str) -> dict:
    """Summarize Thai text in one or two sentences."""
    return {"summary": registry.get("summarizer").summarize(text)}


TOOLS = [extract_entities, classify_scam, summarize_text]
