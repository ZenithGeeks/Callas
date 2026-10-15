"""Registry of pluggable nlp-core parts: NER, scam classifier, summarizer.

    register("classifier", "wangchanberta", lambda: WangchanbertaClassifier(...))
    # then set NLP_CLASSIFIER=wangchanberta in .env

Defaults come from NLP_NER ("rules"), NLP_CLASSIFIER ("llm"), NLP_SUMMARIZER ("llm"): the
process environment first, read at each `get()`, then .env. Each (kind, name) instance is
built once and cached.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any

from llm_serving.base import LLM
from llm_serving.client import parse_label

from .entities import extract_entities
from .types import SCAM_TYPES, Entity, NLPResult

KINDS: tuple[str, ...] = ("ner", "classifier", "summarizer")
ENV_VARS: dict[str, str] = {
    "ner": "NLP_NER",
    "classifier": "NLP_CLASSIFIER",
    "summarizer": "NLP_SUMMARIZER",
}
DEFAULT_NAMES: dict[str, str] = {"ner": "rules", "classifier": "llm", "summarizer": "llm"}

_FACTORIES: dict[str, dict[str, Callable[[], Any]]] = {k: {} for k in KINDS}
_INSTANCES: dict[tuple[str, str], Any] = {}
_LOCK = threading.RLock()


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}; expected one of {list(KINDS)}")


def register(kind: str, name: str, factory: Callable[[], Any]) -> None:
    """Add (or replace) a part. `factory` takes no arguments and returns the instance."""
    _check_kind(kind)
    with _LOCK:
        _FACTORIES[kind][name] = factory
        _INSTANCES.pop((kind, name), None)


def available(kind: str) -> list[str]:
    _check_kind(kind)
    return sorted(_FACTORIES[kind])


def default_name(kind: str) -> str:
    _check_kind(kind)
    from callas_settings import get_settings

    from_process_env = os.environ.get(ENV_VARS[kind], "").strip()
    return from_process_env or getattr(get_settings(), f"nlp_{kind}").strip() or DEFAULT_NAMES[kind]


def get(kind: str, name: str | None = None) -> Any:
    """Return the part registered under `name`, or the default for `kind`. Built once."""
    _check_kind(kind)
    name = name or default_name(kind)
    with _LOCK:
        if (kind, name) in _INSTANCES:
            return _INSTANCES[(kind, name)]
        factory = _FACTORIES[kind].get(name)
        if factory is None:
            raise KeyError(
                f"no {kind} named {name!r}; available: {available(kind)}. "
                f"Register it with nlp_core.registry.register({kind!r}, {name!r}, factory)."
            )
        _INSTANCES[(kind, name)] = factory()
        return _INSTANCES[(kind, name)]


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


register("ner", "rules", RulesNER)
register("classifier", "llm", LLMClassifier)
register("summarizer", "llm", LLMSummarizer)


def analyze_with_tools(
    text: str,
    ner: str | None = None,
    classifier: str | None = None,
    summarizer: str | None = None,
) -> NLPResult:
    """Run the selected NER, summarizer and classifier and combine them into an NLPResult."""
    entities = get("ner", ner).extract(text)
    summary = get("summarizer", summarizer).summarize(text)
    scam, kind, conf = get("classifier", classifier).predict(text)
    return NLPResult(
        entities=entities, summary=summary, label_scam=scam, label_type=kind, confidence=conf
    )
