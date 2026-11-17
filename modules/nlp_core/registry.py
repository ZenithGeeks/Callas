"""Registry of pluggable nlp-core parts: NER, scam classifier, summarizer.

    register("classifier", "mymodel", lambda: MyClassifier(...))
    # then set NLP_CLASSIFIER=mymodel in .env

The fine-tuned classifiers (finetuned.py) are registered as "<gate encoder>+<type encoder>",
as "<encoder>" for both models of one encoder, and as "best".

Defaults come from NLP_NER ("rules"), NLP_CLASSIFIER ("llm"), NLP_SUMMARIZER ("llm"): the
process environment first, read at each `get()`, then .env. Each (kind, name) instance is
built once and cached.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from .parts import LLMClassifier as LLMClassifier  # moved to parts.py, still importable from here
from .parts import LLMSummarizer as LLMSummarizer
from .parts import RulesNER as RulesNER
from .types import NLPResult

log = logging.getLogger(__name__)

KINDS: tuple[str, ...] = ("ner", "classifier", "summarizer")
ENV_VARS: dict[str, str] = {
    "ner": "NLP_NER",
    "classifier": "NLP_CLASSIFIER",
    "summarizer": "NLP_SUMMARIZER",
}
DEFAULT_NAMES: dict[str, str] = {"ner": "rules", "classifier": "llm", "summarizer": "llm"}

# Encoders the notebook fine-tunes, as models/<encoder>-scam-{gate,type}, and their display names.
FINE_TUNED: dict[str, str] = {
    "phayathaibert": "PhayaThaiBERT",
    "wangchanberta": "WangchanBERTa",
    "bge_m3": "BGE-M3",
}
# From models/scam_classifier_metrics.json (run of 2026-11-17, test split):
# gate: WangchanBERTa macro-F1 0.950 vs PhayaThaiBERT 0.947 (McNemar p 0.31), but it raises the
#   alarm before the scam starts in 5.6% of scam calls against 12.9%, and catches more of them.
# type: PhayaThaiBERT 0.889, BGE-M3 0.889, WangchanBERTa 0.872; PhayaThaiBERT is the smaller of the tie.
BEST_FINE_TUNED: tuple[str, str] = ("wangchanberta", "phayathaibert")

_FACTORIES: dict[str, dict[str, Callable[[], Any]]] = {k: {} for k in KINDS}
_INSTANCES: dict[tuple[str, str], Any] = {}
_LOCK = threading.RLock()  # re-entrant: a factory may call get() for another part


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
    """Names registered for this kind, sorted."""
    _check_kind(kind)
    return sorted(_FACTORIES[kind])


def default_name(kind: str) -> str:
    """Name chosen by the environment variable, then .env, then the built-in default."""
    _check_kind(kind)
    from callas_settings import get_settings

    # Settings has one field per kind: nlp_ner, nlp_classifier, nlp_summarizer.

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


def fine_tuned_pair(name: str) -> tuple[str, str] | None:
    """(gate encoder, type encoder) of a fine-tuned classifier name, or None for another name."""
    if name == "best":
        return BEST_FINE_TUNED
    gate, _, type_ = name.partition("+")
    type_ = type_ or gate
    return (gate, type_) if gate in FINE_TUNED and type_ in FINE_TUNED else None


def _fine_tuned(gate: str, type_: str) -> Callable[[], Any]:
    def build():
        # Imported here: torch and transformers load only when a fine-tuned model is chosen.
        from .finetuned import FineTunedClassifier

        return FineTunedClassifier.from_encoders(gate, type_)

    return build


register("ner", "rules", RulesNER)
register("classifier", "llm", LLMClassifier)
for _name in ["best", *FINE_TUNED, *(f"{g}+{t}" for g in FINE_TUNED for t in FINE_TUNED)]:
    register("classifier", _name, _fine_tuned(*fine_tuned_pair(_name)))
register("summarizer", "llm", LLMSummarizer)


def analyze_with_tools(
    text: str,
    ner: str | None = None,
    classifier: str | None = None,
    summarizer: str | None = None,
) -> NLPResult:
    """Run the selected NER, summarizer and classifier and combine them into an NLPResult.

    A failed summary leaves the summary empty: the scam label must not depend on the LLM
    when the classifier is a local model.
    """
    entities = get("ner", ner).extract(text)
    try:
        summary = get("summarizer", summarizer).summarize(text)
    except Exception as exc:  # never log the call text
        log.warning("summary failed (%s), going on without one", type(exc).__name__)
        summary = ""
    label_scam, label_type, confidence = get("classifier", classifier).predict(text)
    return NLPResult(
        entities=entities,
        summary=summary,
        label_scam=label_scam,
        label_type=label_type,
        confidence=confidence,
    )
