"""Safe reply text: LLM reply if it passes every check, otherwise a fixed template."""

from __future__ import annotations

import logging

from llm_serving.base import LLM
from nlp_core.types import NLPResult

from .constants import MAX_REPLY_CHARS
from .leak import leak_reasons
from .memory import CallState
from .templates import pick_template

log = logging.getLogger(__name__)


def safe_reply(result: NLPResult, state: CallState, llm: LLM) -> tuple[str, bool]:
    """Return (text, template_used). Never logs the LLM text itself."""
    try:
        text = llm.complete("reply", state.context())
    except Exception as exc:  # any LLM failure must fall back, never crash the call
        log.warning("reply LLM failed (%s), using template", type(exc).__name__)
        return pick_template(result), True
    text = (text or "").strip()
    if not text:
        log.warning("reply LLM returned empty text, using template")
        return pick_template(result), True
    if len(text) > MAX_REPLY_CHARS:
        log.warning("reply too long (%d chars), using template", len(text))
        return pick_template(result), True
    reasons = leak_reasons(text, state.entities_seen)
    if reasons:
        log.warning("reply blocked by leak check: %s", ",".join(reasons))
        return pick_template(result), True
    return text, False
