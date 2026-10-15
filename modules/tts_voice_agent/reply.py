"""Safe reply text: LLM reply if it passes every check, otherwise a fixed template.

Order of checks, none of which can be skipped:
  1. the reply LLM sees only the REDACTED call context (`redact_context`);
  2. the reply must be non-empty, short, and pass the deterministic `leaks_sensitive` gate;
  3. only if the rules pass, an optional judge may still block it.
Final rule: blocked = leaks_sensitive(reply) OR judge_blocks. The judge can add blocks, never
remove one. A judge that fails, times out or answers with anything but a clear "no leak"
counts as a block.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from llm_serving.base import LLM
from nlp_core.types import NLPResult

from .constants import MAX_REPLY_CHARS
from .leak import leak_reasons
from .memory import CallState
from .redact import redact_context
from .spelling import looks_misspelled
from .templates import pick_template

log = logging.getLogger(__name__)


class ReplyWriter(Protocol):
    def write(self, context: str, result: NLPResult) -> str: ...


class LeakJudge(Protocol):
    def blocks(self, reply: str, context: str) -> bool: ...


def _judge_verdict(judge: Any, text: str, context: str) -> str | None:
    """Reason code if the judge blocks, None only for a clear "no leak" (a real False)."""
    try:
        verdict = judge.blocks(text, context)
    except Exception as exc:  # judge trouble must block, never pass
        log.warning("leak judge failed (%s), using template", type(exc).__name__)
        return "judge_error"
    if verdict is False:
        return None
    return "judge_block" if verdict is True else "judge_error"


_DAYS = "จันทร์ อังคาร พุธ พฤหัสบดี ศุกร์ เสาร์ อาทิตย์".split()
_MONTHS = (
    "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม"
).split()


def current_facts(now: datetime | None = None) -> str:
    """Today's date and time in Thai (Bangkok time, Buddhist year), for the reply prompt."""
    now = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return (
        f"วันนี้วัน{_DAYS[now.weekday()]}ที่ {now.day} {_MONTHS[now.month - 1]} "
        f"{now.year + 543} เวลา {now:%H:%M} น."
    )


def fact_numbers(now: datetime | None = None) -> set[str]:
    """Numbers a reply may contain: those of current_facts(), allowing the minute to tick over."""
    now = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return {
        n
        for delta in (-1, 0, 1)
        for n in re.findall(r"\d+", current_facts(now + timedelta(minutes=delta)))
    }


def safe_reply(
    result: NLPResult,
    state: CallState,
    llm: LLM,
    *,
    writer: ReplyWriter | None = None,
    judge: LeakJudge | None = None,
    now: datetime | None = None,
) -> tuple[str, bool]:
    """Return (text, template_used). Never logs the LLM text itself."""
    facts = current_facts(now)
    context = f"[ข้อมูลปัจจุบัน] {facts}\n" + redact_context(state.context(), state.entities_seen)
    t0 = time.perf_counter()
    try:
        if writer is not None:
            text = writer.write(context, result)
        else:
            text = llm.complete("reply", context)
    except Exception as exc:  # any LLM failure must fall back, never crash the call
        log.warning("reply LLM failed (%s), using template", type(exc).__name__)
        return pick_template(result), True
    write_s = time.perf_counter() - t0
    text = (text or "").strip()
    if not text:
        log.warning("reply LLM returned empty text, using template")
        return pick_template(result), True
    if len(text) > MAX_REPLY_CHARS:
        log.warning("reply too long (%d chars), using template", len(text))
        return pick_template(result), True
    reasons = leak_reasons(text, state.entities_seen, fact_numbers(now))
    if reasons:
        log.warning("reply blocked by leak check: %s", ",".join(reasons))
        return pick_template(result), True
    if looks_misspelled(text):
        log.warning("reply blocked: garbled Thai spelling, using template")
        return pick_template(result), True
    if judge is not None:
        t1 = time.perf_counter()
        code = _judge_verdict(judge, text, context)
        log.info(
            "reply judge reason=%s write_s=%.3f judge_s=%.3f reply_chars=%d context_chars=%d",
            code or "pass",
            write_s,
            time.perf_counter() - t1,
            len(text),
            len(context),
        )
        if code:
            log.warning("reply blocked by leak judge: %s", code)
            return pick_template(result), True
    return text, False
