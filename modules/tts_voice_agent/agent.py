"""Per-turn flow: remember the turn, analyze the call so far, decide, maybe reply."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace

from asr_diarization.types import Turn
from llm_serving.base import LLM
from nlp_core.types import NLPResult

from .decision import Decision, decide
from .memory import CallState, clean_turn
from .reply import safe_reply

log = logging.getLogger(__name__)


@dataclass
class TurnOutcome:
    """What on_turn returns: the nlp-core result for this turn plus the decision."""

    result: NLPResult
    decision: Decision


def on_turn(
    state: CallState, turn: Turn, analyze: Callable[[str], NLPResult], llm: LLM
) -> TurnOutcome:
    """Handle one turn: remember it, analyze the call, decide, and write a reply if benign."""
    turn = clean_turn(turn)
    state.add(turn)
    result = analyze(state.context())
    state.add_entities(result.entities)
    decision = decide(result, state)
    if decision.auto_reply:
        text, template_used = safe_reply(result, state, llm)
        decision = replace(decision, reply_text=text, template_used=template_used)
        state.add(Turn("agent", turn.end, turn.end, text))
    log.info(
        "decision auto_reply=%s reason=%s template_used=%s",
        decision.auto_reply,
        decision.reason,
        decision.template_used,
    )
    return TurnOutcome(result=result, decision=decision)
