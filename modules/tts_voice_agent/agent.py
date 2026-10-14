"""Per-turn flow: remember the turn, analyze the call so far, decide, maybe reply."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from asr_diarization.types import Turn
from llm_serving.base import LLM
from nlp_core.types import NLPResult

from .decision import Decision, decide
from .memory import CallState
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
    state.add(turn)
    result = analyze(state.context())
    state.add_entities(result.entities)
    decision = decide(result, state)
    if decision.auto_reply:
        decision.reply_text, decision.template_used = safe_reply(result, state, llm)
        state.add(Turn("agent", turn.end, turn.end, decision.reply_text))
    log.info(
        "decision auto_reply=%s reason=%s template_used=%s",
        decision.auto_reply,
        decision.reason,
        decision.template_used,
    )
    return TurnOutcome(result=result, decision=decision)
