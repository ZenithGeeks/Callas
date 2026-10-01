"""Reply-or-warn rules. Pure and deterministic: same input, same answer."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_core.types import NLPResult

from .constants import ESCALATE_THRESHOLD
from .memory import CallState


@dataclass
class Decision:
    auto_reply: bool
    reason: str
    reply_text: str | None = None
    template_used: bool = False


def decide(result: NLPResult, state: CallState) -> Decision:
    if state.escalated:
        return Decision(auto_reply=False, reason="call already escalated")
    if result.label_scam and result.confidence >= ESCALATE_THRESHOLD:
        state.escalated = True
        return Decision(auto_reply=False, reason="high-confidence scam, escalate to user")
    if result.label_scam:
        # includes unknown type / confidence 0.0: when unsure, never auto-reply
        state.escalated = True
        return Decision(auto_reply=False, reason="low-confidence scam, escalate for review")
    return Decision(auto_reply=True, reason="benign")
