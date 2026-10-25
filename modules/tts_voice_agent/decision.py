"""Reply-or-warn rules. Deterministic: the same result and state always give the same answer."""

from __future__ import annotations

from dataclasses import dataclass

from nlp_core.types import NLPResult

from .constants import ESCALATE_THRESHOLD
from .memory import CallState


@dataclass
class Decision:
    """Whether to auto-reply on this turn, and why.

    decide() leaves the reply fields empty. Whoever writes the reply returns a copy with them
    set (dataclasses.replace), so the Decision decide() made is never changed.
    """

    auto_reply: bool
    reason: str
    reply_text: str | None = None
    template_used: bool = False


def decide(result: NLPResult, state: CallState) -> Decision:
    """Reply only to a benign call. A scam result escalates, and stays escalated to call end."""
    if state.escalated:
        return Decision(auto_reply=False, reason="call already escalated")
    if not result.label_scam:
        return Decision(auto_reply=True, reason="benign")
    state.escalated = True
    if result.confidence >= ESCALATE_THRESHOLD:
        return Decision(auto_reply=False, reason="high-confidence scam, escalate to user")
    # includes unknown type / confidence 0.0: when unsure, never auto-reply
    return Decision(auto_reply=False, reason="low-confidence scam, escalate for review")
