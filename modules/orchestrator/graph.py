"""LangGraph graphs: one call turn, and one text message.

Turn graph:

    START -> remember -> analyze -> decide --auto_reply--> reply -> speak -> END
                                          |
                                          +--otherwise--> escalate -> END

(`speak` is left out of the graph when no speaker is injected.)

Safety rules that live here:
  * `route_after_decide` is a plain Python function that reads only `Decision.auto_reply`.
    No LLM output can change the route; only `tts_voice_agent.decision.decide` decides.
  * The only node that makes speech text is `reply`, and it calls `safe_reply` (leak check,
    template fallback). `speak` only voices `reply_text`.
  * Call memory is the CallState objects in `Runtime.states`, in process memory only.
    The graphs are compiled without a checkpointer, so LangGraph stores nothing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from asr_diarization.types import Turn
from langgraph.graph import END, START, StateGraph
from llm_serving.base import LLM
from nlp_core.registry import analyze_with_tools
from nlp_core.types import NLPResult
from tts_voice_agent.constants import ESCALATE_THRESHOLD
from tts_voice_agent.decision import Decision, decide
from tts_voice_agent.memory import CallState
from tts_voice_agent.reply import safe_reply

log = logging.getLogger(__name__)


class TurnState(TypedDict, total=False):
    call_id: str
    turn: Turn
    context: str
    nlp: NLPResult
    decision: Decision
    reply_text: str
    template_used: bool
    escalation_reason: str
    audio: bytes | None
    speak_error: str
    timings: dict[str, float]


class MessageState(TypedDict, total=False):
    text: str
    nlp: NLPResult
    verdict: dict
    timings: dict[str, float]


@dataclass
class Runtime:
    """Everything the graphs need from outside. Tests pass fakes here.

    llm: used only for reply text (via safe_reply).
    speaker: object with synthesize_bytes(text) -> bytes, or None to skip speech.
    ner / classifier / summarizer: registry names; None means the env-var default.
    states: call_id -> CallState. Owned by CallSessions; never persisted.
    """

    llm: LLM
    speaker: Any | None = None
    ner: str | None = None
    classifier: str | None = None
    summarizer: str | None = None
    states: dict[str, CallState] = field(default_factory=dict)

    def analyze(self, text: str) -> NLPResult:
        return analyze_with_tools(
            text, ner=self.ner, classifier=self.classifier, summarizer=self.summarizer
        )


def _timed(name: str, fn: Callable[[Any], dict]) -> Callable[[Any], dict]:
    def wrapper(state):
        t0 = time.perf_counter()
        update = dict(fn(state))
        update["timings"] = {**state.get("timings", {}), name: time.perf_counter() - t0}
        return update

    wrapper.__name__ = name
    return wrapper


def route_after_decide(state: TurnState) -> str:
    """Plain routing. Reads only Decision.auto_reply, nothing an LLM produced."""
    return "reply" if state["decision"].auto_reply is True else "escalate"


def build_turn_graph(rt: Runtime):
    def remember(state: TurnState) -> dict:
        rt.states[state["call_id"]].add(state["turn"])
        return {}

    def analyze(state: TurnState) -> dict:
        cs = rt.states[state["call_id"]]
        context = cs.context()
        nlp = rt.analyze(context)
        cs.add_entities(nlp.entities)
        return {"context": context, "nlp": nlp}

    def decide_node(state: TurnState) -> dict:
        decision = decide(state["nlp"], rt.states[state["call_id"]])
        log.info("decision auto_reply=%s reason=%s", decision.auto_reply, decision.reason)
        return {"decision": decision}

    def reply(state: TurnState) -> dict:
        text, template_used = safe_reply(state["nlp"], rt.states[state["call_id"]], rt.llm)
        decision = state["decision"]
        decision.reply_text, decision.template_used = text, template_used
        return {"reply_text": text, "template_used": template_used}

    def speak(state: TurnState) -> dict:
        try:
            return {"audio": rt.speaker.synthesize_bytes(state["reply_text"])}
        except Exception as exc:  # TTS trouble must not lose the text reply
            log.warning("speech failed (%s)", type(exc).__name__)
            return {"audio": None, "speak_error": type(exc).__name__}

    def escalate(state: TurnState) -> dict:
        return {"escalation_reason": state["decision"].reason}

    g = StateGraph(TurnState)
    g.add_node("remember", _timed("remember", remember))
    g.add_node("analyze", _timed("analyze", analyze))
    g.add_node("decide", _timed("decide", decide_node))
    g.add_node("reply", _timed("reply", reply))
    g.add_node("escalate", _timed("escalate", escalate))
    g.add_edge(START, "remember")
    g.add_edge("remember", "analyze")
    g.add_edge("analyze", "decide")
    g.add_conditional_edges(
        "decide", route_after_decide, {"reply": "reply", "escalate": "escalate"}
    )
    if rt.speaker is not None:
        g.add_node("speak", _timed("speak", speak))
        g.add_edge("reply", "speak")
        g.add_edge("speak", END)
    else:
        g.add_edge("reply", END)
    g.add_edge("escalate", END)
    return g.compile()


def verdict_from(nlp: NLPResult) -> dict:
    if not nlp.label_scam:
        level = "safe"
    elif nlp.confidence >= ESCALATE_THRESHOLD:
        level = "scam"
    else:
        level = "suspicious"  # includes unknown type: never reported as safe
    return {
        "level": level,
        "label_scam": nlp.label_scam,
        "label_type": nlp.label_type,
        "confidence": nlp.confidence,
        "summary": nlp.summary,
    }


def build_message_graph(rt: Runtime):
    """Text message: analyze -> verdict. No reply, no memory."""

    def analyze(state: MessageState) -> dict:
        return {"nlp": rt.analyze(state["text"])}

    def verdict(state: MessageState) -> dict:
        return {"verdict": verdict_from(state["nlp"])}

    g = StateGraph(MessageState)
    g.add_node("analyze", _timed("analyze", analyze))
    g.add_node("verdict", _timed("verdict", verdict))
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "verdict")
    g.add_edge("verdict", END)
    return g.compile()
