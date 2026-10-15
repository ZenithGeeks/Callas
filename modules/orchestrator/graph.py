"""LangGraph graphs: one call turn (`build_turn_graph`) and one text message.

Turn graph:

    START -> remember -> analyze -> decide --auto_reply--> reply -> speak -> END
                                          |
                                          +--otherwise--> escalate -> END

`speak` is left out when no speaker is injected. Read `route_after_decide` and `reply` first:
they carry the safety rules (README, "Where the safety rules live"). Per-call dependencies
come through the runtime context (`TurnContext`), passed as `context=` on every invoke or
stream. The graphs hold no call data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime as GraphRuntime
from langgraph.types import RetryPolicy

from asr_diarization.types import Turn
from callas_settings import get_settings
from llm_serving.base import LLM
from llm_serving.client import LLMUnavailable
from nlp_core.registry import analyze_with_tools
from nlp_core.types import NLPResult
from tts_voice_agent import decision as decision_rules
from tts_voice_agent.constants import ESCALATE_THRESHOLD
from tts_voice_agent.decision import Decision
from tts_voice_agent.memory import CallState, clean_turn
from tts_voice_agent.reply import safe_reply

log = logging.getLogger(__name__)


class TurnState(TypedDict, total=False):
    """What flows through the turn graph. Each node adds its own keys."""

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
    """What flows through the message graph."""

    text: str
    nlp: NLPResult
    verdict: dict
    timings: dict[str, float]


@dataclass
class TurnContext:
    """Run-scoped dependencies, read by nodes as `runtime.context`. Built by Runtime.context()."""

    llm: LLM
    speaker: Any | None
    ner: str | None
    classifier: str | None
    summarizer: str | None
    states: dict[str, CallState]
    writer: Any | None
    judge: Any | None

    def analyze(self, text: str) -> NLPResult:
        """Run NER, classifier and summarizer on `text` with the names chosen for this run."""
        return analyze_with_tools(
            text, ner=self.ner, classifier=self.classifier, summarizer=self.summarizer
        )


@dataclass
class Runtime:
    """Everything the graphs need from outside. Tests pass fakes here.

    llm: used only for reply text (via safe_reply).
    speaker: object with synthesize_bytes(text) -> bytes, or None to skip speech.
    ner / classifier / summarizer: registry names; None means the env-var default.
    states: call_id -> CallState. Owned by CallSessions; never persisted.
    writer / judge: optional reply writer (`write(context, result)`) and leak judge
    (`blocks(reply, context)`). When left as None they are built once from the settings:
    REPLY_ENGINE=dspy gives a DSPy writer, LEAK_JUDGE=on gives a DSPy judge; the defaults
    (prompt / off) give None, which keeps the plain LLM reply path. Pass fakes in tests.
    """

    llm: LLM
    speaker: Any | None = None
    ner: str | None = None
    classifier: str | None = None
    summarizer: str | None = None
    states: dict[str, CallState] = field(default_factory=dict)
    writer: Any | None = None
    judge: Any | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        if self.writer is None and settings.reply_engine == "dspy":
            from llm_serving.dspy_programs import DSPyReplyWriter

            self.writer = DSPyReplyWriter()
        if self.judge is None and settings.leak_judge == "on":
            from llm_serving.dspy_programs import DSPyLeakJudge

            self.judge = DSPyLeakJudge()

    def context(self) -> TurnContext:
        """The context object to pass as `context=` when invoking a graph."""
        return TurnContext(
            llm=self.llm,
            speaker=self.speaker,
            ner=self.ner,
            classifier=self.classifier,
            summarizer=self.summarizer,
            states=self.states,
            writer=self.writer,
            judge=self.judge,
        )


def _timed(name: str, fn: Callable[[Any, Any], dict]) -> Callable[..., dict]:
    """Wrap a node `fn(state, ctx)` so it adds its run time to `state["timings"]`."""

    # LangGraph injects `runtime` by parameter name, so the wrapper must declare it.
    def wrapper(state, runtime: GraphRuntime[TurnContext]):
        t0 = time.perf_counter()
        update = dict(fn(state, runtime.context))
        update["timings"] = {**state.get("timings", {}), name: time.perf_counter() - t0}
        return update

    wrapper.__name__ = name
    return wrapper


def route_after_decide(state: TurnState) -> str:
    """Plain routing. Reads only Decision.auto_reply, nothing an LLM produced.

    Kept as a pure function on add_conditional_edges, not Command routing from inside a node:
    the tests prove the route depends only on Decision.auto_reply.
    """
    return "reply" if state["decision"].auto_reply is True else "escalate"


# Retry only the analyze node, and only when the LLM server was unreachable (connection error
# or timeout). One retry. Reply is not retried: safe_reply already falls back to a template.
ANALYZE_RETRY = RetryPolicy(
    max_attempts=2, initial_interval=0.3, backoff_factor=2.0, retry_on=LLMUnavailable
)


def remember(state: TurnState, ctx: TurnContext) -> dict:
    """Clean the turn text once and add it to call memory."""
    turn = clean_turn(state["turn"])  # one clean text for every tool and for the screen
    ctx.states[state["call_id"]].add(turn)
    return {"turn": turn}


def analyze_turn(state: TurnState, ctx: TurnContext) -> dict:
    """Run the NLP tools on the whole call so far and remember the entities found."""
    call = ctx.states[state["call_id"]]
    context = call.context()
    nlp = ctx.analyze(context)
    call.add_entities(nlp.entities)
    return {"context": context, "nlp": nlp}


def decide(state: TurnState, ctx: TurnContext) -> dict:
    """Choose between a reply and escalation. The rules are in tts_voice_agent.decision."""
    decision = decision_rules.decide(state["nlp"], ctx.states[state["call_id"]])
    log.info("decision auto_reply=%s reason=%s", decision.auto_reply, decision.reason)
    return {"decision": decision}


def reply(state: TurnState, ctx: TurnContext) -> dict:
    """Write the reply with safe_reply. This is the only node that makes speech text."""
    call = ctx.states[state["call_id"]]
    text, template_used = safe_reply(
        state["nlp"], call, ctx.llm, writer=ctx.writer, judge=ctx.judge
    )
    # The final state's Decision carries the reply, so it is filled in place (not in the update).
    decision = state["decision"]
    decision.reply_text, decision.template_used = text, template_used
    caller_turn = state["turn"]
    # Remember what we said, so the next reply can follow on from it.
    call.add(Turn("agent", caller_turn.end, caller_turn.end, text))
    return {"reply_text": text, "template_used": template_used}


def speak(state: TurnState, ctx: TurnContext) -> dict:
    """Voice reply_text. It never makes text of its own."""
    try:
        return {"audio": ctx.speaker.synthesize_bytes(state["reply_text"])}
    except Exception as exc:  # TTS trouble must not lose the text reply
        log.warning("speech failed (%s)", type(exc).__name__)
        return {"audio": None, "speak_error": type(exc).__name__}


def escalate(state: TurnState, ctx: TurnContext) -> dict:
    """Record why the call went to a person. Nothing is said to the caller."""
    return {"escalation_reason": state["decision"].reason}


def build_turn_graph(rt: Runtime):
    """Compile the turn graph once. Invoke it with `context=rt.context()`.

    `rt` only decides whether the speak node exists; nodes read dependencies from the runtime
    context. No node cache (CachePolicy / compile(cache=...)) on purpose: an in-memory cache keyed
    on node input would keep call text after the call ends, which breaks the rule that call memory
    is deleted when the call ends.
    """
    g = StateGraph(TurnState, context_schema=TurnContext)
    g.add_node("remember", _timed("remember", remember))
    g.add_node("analyze", _timed("analyze", analyze_turn), retry_policy=ANALYZE_RETRY)
    g.add_node("decide", _timed("decide", decide))
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
    return g.compile()  # no checkpointer: call memory lives only in ctx.states


def verdict_from(nlp: NLPResult) -> dict:
    """Turn an NLP result into the verdict dict: level safe, suspicious or scam."""
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


def analyze_text(state: MessageState, ctx: TurnContext) -> dict:
    """Run the NLP tools on one text message."""
    return {"nlp": ctx.analyze(state["text"])}


def verdict(state: MessageState, ctx: TurnContext) -> dict:
    """Make the verdict from the NLP result."""
    return {"verdict": verdict_from(state["nlp"])}


def build_message_graph(rt: Runtime):
    """Text message: analyze -> verdict. No reply, no memory. Invoke with `context=`."""
    g = StateGraph(MessageState, context_schema=TurnContext)
    g.add_node("analyze", _timed("analyze", analyze_text), retry_policy=ANALYZE_RETRY)
    g.add_node("verdict", _timed("verdict", verdict))
    g.add_edge(START, "analyze")
    g.add_edge("analyze", "verdict")
    g.add_edge("verdict", END)
    return g.compile()  # no checkpointer, no memory
