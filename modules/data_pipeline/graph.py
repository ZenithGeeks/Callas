"""The call graph (LangGraph): START -> plan -> caller <-> receiver -> finish -> END."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from .agents import Agents
from .audio import speak_turn
from .budget import ContextTooLong
from .config import Config
from .fill import SlotFiller
from .finish import TOO_LONG_FOR_CONTEXT, call_text, finish_call, leak_summary
from .scenarios import CallConfig
from .store import Store
from .tts import Voices
from .turns import is_model_error, write_plan, write_turn

log = logging.getLogger("data_pipeline")


def retry_policy(initial_interval: float = 5.0) -> RetryPolicy:
    """A malformed answer or a rate limit is retried by LangGraph with waits of 5, 15 and 45 s.
    Everything else fails at once, and a call that stopped halfway resumes from the store."""
    return RetryPolicy(
        max_attempts=4,
        initial_interval=initial_interval,
        backoff_factor=3.0,
        max_interval=120.0,
        retry_on=is_model_error,
    )


MODEL_RETRY = retry_policy()


class SimState(TypedDict):
    """Graph state: the call record. Nodes change it in place and return it, so the default
    overwrite reducer is all that is needed."""

    record: dict


@dataclass
class SimContext:
    """What every node needs that is not state: passed to invoke(context=...) and read in a
    node through its `runtime: Runtime[SimContext]` parameter."""

    agents: Agents
    config: Config
    call: CallConfig
    store: Store
    voices: Voices | None
    filler: SlotFiller


def plan(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Scam calls only: write the caller's plan once, before the first turn."""
    ctx, record = runtime.context, state["record"]
    if ctx.call.label_scam and record["generation"]["briefing"] is None:
        record["generation"]["briefing"] = write_plan(ctx.agents, ctx.call, ctx.config, ctx.store)
    ctx.store.save(record)
    return {"record": record}


def add_turn(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Add one turn (caller or receiver, by turn order), speak it, save the call.

    Safe to retry: the turn is built first, from the record as it was; only then is it added to
    a copy of the record, which becomes the new state once the node returns. If anything after
    the model answer fails and LangGraph runs the node again, the state still has the old
    turns, so a turn is never added twice or given the wrong speaker."""
    ctx = runtime.context
    old = state["record"]["generation"]
    try:
        turn = write_turn(
            ctx.agents,
            ctx.call,
            ctx.config,
            old["turns"],
            ctx.filler,
            (old["briefing"] or {}).get("plan", []),
        )
    except ContextTooLong as exc:
        # The prompt cannot be made to fit, so the model is not asked. The call ends here,
        # rejected, with the turns it has.
        log.warning("%s: %s", state["record"]["id"], exc)
        record = copy.deepcopy(state["record"])
        generation = record["generation"]
        call_text(record)
        leak_summary(record)
        generation["reasons"] = [TOO_LONG_FOR_CONTEXT]
        generation["status"] = "rejected"
        ctx.store.save(record)
        return {"record": record}
    record = copy.deepcopy(state["record"])
    turns = record["generation"]["turns"]
    turns.append(turn)
    if ctx.voices is not None:
        speak_turn(ctx.voices, record, len(turns) - 1, ctx.store.folder)
    ctx.store.save(record)
    return {"record": record}


def finish(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Build the text, join the audio, ask the judge, and set accepted or rejected."""
    ctx, record = runtime.context, state["record"]
    if record["generation"]["status"] == "in_progress":  # else it ended early (too long)
        finish_call(record, ctx.agents, ctx.config, ctx.voices, ctx.store.folder)
    ctx.store.save(record)
    return {"record": record}


def next_step(
    state: SimState, runtime: Runtime[SimContext]
) -> Literal["caller", "receiver", "finish"]:
    """Whose turn it is, or "finish" when someone hung up or the call reached its length."""
    turns = state["record"]["generation"]["turns"]
    if state["record"]["generation"]["status"] != "in_progress":
        return "finish"
    if len(turns) >= runtime.context.call.max_turns or (turns and turns[-1]["end_call"]):
        return "finish"
    return "caller" if len(turns) % 2 == 0 else "receiver"


def build_graph(retry: RetryPolicy = MODEL_RETRY):
    graph = StateGraph(SimState, context_schema=SimContext)
    graph.add_node("plan", plan, retry_policy=retry)
    graph.add_node("caller", add_turn, retry_policy=retry)
    graph.add_node("receiver", add_turn, retry_policy=retry)
    graph.add_node("finish", finish, retry_policy=retry)
    graph.add_edge(START, "plan")
    # Each node lists only the moves it can make, so the drawing shows the real call and a
    # routing mistake fails loudly instead of taking a wrong turn.
    # plan can lead to any of them: a resumed call goes straight to whoever speaks next
    graph.add_conditional_edges(
        "plan", next_step, {"caller": "caller", "receiver": "receiver", "finish": "finish"}
    )
    graph.add_conditional_edges("caller", next_step, {"receiver": "receiver", "finish": "finish"})
    graph.add_conditional_edges("receiver", next_step, {"caller": "caller", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile()  # no checkpointer: the store already saves the call every step


CALL_GRAPH = build_graph()
