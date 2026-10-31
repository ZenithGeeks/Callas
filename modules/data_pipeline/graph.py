"""The call graph: one simulated phone call, turn by turn.

    START -> plan -> caller <-> receiver -> finish -> END

Read this file first. records.py has the call record and its parts, turns.py writes a turn,
finish.py checks the finished call, audio.py speaks it, and simulate.run_call loads the call
from the store and runs this graph.

`plan` is the node that writes the Briefing; the steps it holds are `Briefing.plan`, called
`plan_steps` where a turn reads them. `call` is the CallConfig (scenarios.py), `record` the
stored CallRecord.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass, replace
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from .agents import Agents
from .audio import speak_turn
from .budget import ContextTooLong
from .config import Config
from .fill import SlotFiller
from .finish import Ending, ended_early, finish_call
from .records import Briefing, CallRecord, GenerationStatus, Turn
from .scenarios import CallConfig
from .store import Store
from .tts import Voices
from .turns import CALLER, RECEIVER, is_model_error, read_footing, write_plan, write_turn

log = logging.getLogger("data_pipeline")


class CallState(TypedDict):
    """What changes while a call runs. Each node returns only the fields it changed."""

    briefing: Briefing | None  # the scam caller's plan; None for a normal call
    footing: str | None  # the receiver's footing; None until read on its first turn
    turns: Annotated[list[Turn], operator.add]  # a node returns the one turn it adds
    status: GenerationStatus
    ending: Ending | None  # the text, audio, judge, leaks and reasons, once the call is over


@dataclass
class SimContext:
    """What the nodes need that is not state, read as `runtime.context`."""

    agents: Agents
    config: Config
    call: CallConfig
    store: Store
    voices: Voices | None  # None: text only
    filler: SlotFiller  # this call's values for the {SLOTS}
    record: CallRecord  # the call as loaded or created; to_record() lays the state over it


def initial_state(record: CallRecord) -> CallState:
    """The state of a stored call that is new or still in progress."""
    generation = record.generation
    return {
        "briefing": generation.briefing,
        "footing": generation.footing,
        "turns": generation.turns,
        "status": generation.status,
        "ending": None,
    }


def to_record(record: CallRecord, state: CallState) -> CallRecord:
    """The call as the store keeps it: `record` with the state written into it."""
    generation = replace(
        record.generation,
        briefing=state["briefing"],
        footing=state["footing"],
        turns=state["turns"],
        status=state["status"],
    )
    ending = state["ending"]
    if ending is None:
        return replace(record, generation=generation)
    generation = replace(
        generation,
        receiver_leaked=ending.receiver_leaked,
        first_leak_turn=ending.first_leak_turn,
        scam_start_from=ending.scam_start_from,
        judge=ending.judge,
        reasons=ending.reasons,
    )
    review = record.review
    if state["status"] in ("accepted", "rejected"):
        auto_status = "approved" if state["status"] == "accepted" else "rejected"
        conf = (
            ending.judge.confidence
            if ending.judge and ending.judge.confidence is not None
            else 0.0
        )
        conf_pct = f"{int(round(conf * 100))}%"
        if auto_status == "approved":
            note = f"Auto-approved by Judge (confidence: {conf_pct})"
        else:
            reasons_str = ", ".join(ending.reasons) if ending.reasons else "Failed checks"
            issues = (
                f" - {ending.judge.issues}"
                if ending.judge and ending.judge.issues and ending.judge.issues != "none"
                else ""
            )
            note = f"Auto-rejected by Judge (confidence: {conf_pct}): {reasons_str}{issues}"
        review = replace(review, status=auto_status, note=note)

    return replace(
        record,
        text=ending.text,
        speaker_turns=ending.speaker_turns,
        entities=ending.entities,
        audio_path=ending.audio_path,
        reference_summary=ending.reference_summary,
        scam_start_turn=ending.scam_start_turn,
        generation=generation,
        review=review,
    )


def plan(state: CallState, runtime: Runtime[SimContext]) -> dict:
    """Scam calls only: write the caller's plan once, before the first turn."""
    ctx = runtime.context
    update = {}
    if ctx.call.label_scam and state["briefing"] is None:
        update["briefing"] = write_plan(ctx.agents, ctx.call, ctx.config, ctx.store)
    ctx.store.save_call(to_record(ctx.record, {**state, **update}))
    return update


def caller(state: CallState, runtime: Runtime[SimContext]) -> dict:
    """Add the caller's next line."""
    return _add_turn(CALLER, state, runtime.context)


def receiver(state: CallState, runtime: Runtime[SimContext]) -> dict:
    """Add the receiver's next line."""
    return _add_turn(RECEIVER, state, runtime.context)


def _add_turn(speaker: str, state: CallState, ctx: SimContext) -> dict:
    """Write the speaker's turn, speak it and save the call.

    Safe to retry: the call is saved with the new turn before the node returns, and LangGraph
    adds the turn to the state only once the node has returned. If the save fails and the node
    runs again, the state still has the old turns, so a turn is never added twice.
    """
    turns = state["turns"]
    plan_steps = state["briefing"].plan if state["briefing"] else []  # a normal call has none
    update = {}
    if speaker == RECEIVER and state["footing"] is None:
        # Read once, on the receiver's first turn, and saved with the call, so the register never
        # drifts and a resume reuses it instead of asking again. "" (a stranger) counts as read.
        update["footing"] = read_footing(ctx.agents, ctx.call, ctx.config, turns, ctx.filler)
    footing = update.get("footing", state["footing"]) or ""
    try:
        turn = write_turn(
            ctx.agents, ctx.call, ctx.config, turns, ctx.filler, plan_steps, speaker, footing
        )
    except ContextTooLong as exc:
        # The prompt cannot be made to fit, so the model is not asked. The call ends here,
        # rejected, with the turns it has.
        log.warning("%s: %s", ctx.record.id, exc)
        update |= {"status": "rejected", "ending": ended_early(turns)}
        ctx.store.save_call(to_record(ctx.record, {**state, **update}))
        return update
    if ctx.voices is not None:
        turn = speak_turn(ctx.voices, turn, len(turns), ctx.record, ctx.store.folder)
    ctx.store.save_call(to_record(ctx.record, {**state, **update, "turns": [*turns, turn]}))
    return {**update, "turns": [turn]}


def finish(state: CallState, runtime: Runtime[SimContext]) -> dict:
    """Build the text, join the audio, ask the judge, and set accepted or rejected."""
    ctx = runtime.context
    update = {}
    # Not in progress: a turn ended the call early and already put ended_early() in the state.
    if state["status"] == "in_progress":
        record = to_record(ctx.record, state)
        # the judge reads the wiki notes the planner read, so both check against the same patterns
        briefing = record.generation.briefing
        notes = (
            ctx.store.notes_named(record.label_type, briefing.notes_used)
            if record.label_scam and briefing
            else []
        )
        ending = finish_call(record, ctx.agents, ctx.config, ctx.voices, ctx.store.folder, notes)
        update = {"ending": ending, "status": "rejected" if ending.reasons else "accepted"}
    finished = to_record(ctx.record, {**state, **update})
    ctx.store.save_call(finished)
    if finished.generation.status in ("accepted", "rejected"):
        ctx.store.set_review(finished.id, finished.review.status, finished.review.note)
    return update


def next_speaker(
    state: CallState, runtime: Runtime[SimContext]
) -> Literal["caller", "receiver", "finish"]:
    """Whose turn it is, or "finish" when the call ended, someone hung up or it is long enough."""
    turns = state["turns"]
    if state["status"] != "in_progress":
        return "finish"
    if len(turns) >= runtime.context.call.max_turns or (turns and turns[-1].end_call):
        return "finish"
    return "caller" if len(turns) % 2 == 0 else "receiver"


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


def build_graph(retry: RetryPolicy = MODEL_RETRY):
    """Wire the call graph. It has no checkpointer: every node saves the call to the store."""
    builder = StateGraph(CallState, context_schema=SimContext)
    builder.add_node(plan, retry_policy=retry)
    builder.add_node(caller, retry_policy=retry)
    builder.add_node(receiver, retry_policy=retry)
    builder.add_node(finish, retry_policy=retry)
    builder.add_edge(START, "plan")
    # Each node lists only the moves it can make, so the drawing shows the real call and a
    # routing mistake fails loudly. plan can lead to any of them: a resumed call goes straight
    # to whoever speaks next.
    builder.add_conditional_edges("plan", next_speaker, ["caller", "receiver", "finish"])
    builder.add_conditional_edges("caller", next_speaker, ["receiver", "finish"])
    builder.add_conditional_edges("receiver", next_speaker, ["caller", "finish"])
    builder.add_edge("finish", END)
    return builder.compile()


CALL_GRAPH = build_graph()
