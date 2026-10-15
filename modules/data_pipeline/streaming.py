"""Run one call and report it step by step, for the /studio page.

`stream_call` does what `simulate.run_call` does, but drives the same LangGraph graph with
`graph.stream(stream_mode="updates")` and yields an Event for every step the browser should
show: "thinking" before an agent starts, then "plan", "turn", "ended" (the call stopped early)
or "verdict" with the saved record.
The call is saved by the graph nodes exactly as in a command line run.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from langgraph.runtime import Runtime

from .agents import Agents
from .capture import Capture, ModuleRun
from .config import Config, ReceiverProfile, Scenario
from .fill import SlotFiller
from .records import CallRecord, Generation
from .scenarios import CallConfig, sample_config
from .simulate import (
    CALL_GRAPH,
    SimContext,
    _open_record,
    call_id,
    initial_state,
    is_model_error,
    next_speaker,
)
from .store import Store


class CallConflict(ValueError):
    """The call cannot be run: it is finished, or it was started with other settings."""


@dataclass
class CallSpec:
    """What the studio form asks for."""

    call_type: str
    seed: int
    scenario: Scenario | None = None  # None: the one the seed picks
    receiver: ReceiverProfile | None = None  # None: the one the seed picks
    receiver_name: str = ""  # the name the caller hears; "" keeps the one the seed picks

    @property
    def record_id(self) -> str:
        return call_id(self.call_type, self.seed, self.receiver)


@dataclass
class Event:
    """One step for the studio page to show."""

    kind: str  # thinking | plan | turn | ended | verdict
    agent: str  # who is working or worked: planner, scam_caller, legit_caller, receiver, judge
    record: dict  # the call as the store keeps it (CallRecord.to_dict)
    index: int | None = None  # turn: 0-based position
    modules: list[ModuleRun] = field(default_factory=list)


def stream_call(
    spec: CallSpec,
    store: Store,
    agents: Agents,
    config: Config,
    voices: Any,
    capture: Capture,
    cancel: threading.Event | None = None,
) -> Iterator[Event]:
    """Run (or resume) the call and yield what happens. Stops early when `cancel` is set; the
    call stays in the store as unfinished and a command line run can resume it."""
    context = _open_context(spec, store, agents, config, voices)
    record = context.record
    # A resumed call shows what is already there, in call order: its plan, then its turns.
    plan_shown = record.generation.briefing is not None
    if plan_shown:
        yield Event("plan", "planner", record.to_dict())
    for index, turn in enumerate(record.generation.turns):
        yield Event("turn", turn.agent, record.to_dict(), index)
    if record.label_scam and record.generation.briefing is None:
        pending = Event("thinking", "planner", record.to_dict())
    else:
        pending = _thinking(record, context)
    yield pending
    written = len(record.generation.turns)

    updates = CALL_GRAPH.stream(
        initial_state(record),
        context=context,
        config={"recursion_limit": context.call.max_turns + 5},
        stream_mode="updates",
    )
    for update in updates:
        (node,) = update  # stream_mode="updates" gives {node name: its update}, one per step
        record = store.load_call(record.id)  # as the node saved it
        modules = capture.take()
        turns = record.generation.turns
        if node == "plan":
            if record.generation.briefing is not None and not plan_shown:
                yield Event("plan", "planner", record.to_dict(), modules=modules)
        elif node == "finish":
            yield Event("verdict", "judge", record.to_dict(), modules=modules)
            return
        elif len(turns) == written:
            # the turn node ended the call early (a prompt that does not fit): no turn was added
            yield Event("ended", pending.agent, record.to_dict(), written, modules)
            continue  # the finish step follows at once
        else:
            written = len(turns)
            yield Event("turn", turns[-1].agent, record.to_dict(), written - 1, modules)
        if cancel is not None and cancel.is_set():
            return
        pending = _thinking(record, context)
        yield pending


def _open_context(
    spec: CallSpec, store: Store, agents: Agents, config: Config, voices: Any
) -> SimContext:
    """The context the graph runs with: the stored call, or a new one with the form's choices."""
    call = sample_config(spec.call_type, spec.seed, config, spec.receiver)
    if spec.scenario is not None:
        call = replace(call, scenario=spec.scenario)
    record = _open_record(store, call, config, agents, spec.receiver)
    if record.generation.status != "in_progress":
        raise CallConflict(f"{record.id} is already finished")
    call, name = _resume_overrides(spec, call, record.generation)
    # the overrides stay with the stored call
    stored_call = {**record.generation.call, "receiver_name": name}
    record = replace(record, generation=replace(record.generation, call=stored_call))
    filler = SlotFiller(call, config.details)
    if name:
        filler.values["RECEIVER_NAME"] = name
    return SimContext(agents, config, call, store, voices, filler, record)


def _thinking(record: CallRecord, context: SimContext) -> Event:
    """The "thinking" event for the agent that speaks next (the judge when the call is over)."""
    step = next_speaker(initial_state(record), Runtime(context=context))
    caller = "scam_caller" if context.call.label_scam else "legit_caller"
    agent = {"caller": caller, "receiver": "receiver", "finish": "judge"}[step]
    return Event("thinking", agent, record.to_dict())


def _resume_overrides(
    spec: CallSpec, call: CallConfig, generation: Generation
) -> tuple[CallConfig, str]:
    """The call and the receiver name to use. A call with no turns yet takes what the form asked
    for. A call being resumed keeps the scenario and name it was started with: the form may leave
    them empty but may not ask for others, because the turns already written used the old ones."""
    stored = generation.call
    if not generation.turns and generation.briefing is None:
        return call, spec.receiver_name
    scenario = Scenario.model_validate(stored["scenario"])
    name = stored.get("receiver_name", "")
    if spec.scenario is not None and spec.scenario.key != scenario.key:
        raise CallConflict(
            f"{spec.record_id} was started with scenario {scenario.key}. Resume it with that "
            "scenario or none, or pick another seed."
        )
    if spec.receiver_name and spec.receiver_name != name:
        raise CallConflict(
            f"{spec.record_id} was started with another receiver name. Resume it with that "
            "name or none, or pick another seed."
        )
    return replace(call, scenario=scenario), name


def mark_model_error(store: Store, record_id: str, exc: BaseException) -> None:
    """Same as the command line run: a model error that survived the retries rejects the call
    (llm_error:...). Any other error leaves the call unfinished so it can be resumed."""
    if is_model_error(exc):
        store.reject_after_model_error(record_id, exc)


def next_free_seed(call_type: str, taken_ids: set[str], receiver: ReceiverProfile | None) -> int:
    """The smallest seed whose call id is not in the store yet."""
    return next(
        seed for seed in itertools.count() if call_id(call_type, seed, receiver) not in taken_ids
    )
