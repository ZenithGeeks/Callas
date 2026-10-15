"""Act out one call turn by turn and keep it in the SQLite store.

The call runs as a LangGraph graph (build_graph; the README has the drawing):

    START -> plan -> caller <-> receiver -> finish -> END

`plan` writes the scam caller's plan (scam calls only), `caller` and `receiver` add one turn
each, a routing function picks whose turn is next or ends the call, and `finish` builds the
text, joins the audio and asks the judge. Each node saves the call to the store before it
returns.

The store is the only state. A call is loaded from it (or created), every turn is built from the
turns already stored, and the call is written back after every turn, so a stopped run picks up
where it left off. Records follow the spec's schema; `generation` and `review` are extra and are
dropped when the dataset is built.

Before a scam call starts, the planner writes the caller's plan from the scenario and the newest
pattern notes for its scam type (patterns.py). The plan is stored with the call.

What the agents are told comes from the config (config.py); this file only decides whose turn
it is, fills slots, speaks each line, and checks the finished call.

Naming: `call` is the CallConfig of one call (scenarios.py); `config` is the loaded YAML Config.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

import dspy
import openai
from dspy.utils.exceptions import AdapterParseError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy

from llm_serving.client import LLMUnavailable
from llm_serving.dspy_programs import _UNAVAILABLE as UNAVAILABLE

from .agents import Agents
from .config import Config, ConfigError
from .fill import SLOT_RE, SlotFiller
from .patterns import render_notes
from .scenarios import CallConfig, ReceiverProfile, sample_config
from .store import RECORD_SCHEMA, Store
from .tts import Voices, join_call, pick_voices

log = logging.getLogger("data_pipeline")

CALLER, RECEIVER = "A", "B"  # speaker ids in the record; the caller always speaks first


def is_model_error(exc: BaseException) -> bool:
    """True for a failure that comes from the model's answer or from rate limiting: a reply
    DSPy could not parse, or a 429 / 5xx answer. Anything else (a bug, a bad config, a store
    error) is not the model's fault, so it is neither retried nor blamed on the model."""
    if isinstance(exc, (AdapterParseError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    ):
        return True
    # DSPy wraps rate limits in its own classes (LMRateLimitError, lm15.RateLimitError)
    return any("RateLimit" in cls.__name__ for cls in type(exc).__mro__)


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


# ---- one call --------------------------------------------------------------------------------


def run_call(
    call_type: str,
    seed: int,
    store: Store,
    agents: Agents,
    config: Config,
    voices: Voices | None = None,
    receiver: ReceiverProfile | None = None,
) -> dict:
    """Generate (or finish) one call and return its record.

    `receiver` replaces the sampled receiver with a person chosen by hand; such calls get
    their own id (see record_id), so they never overwrite the sampled call for that seed.
    With `voices`, each turn is spoken as soon as it is written (audio/<id>/NN_<speaker>.wav),
    and the finished call is joined into audio/<id>.wav. Audio paths are relative to the
    store's folder.
    """
    call = sample_config(call_type, seed, config, receiver)
    record = _open_record(store, call, config, agents, receiver)
    if record["generation"]["status"] != "in_progress":
        return record
    context = SimContext(
        agents=agents,
        config=config,
        call=call,
        store=store,
        voices=voices,
        filler=SlotFiller(call, config.details),
    )
    # One step per turn plus plan and finish; a call that runs past this is a bug.
    limit = call.max_turns + 5
    result = CALL_GRAPH.invoke(
        {"record": record}, context=context, config={"recursion_limit": limit}
    )
    return result["record"]


def record_id(call_type: str, seed: int, receiver_key: str = "") -> str:
    """romance_00001, or romance_teacher_cm_00001 for a call with a hand-picked receiver."""
    middle = f"_{receiver_key}" if receiver_key else ""
    return f"{call_type}{middle}_{seed:05d}"


def call_id(call_type: str, seed: int, receiver: ReceiverProfile | None = None) -> str:
    """The id of a call. The receiver's key is part of it only when the receiver was chosen
    by hand (--receiver, --receiver-file); a sampled receiver, even a hand-written profile the
    sampler drew, does not change the id."""
    return record_id(call_type, seed, receiver.key if receiver else "")


def receiver_hash(receiver: ReceiverProfile | None) -> str:
    """Short hash of a hand-chosen receiver's fields, "" when the receiver was sampled."""
    if receiver is None:
        return ""
    text = json.dumps(receiver.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def new_record(
    call: CallConfig,
    config: Config,
    agents: Agents,
    receiver: ReceiverProfile | None = None,
) -> dict[str, Any]:
    """`receiver` is the hand-chosen receiver, if any: it decides the id and is remembered
    (as a hash in generation.call) so a resume can tell it from a different one."""
    caller_voice, receiver_voice = pick_voices(
        config.agents.tts.voices,
        call.caller_gender,
        call.receiver.gender,
        f"{call.call_type}:{call.seed}",
    )
    return {
        "id": call_id(call.call_type, call.seed, receiver),
        "channel": "call",
        "text": "",
        "audio_path": None,
        "speaker_turns": [],
        "label_scam": call.label_scam,
        "label_type": call.label_type,
        "entities": [],
        "reference_summary": "",
        "scam_start_turn": None,
        "source": "synthetic",
        "generation": {
            "schema": RECORD_SCHEMA,
            "status": "in_progress",
            "reasons": [],
            "config_name": config.name,
            "config_hash": config.hash,
            "models": agents.models(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "call": {**call.to_dict(), "receiver_override": receiver_hash(receiver)},
            "voices": {CALLER: caller_voice, RECEIVER: receiver_voice},
            "briefing": None,
            "turns": [],
            "scam_start_from": None,
            "judge": None,
        },
        "review": {"status": "pending", "note": "", "reviewed_at": None},
    }


def _open_record(
    store: Store,
    call: CallConfig,
    config: Config,
    agents: Agents,
    receiver: ReceiverProfile | None = None,
) -> dict:
    """The stored call to resume, or a new one. A half-done call needs its original config
    and the same hand-chosen receiver (or none) it was started with."""
    wanted_id = call_id(call.call_type, call.seed, receiver)
    record = store.load(wanted_id)
    if record is None:
        return new_record(call, config, agents, receiver)
    generation = record["generation"]
    if generation["status"] != "in_progress":
        return record
    made_with = generation.get("config_hash")
    if made_with != config.hash:
        raise ConfigError(
            f"{wanted_id} was started with config {made_with}, not {config.hash}; "
            "finish it with the old config, or delete it from the store"
        )
    stored = generation["call"].get("receiver_override", "")
    if stored != receiver_hash(receiver):
        raise ConfigError(
            f"{wanted_id} was started with a different receiver profile "
            f"(stored {stored or 'none'}, now {receiver_hash(receiver) or 'none'}); "
            "use the original receiver, or delete the call from the store"
        )
    return record


def _plan(agents: Agents, call: CallConfig, config: Config, store: Store) -> dict:
    """The scam caller's plan for this call, from its scenario and recent pattern notes."""
    notes = store.notes(call.call_type, limit=config.agents.planner.notes_limit)
    last = (call.max_turns - 1) // 2
    scenario = call.scenario
    turns = (
        f"The caller has {last + 1} turns. The first ask may come on turn "
        f"{call.reveal_turn + 1} at the earliest."
    )
    with dspy.context(lm=agents.lms["planner"]):
        pred = _model_call(
            lambda: agents.planner(
                scam_type=config.agents.judge.intended[call.call_type],
                role=scenario.role,
                goal=scenario.goal,
                tactics=scenario.tactics,
                pattern_notes=render_notes(notes),
                turns=turns,
            )
        )
    return {
        "plan": [step.strip() for step in pred.plan if step.strip()],
        # summaries, not ids: a wiki sync re-creates the notes and gives them new ids
        "notes_used": [note["summary"] for note in notes],
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
    }


def _next_turn(
    agents: Agents,
    call: CallConfig,
    config: Config,
    turns: list[dict],
    filler: SlotFiller,
    plan: list[str],
) -> dict:
    """Ask the agent whose turn it is for one line, fill its slots, and describe the turn."""
    started = time.perf_counter()
    speaker, agent_name, pred = _model_call(lambda: _speak(agents, call, config, turns, plan))
    raw = (pred.line or "").strip()
    text, entities, problems = filler.fill(raw)
    if not text:
        problems.append("empty_line")
    return {
        "speaker": speaker,
        "agent": agent_name,
        "raw": raw,
        "text": text,
        "entities": entities,
        "asks_now": bool(getattr(pred, "asks_now", False)),
        "end_call": bool(pred.end_call),
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
        "problems": problems,
        "spelling": spelling(raw),
        "latency_s": round(time.perf_counter() - started, 3),
        "audio": None,
        "audio_s": None,
    }


def _finish_call(
    record: dict, agents: Agents, config: Config, voices: Voices | None, out_dir: Path
) -> None:
    generation = record["generation"]
    _finish_text(record)
    if voices is not None:
        _join_audio(record, out_dir, config.agents.tts.gap_s)
    generation["judge"] = _judge(agents, config, record)
    record["reference_summary"] = generation["judge"]["summary"]
    if record["label_scam"]:
        record["scam_start_turn"], generation["scam_start_from"] = _scam_start(record)
    generation["reasons"] = _verdict(record, config)
    generation["status"] = "rejected" if generation["reasons"] else "accepted"


# ---- what each agent is given ----------------------------------------------------------------


def _speak(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict], plan: list[str]
) -> tuple[str, str, dspy.Prediction]:
    """Run the agent whose turn it is. Returns (speaker, agent name, prediction)."""
    if len(turns) % 2 == 1:
        return RECEIVER, "receiver", _receiver_says(agents, call, config, turns)
    agent_name, pred = _caller_says(agents, call, config, turns, plan)
    return CALLER, agent_name, pred


def _receiver_says(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict]
) -> dspy.Prediction:
    own_turn = len(turns) // 2
    with dspy.context(lm=agents.lms["receiver"]):
        return agents.receiver(
            # the profile key is for choosing a profile on the command line, not for the model
            persona=call.receiver.model_dump(exclude={"key"}),
            voice=config.agents.speakers.voice[call.receiver.gender],
            stage=stage(config, own_turn),
            conversation=_receiver_view(turns, call, config),
        )


def _receiver_view(turns: list[dict], call: CallConfig, config: Config) -> str:
    """The call as the receiver hears it: real values, except its own name, which stays the
    {RECEIVER_NAME} slot that its instructions call "your name". Without this the receiver did
    not recognise its own name and answered "you have the wrong number"."""
    own_name = SlotFiller(call, config.details).values["RECEIVER_NAME"]
    return conversation(turns, config, raw=False).replace(own_name, "{RECEIVER_NAME}")


def _caller_says(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict], plan: list[str]
) -> tuple[str, dspy.Prediction]:
    own_turn = len(turns) // 2
    scenario = call.scenario
    inputs = dict(
        role=scenario.role,
        voice=config.agents.speakers.voice[call.caller_gender],
        org="{ORG}" if scenario.org else "",
        goal=scenario.goal,
        receiver_name="{RECEIVER_NAME}" if call.caller_knows_name else "unknown",
        conversation=conversation(turns, config, raw=True),
    )
    if call.label_scam:
        with dspy.context(lm=agents.lms["scam_caller"]):
            pred = agents.scam_caller(
                **inputs,
                tactics=scenario.tactics,
                pacing=pacing(call, config, own_turn),
                plan=plan,
            )
        return "scam_caller", pred
    with dspy.context(lm=agents.lms["legit_caller"]):
        pred = agents.legit_caller(**inputs, manner=scenario.tactics)
    return "legit_caller", pred


def conversation(turns: list[dict], config: Config, *, raw: bool) -> str:
    """The call so far as Thai lines. Callers see their slots ({PHONE}); receivers see values."""
    speakers = config.agents.speakers
    if not turns:
        return speakers.empty_call
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    key = "raw" if raw else "text"
    return "\n".join(f"{tag[turn['speaker']]}: {turn[key]}" for turn in turns)


def pacing(call: CallConfig, config: Config, own_turn: int) -> str:
    """Tell the scam caller where it is in the call: before, at, or after its reveal turn."""
    texts = config.agents.scam_caller.pacing
    last = (call.max_turns - 1) // 2  # the caller's last possible own turn (0-based)
    if own_turn >= last - 1 and own_turn >= call.reveal_turn:
        text = texts.last
    elif own_turn < call.reveal_turn:
        text = texts.before
    elif own_turn == call.reveal_turn:
        text = texts.start
    else:
        text = texts.push
    return text.format(turn=own_turn + 1, turns=last + 1)


def stage(config: Config, own_turn: int) -> str:
    receiver = config.agents.receiver
    text = receiver.stage.stay if own_turn < receiver.stay_on_line else receiver.stage.free
    return text.format(turn=own_turn + 1)


# ---- model calls -----------------------------------------------------------------------------


def _model_call(fn):
    """Run one DSPy call. A connection failure becomes LLMUnavailable, which stops the run
    instead of being retried (see MODEL_RETRY)."""
    try:
        return fn()
    except UNAVAILABLE as exc:
        raise LLMUnavailable(f"cannot reach LLM server: {type(exc).__name__}") from exc


def spelling(raw: str) -> list[str]:
    """Words that look misspelled. Checked on the line before filling, so names are skipped."""
    from tts_voice_agent.spelling import spelling_problems

    return spelling_problems(SLOT_RE.sub(" ", raw))


# ---- finishing a call ------------------------------------------------------------------------


def _finish_text(record: dict) -> None:
    """Build text, speaker_turns and entity spans from the turns."""
    lines, spans, offset = [], [], 0
    turns = record["generation"]["turns"]
    for turn in turns:
        prefix = f"{turn['speaker']}: "
        for entity in turn["entities"]:
            start = offset + len(prefix) + entity["start"]
            spans.append({**entity, "start": start, "end": start + len(entity["text"])})
        line = prefix + turn["text"]
        lines.append(line)
        offset += len(line) + 1  # newline
    record["text"] = "\n".join(lines)
    record["speaker_turns"] = [
        {"speaker": turn["speaker"], "start": None, "end": None, "text": turn["text"]}
        for turn in turns
    ]
    record["entities"] = spans


def _speak_turn(voices: Voices, record: dict, index: int, out_dir: Path) -> None:
    turn = record["generation"]["turns"][index]
    if not turn["text"]:
        return
    rel = f"audio/{record['id']}/{index:02d}_{turn['speaker']}.wav"
    voice = record["generation"]["voices"][turn["speaker"]]
    try:
        turn["audio_s"] = round(voices.speak(turn["text"], voice, out_dir / rel), 3)
        turn["audio"] = rel
    except Exception as exc:  # a turn without audio is flagged, the call text is still kept
        log.warning("tts failed for %s turn %d: %s", record["id"], index, type(exc).__name__)
        turn["problems"].append("tts_failed")


def _join_audio(record: dict, out_dir: Path, gap_s: float) -> None:
    turns = record["generation"]["turns"]
    if not turns or any(turn["audio"] is None for turn in turns):
        return
    rel = f"audio/{record['id']}.wav"
    times = join_call([out_dir / turn["audio"] for turn in turns], out_dir / rel, gap_s)
    record["audio_path"] = rel
    for speaker_turn, (start, end) in zip(record["speaker_turns"], times, strict=True):
        speaker_turn["start"], speaker_turn["end"] = start, end


def _judge(agents: Agents, config: Config, record: dict) -> dict:
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    transcript = "\n".join(
        f"[{i}] {tag[turn['speaker']]}: {turn['text']}"
        for i, turn in enumerate(record["generation"]["turns"])
    )
    intended = config.agents.judge.intended[record["label_type"] or "benign"]
    with dspy.context(lm=agents.lms["judge"]):
        pred = _model_call(lambda: agents.judge(intended=intended, transcript=transcript))
    return {
        "label_holds": bool(pred.label_holds),
        "scam_start_turn": int(pred.scam_start_turn),
        "natural_thai": int(pred.natural_thai),
        "giveaway": bool(pred.giveaway),
        "summary": (pred.summary or "").strip(),
        "issues": (pred.issues or "").strip(),
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
    }


def _scam_start(record: dict) -> tuple[int | None, str | None]:
    """First turn where the scam shows: the caller's own flag, else the judge's answer.

    Callers often under-report their own ask (a link "to check the parcel" is an ask), while
    the judge reads the whole call. The source is kept so the two can be compared later.
    """
    turns = record["generation"]["turns"]
    reveal = record["generation"]["call"]["reveal_turn"]
    flagged = None
    for i, turn in enumerate(turns):
        if turn["speaker"] != CALLER or not turn["asks_now"]:
            continue
        if i // 2 < reveal:  # the plan forbids asking before the reveal turn: not a scam move
            log.warning(
                "%s turn %d: asks_now before reveal turn %d, ignored", record["id"], i, reveal
            )
            continue
        flagged = i
        break
    if flagged is not None:
        return flagged, "caller"
    judge = record["generation"]["judge"]
    at = judge["scam_start_turn"]
    # the judge must point at a caller turn: the receiver never makes a scam move
    if judge["label_holds"] and 0 <= at < len(turns) and turns[at]["speaker"] == CALLER:
        return at, "judge"
    return None, None


def _verdict(record: dict, config: Config) -> list[str]:
    """Reasons to reject the call; empty means accepted."""
    generation, reasons = record["generation"], []
    turns = generation["turns"]
    if len(turns) < config.scenarios.calls.min_turns:
        reasons.append("too_short")
    reasons += sorted({problem for turn in turns for problem in turn["problems"]})
    if any("tone_mark_order" in turn["spelling"] for turn in turns):
        reasons.append("tone_mark_order")
    judge = generation["judge"]
    if not judge["label_holds"]:
        reasons.append("judge_label_does_not_hold")
    if judge["giveaway"]:
        reasons.append("judge_giveaway")
    if record["label_scam"] and record["scam_start_turn"] is None:
        reasons.append("no_scam_move")
    if not record["label_scam"] and judge["scam_start_turn"] >= 0:
        reasons.append("judge_saw_scam_in_normal_call")
    return reasons


# ---- the call graph (LangGraph) --------------------------------------------------------------


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


def plan_node(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Scam calls only: write the caller's plan once, before the first turn."""
    ctx, record = runtime.context, state["record"]
    if ctx.call.label_scam and record["generation"]["briefing"] is None:
        record["generation"]["briefing"] = _plan(ctx.agents, ctx.call, ctx.config, ctx.store)
    ctx.store.save(record)
    return {"record": record}


def turn_node(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Add one turn (caller or receiver, by turn order), speak it, save the call.

    Safe to retry: the turn is built first, from the record as it was; only then is it added to
    a copy of the record, which becomes the new state once the node returns. If anything after
    the model answer fails and LangGraph runs the node again, the state still has the old
    turns, so a turn is never added twice or given the wrong speaker."""
    ctx = runtime.context
    old = state["record"]["generation"]
    turn = _next_turn(
        ctx.agents,
        ctx.call,
        ctx.config,
        old["turns"],
        ctx.filler,
        (old["briefing"] or {}).get("plan", []),
    )
    record = copy.deepcopy(state["record"])
    turns = record["generation"]["turns"]
    turns.append(turn)
    if ctx.voices is not None:
        _speak_turn(ctx.voices, record, len(turns) - 1, ctx.store.folder)
    ctx.store.save(record)
    return {"record": record}


def finish_node(state: SimState, runtime: Runtime[SimContext]) -> SimState:
    """Build the text, join the audio, ask the judge, and set accepted or rejected."""
    ctx, record = runtime.context, state["record"]
    _finish_call(record, ctx.agents, ctx.config, ctx.voices, ctx.store.folder)
    ctx.store.save(record)
    return {"record": record}


def next_step(
    state: SimState, runtime: Runtime[SimContext]
) -> Literal["caller", "receiver", "finish"]:
    """Whose turn it is, or "finish" when someone hung up or the call reached its length."""
    turns = state["record"]["generation"]["turns"]
    if len(turns) >= runtime.context.call.max_turns or (turns and turns[-1]["end_call"]):
        return "finish"
    return "caller" if len(turns) % 2 == 0 else "receiver"


def build_graph(retry: RetryPolicy = MODEL_RETRY):
    graph = StateGraph(SimState, context_schema=SimContext)
    graph.add_node("plan", plan_node, retry_policy=retry)
    graph.add_node("caller", turn_node, retry_policy=retry)
    graph.add_node("receiver", turn_node, retry_policy=retry)
    graph.add_node("finish", finish_node, retry_policy=retry)
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
