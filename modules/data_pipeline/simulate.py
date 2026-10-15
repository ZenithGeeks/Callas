"""Act out one call turn by turn and keep it in the SQLite store.

`run_call` loads the call from the store (or creates it) and runs the call graph (graph.py) on
it. The store is the only state: every node saves the call, so a stopped run picks up where it
left off. Records follow the spec's schema; `generation` and `review` are extra and are dropped
when the dataset is built.

Naming: `call` is the CallConfig of one call (scenarios.py); `config` is the loaded YAML Config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .agents import Agents
from .config import Config, ConfigError, Scenario
from .fill import SlotFiller
from .graph import (
    CALL_GRAPH,
    SimContext,
    build_graph,
    initial_state,
    next_speaker,
    retry_policy,
    to_record,
)
from .scenarios import CallConfig, ReceiverProfile, sample_config
from .store import RECORD_SCHEMA, Store
from .tts import Voices, pick_voices
from .turns import CALLER, RECEIVER, is_model_error, speech_style

# Names other modules and the tests import from here, although they live in graph.py or turns.py.
__all__ = [
    "CALL_GRAPH",
    "SimContext",
    "build_graph",
    "call_id",
    "initial_state",
    "is_model_error",
    "new_record",
    "next_speaker",
    "receiver_hash",
    "record_id",
    "retry_policy",
    "run_call",
]


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
    call, filler = _as_started(call, record, config)
    context = SimContext(agents, config, call, store, voices, filler, record)
    # One step per turn plus plan and finish; a call that runs past this is a bug.
    limit = call.max_turns + 5
    state = CALL_GRAPH.invoke(
        initial_state(record), context=context, config={"recursion_limit": limit}
    )
    return to_record(record, state)


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
    """A new, empty call record for `call`.

    `receiver` is the hand-chosen receiver, if any: it decides the id and is remembered (as a
    hash in generation.call) so a resume can tell it from a different one.
    """
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
            "speech_style": speech_style(call, config),
            "briefing": None,
            "turns": [],
            "receiver_leaked": [],
            "first_leak_turn": None,
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


def _as_started(call: CallConfig, record: dict, config: Config) -> tuple[CallConfig, SlotFiller]:
    """The call and its slot values as the record was started. The studio can pick another
    scenario and receiver name than the seed does; both are kept in generation.call, so a
    command line resume must use them too (#63)."""
    stored = record["generation"]["call"]
    call = replace(call, scenario=Scenario.model_validate(stored["scenario"]))
    filler = SlotFiller(call, config.details)
    if stored.get("receiver_name"):
        filler.values["RECEIVER_NAME"] = stored["receiver_name"]
    return call, filler
