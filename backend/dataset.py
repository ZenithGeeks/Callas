"""The /dataset review page: read simulated calls from the SQLite store, save the review.

Calls live in the store written by `python -m data_pipeline.synthesize` (default
data/synthetic/calls.db; set CALLAS_DATASET_DB to read another). This module only reads calls
and changes their review; it never touches the call itself. Audio files sit in an audio/ folder
next to the database. Ids and audio names are checked against strict patterns before any path
is built, so a request can only reach files inside that folder.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

from data_pipeline.store import Store

from . import views

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "calls.db"
ID_RE = re.compile(r"[a-z_]+_\d{5}")
TURN_AUDIO_RE = re.compile(r"\d{2}_[AB]\.wav")
REVIEW_STATES = ("pending", "approved", "rejected")
MAX_NOTE_CHARS = 500

CALL_TYPE_NAMES = {**{k: v for k, v in views.TYPE_NAMES.items() if k}, "benign": "Normal call"}
SPEAKER_NAMES = {"A": "Caller", "B": "Receiver"}

# Why the generator rejected a call, in words. Unknown codes are shown as they are.
REASON_TEXT = {
    "too_short": "Too short",
    "judge_label_does_not_hold": "Judge: does not show the intended call",
    "judge_giveaway": "Judge: names the label outright",
    "no_scam_move": "No scam move found",
    "judge_saw_scam_in_normal_call": "Judge saw a scam in a normal call",
    "tone_mark_order": "Tone mark before a vowel",
    "number_without_slot": "Number written without a slot",
    "empty_line": "Empty line",
    "tts_failed": "Speech failed on a turn",
    "stray_brace": "A slot was left unfilled (stray { or })",
    "line_break": "A line had a line break or a speaker tag",
}


def db_path() -> Path:
    return Path(os.environ.get("CALLAS_DATASET_DB") or DEFAULT_DB)


@cache
def _store_at(path: str) -> Store:
    return Store(path)


def store() -> Store | None:
    """The store for the configured database file (one per path, shared across requests).

    None when the file does not exist: opening a Store would create it and its folders, and a
    page view must not do that.
    """
    path = db_path()
    return _store_at(str(path)) if path.is_file() else None


def reason_text(code: str) -> str:
    if code.startswith("unknown_slot:"):
        return f"Unknown slot {{{code.split(':', 1)[1]}}}"
    if code.startswith("llm_error:"):
        return f"Model error ({code.split(':', 1)[1]})"
    return REASON_TEXT.get(code, code)


def load(db: Store, record_id: str) -> dict | None:
    """The call record, or None if the id is malformed or unknown."""
    if not ID_RE.fullmatch(record_id):
        return None
    return db.load(record_id)


def audio_file(db: Store, record_id: str, name: str | None = None) -> Path | None:
    """The joined call (name None) or one turn's WAV, if it exists."""
    if not ID_RE.fullmatch(record_id):
        return None
    if name is None:
        path = db.folder / "audio" / f"{record_id}.wav"
    elif TURN_AUDIO_RE.fullmatch(name):
        path = db.folder / "audio" / record_id / name
    else:
        return None
    return path if path.is_file() else None


def save_review(db: Store, record_id: str, status: str, note: str | None) -> dict | None:
    """Set the review and return the updated record, or None if the call is unknown.

    note None means "leave the note as it is"; Undo sends no note so it cannot wipe one.
    """
    if status not in REVIEW_STATES:
        raise ValueError(f"review status must be one of {REVIEW_STATES}")
    if not ID_RE.fullmatch(record_id):
        return None
    if note is None:
        record = db.load(record_id)
        if record is None:
            return None
        note = (record.get("review") or {}).get("note", "")
    if not db.set_review(record_id, status, note.strip()[:MAX_NOTE_CHARS]):
        return None
    return db.load(record_id)


# ---- the list -------------------------------------------------------------------------------


@dataclass
class Row:
    id: str
    call_type: str
    type_name: str
    seed: str
    label_scam: bool
    status: str  # generator: accepted | rejected | in_progress
    review: str  # pending | approved | rejected
    turns: int
    seconds: float | None  # end of the last turn
    summary: str


def _row(rec: dict) -> Row:
    gen = rec.get("generation", {})
    call_type = rec.get("label_type") or "benign"
    turns = rec.get("speaker_turns") or gen.get("turns") or []
    end = turns[-1].get("end") if turns and isinstance(turns[-1], dict) else None
    return Row(
        id=rec["id"],
        call_type=call_type,
        type_name=CALL_TYPE_NAMES.get(call_type, call_type),
        seed=rec["id"].rsplit("_", 1)[-1],
        label_scam=bool(rec.get("label_scam")),
        status=gen.get("status", "in_progress"),
        review=rec.get("review", {}).get("status", "pending"),
        turns=len(turns),
        seconds=end,
        summary=rec.get("reference_summary", ""),
    )


def _summary_row(row) -> Row:
    """A list row from store.summaries(): no turns are loaded for the queue."""
    return Row(
        id=row["id"],
        call_type=row["call_type"],
        type_name=CALL_TYPE_NAMES.get(row["call_type"], row["call_type"]),
        seed=row["id"].rsplit("_", 1)[-1],
        label_scam=bool(row["label_scam"]),
        status=row["status"],
        review=row["review_status"],
        turns=row["turns"],
        seconds=row["seconds"],
        summary=row["reference_summary"],
    )


# The queue's views, in tab order: (label, which rows it shows).
VIEWS: dict[str, tuple[str, Any]] = {
    "todo": ("To review", lambda r: r.status == "accepted" and r.review == "pending"),
    "approved": ("Approved", lambda r: r.review == "approved"),
    "rejected": ("Rejected", lambda r: r.review == "rejected"),
    "failed": ("Failed checks", lambda r: r.status == "rejected" and r.review == "pending"),
    "all": ("All", lambda r: True),
}
DEFAULT_VIEW = "todo"


@dataclass
class Listing:
    rows: list[Row]  # the rows of the chosen type and view
    view: str
    counts: dict[str, int] = field(default_factory=dict)  # per view, for the chosen type
    reviewed: int = 0  # of the chosen type
    approved: int = 0
    rejected: int = 0
    total: int = 0  # of the chosen type


def empty_listing(view: str = DEFAULT_VIEW) -> Listing:
    """What the page shows when there is no database yet."""
    view = view if view in VIEWS else DEFAULT_VIEW
    return Listing(rows=[], view=view, counts=dict.fromkeys(VIEWS, 0))


def listing(db: Store, call_type: str = "", view: str = DEFAULT_VIEW) -> Listing:
    """Every call's summary row, kept to one call type (or all), with counts per view."""
    view = view if view in VIEWS else DEFAULT_VIEW
    rows = [_summary_row(r) for r in db.summaries()]
    if call_type:
        rows = [r for r in rows if r.call_type == call_type]
    return Listing(
        rows=[r for r in rows if VIEWS[view][1](r)],
        view=view,
        counts={k: sum(1 for r in rows if pick(r)) for k, (_, pick) in VIEWS.items()},
        reviewed=sum(r.review != "pending" for r in rows),
        approved=sum(r.review == "approved" for r in rows),
        rejected=sum(r.review == "rejected" for r in rows),
        total=len(rows),
    )


# ---- one call -------------------------------------------------------------------------------


@dataclass
class Turn:
    n: int  # 1-based, as on the /call page
    speaker: str
    who: str
    text: str
    entities: list[dict]
    start: float | None
    end: float | None
    audio: str | None  # URL
    reasoning: str
    flags: list[str]
    is_scam_start: bool


@dataclass
class Detail:
    record: dict
    row: Row
    turns: list[Turn]
    scam_start_n: int | None
    scam_start_from: str | None
    reasons: list[str]
    judge: dict | None
    call_audio: str | None
    setup: dict[str, Any]
    receiver: dict[str, Any]  # the receiver's profile, shown at the top of the chat
    plan: list[str]  # the scam caller's plan, if it had one


def _turn_flags(raw_turn: dict) -> list[str]:
    """Short warnings shown under a turn: failed checks and suspect spellings."""
    flags = [reason_text(p) for p in raw_turn.get("problems", [])]
    # Single characters are tokenizer splits, not typos; showing them is only noise.
    words = [w for w in raw_turn.get("spelling", []) if len(w.strip()) > 1]
    if words:
        flags.append(f"Check spelling: {', '.join(words[:3])}")
    return flags


def _build_turns(db: Store, record: dict) -> list[Turn]:
    gen = record.get("generation", {})
    rid = record["id"]
    scam_start = record.get("scam_start_turn")  # 0-based index, or None
    timings = record.get("speaker_turns") or []  # same order as the generated turns
    turns = []
    for i, raw in enumerate(gen.get("turns", [])):
        audio_name = Path(raw["audio"]).name if raw.get("audio") else None
        has_audio = audio_name and audio_file(db, rid, audio_name)
        timing = timings[i] if i < len(timings) else {}
        turns.append(
            Turn(
                n=i + 1,
                speaker=raw["speaker"],
                who=SPEAKER_NAMES.get(raw["speaker"], raw["speaker"]),
                text=raw.get("text", ""),
                entities=raw.get("entities", []),
                start=timing.get("start"),
                end=timing.get("end"),
                audio=f"/dataset/audio/{rid}/{audio_name}" if has_audio else None,
                reasoning=raw.get("reasoning", ""),
                flags=_turn_flags(raw),
                is_scam_start=scam_start == i,
            )
        )
    return turns


def _setup_table(gen: dict) -> dict[str, str]:
    """Label -> value for the "How it was made" list. Empty values are left out."""
    call = gen.get("call", {})
    scenario = call.get("scenario", {})
    table = {
        "Scenario": scenario.get("key", ""),
        "Caller plays": scenario.get("role", ""),
        "Caller wants": scenario.get("goal", ""),
        "How": scenario.get("tactics", ""),
        "Voices": " / ".join(gen.get("voices", {}).values()),
        "Models": ", ".join(sorted(set(gen.get("models", {}).values()))),
        "Config": f"{gen.get('config_name', '')} {gen.get('config_hash', '')}".strip(),
        "Seed": str(call.get("seed", "")),
    }
    return {label: value for label, value in table.items() if value}


def detail(db: Store, record: dict) -> Detail:
    gen = record.get("generation", {})
    rid = record["id"]
    scam_start = record.get("scam_start_turn")
    receiver = dict(gen.get("call", {}).get("receiver", {}))
    if "job" in receiver:  # calls made before receivers had full profiles
        receiver["occupation"] = receiver.pop("job")
    return Detail(
        record=record,
        row=_row(record),
        turns=_build_turns(db, record),
        scam_start_n=None if scam_start is None else scam_start + 1,
        scam_start_from=gen.get("scam_start_from"),
        reasons=[reason_text(r) for r in gen.get("reasons", [])],
        judge=gen.get("judge"),
        call_audio=f"/dataset/audio/{rid}" if audio_file(db, rid) else None,
        setup=_setup_table(gen),
        receiver=receiver,
        plan=(gen.get("briefing") or {}).get("plan", []),
    )
