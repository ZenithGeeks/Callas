"""SQLite store for simulated calls: one file holds every call, its turns, the judge's verdict,
the human review and the scam pattern notes.

`save()` writes a whole call record (a dict in the spec's schema plus `generation` and `review`
blocks) in one transaction after every turn, and `load()` rebuilds it from the tables, so a
stopped run resumes from the turns table. The methods come in pairs: `load_call`, `save_call`
and `calls` give and take a records.CallRecord, which the simulator uses; `load`, `save` and
`call_dicts` give and take the same record as a dict, which the web pages and exports read.

WAL mode lets the generator write while the /dataset page reads. Each thread gets its own
connection (sqlite3 connections must not be shared across threads). `Store(path, read_only=True)`
reads a file without changing it, even one too old to have every column (see export.py).

Not supported: two generator runs writing the same database at once. Each call is saved turn by
turn and nothing locks a call, so two runs that reach the same call would write over each
other's turns. Run one generator per database file (use --db to split work).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .records import CallRecord

DEFAULT_DB = Path("data/synthetic/calls.db")  # relative to where you run from: the repo root
SCHEMA_VERSION = 1  # of the tables
RECORD_SCHEMA = 1  # of the call records built from them (generation.schema)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS calls (
    id               TEXT PRIMARY KEY,
    call_type        TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    label_scam       INTEGER NOT NULL,
    label_type       TEXT,
    status           TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress | accepted | rejected
    reasons          TEXT NOT NULL DEFAULT '[]',           -- JSON list of reason codes
    config_name      TEXT NOT NULL DEFAULT '',
    config_hash      TEXT NOT NULL DEFAULT '',
    models           TEXT NOT NULL DEFAULT '{}',           -- JSON: agent -> model name
    call             TEXT NOT NULL DEFAULT '{}',           -- JSON: the sampled CallConfig
    voices           TEXT NOT NULL DEFAULT '{}',           -- JSON: speaker -> TTS voice
    briefing         TEXT,                                 -- JSON: scam caller's plan, if any
    judge            TEXT,                                 -- JSON: the judge's answer
    scam_start_turn  INTEGER,
    scam_start_from  TEXT,                                 -- caller | judge
    receiver_leaked  TEXT NOT NULL DEFAULT '[]',           -- JSON: labels the receiver gave away
    first_leak_turn  INTEGER,                              -- index of the first leaking turn
    speech_style     TEXT NOT NULL DEFAULT '',             -- dash | pause: how numbers are said
    reference_summary TEXT NOT NULL DEFAULT '',
    audio_path       TEXT,
    review_status    TEXT NOT NULL DEFAULT 'pending',      -- pending | approved | rejected
    review_note      TEXT NOT NULL DEFAULT '',
    reviewed_at      TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS calls_by_type ON calls (call_type, status, review_status);

CREATE TABLE IF NOT EXISTS turns (
    call_id    TEXT NOT NULL REFERENCES calls (id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,             -- 0-based speaker turn
    speaker    TEXT NOT NULL,                -- A caller, B receiver
    agent      TEXT NOT NULL DEFAULT '',
    raw        TEXT NOT NULL DEFAULT '',     -- the line as written, with slots
    text       TEXT NOT NULL DEFAULT '',     -- slots filled
    spoken_text TEXT NOT NULL DEFAULT '',    -- what the voice was given (numbers spelled out)
    entities   TEXT NOT NULL DEFAULT '[]',   -- JSON spans within `text`
    asks_now   INTEGER NOT NULL DEFAULT 0,
    end_call   INTEGER NOT NULL DEFAULT 0,
    reasoning  TEXT NOT NULL DEFAULT '',
    problems   TEXT NOT NULL DEFAULT '[]',
    leaked     TEXT NOT NULL DEFAULT '[]',   -- JSON labels this receiver turn gave away
    spelling   TEXT NOT NULL DEFAULT '[]',
    latency_s  REAL,
    prompt_tokens INTEGER,                   -- estimated size of the prompt sent for this turn
    usage      TEXT NOT NULL DEFAULT '{}',   -- JSON real tokens from the server (usage.py)
    audio      TEXT,                         -- path relative to the store's folder
    audio_s    REAL,
    start_s    REAL,                         -- position in the joined call audio
    end_s      REAL,
    PRIMARY KEY (call_id, idx)
);

CREATE TABLE IF NOT EXISTS pattern_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scam_type  TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',     -- URL or a short name of where it came from
    published  TEXT,
    summary    TEXT NOT NULL,
    tactics    TEXT NOT NULL DEFAULT '[]',   -- JSON list of short tactic descriptions
    phrases    TEXT NOT NULL DEFAULT '[]',   -- JSON list of typical Thai phrases
    active     INTEGER NOT NULL DEFAULT 1,
    added_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_by_type ON pattern_notes (scam_type, active);
"""

# The values come from _call_values, in this order. On a conflict the review columns and
# created_at are left alone (see Store.save).
_UPSERT_CALL = """
INSERT INTO calls (
    id, call_type, seed, label_scam, label_type, status, reasons,
    config_name, config_hash, models, call, voices, briefing, judge,
    scam_start_turn, scam_start_from, receiver_leaked, first_leak_turn,
    speech_style,
    reference_summary, audio_path,
    review_status, review_note, reviewed_at, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?)
ON CONFLICT (id) DO UPDATE SET
    status = excluded.status, reasons = excluded.reasons,
    models = excluded.models, briefing = excluded.briefing,
    judge = excluded.judge, scam_start_turn = excluded.scam_start_turn,
    scam_start_from = excluded.scam_start_from,
    receiver_leaked = excluded.receiver_leaked,
    first_leak_turn = excluded.first_leak_turn,
    speech_style = excluded.speech_style,
    reference_summary = excluded.reference_summary,
    audio_path = excluded.audio_path, updated_at = excluded.updated_at
"""

_TURN_COLUMNS = (
    "speaker",
    "agent",
    "raw",
    "text",
    "spoken_text",
    "entities",
    "asks_now",
    "end_call",
    "reasoning",
    "problems",
    "leaked",
    "spelling",
    "latency_s",
    "prompt_tokens",
    "usage",
    "audio",
    "audio_s",
)
_JSON_TURN_FIELDS = ("entities", "problems", "leaked", "spelling", "usage")
# Columns added after the first release: (table, column, definition). An existing
# database gets them with ALTER TABLE when it is opened.
_ADDED_COLUMNS = (
    ("calls", "receiver_leaked", "TEXT NOT NULL DEFAULT '[]'"),
    ("calls", "first_leak_turn", "INTEGER"),
    ("calls", "speech_style", "TEXT NOT NULL DEFAULT ''"),
    ("turns", "spoken_text", "TEXT NOT NULL DEFAULT ''"),
    ("turns", "prompt_tokens", "INTEGER"),
    ("turns", "leaked", "TEXT NOT NULL DEFAULT '[]'"),
    ("turns", "usage", "TEXT NOT NULL DEFAULT '{}'"),
)
_BOOL_TURN_FIELDS = ("asks_now", "end_call")


def _add_missing_columns(db: sqlite3.Connection) -> None:
    """Add the columns an older database file does not have yet."""
    for table, column, definition in _ADDED_COLUMNS:
        have = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if column in have:
            continue
        try:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except sqlite3.OperationalError as exc:
            # another process opening the same file added it between our check and our ALTER
            if "duplicate column" not in str(exc).lower():
                raise


def _missing_columns(db: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """table -> {column: its default as SQL} for the added columns an older file does not have."""
    missing: dict[str, dict[str, str]] = {}
    for table, column, definition in _ADDED_COLUMNS:
        have = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if column not in have:
            default = definition.partition(" DEFAULT ")[2] or "NULL"
            missing.setdefault(table, {})[column] = default
    return missing


def now() -> str:
    """The current UTC time as an ISO string, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """All access to one calls.db. Safe to share across threads.

    A read-only store opens an existing file and never writes to it: it does not create the
    tables or add the columns an older file lacks. Loading gives those columns their default.
    """

    def __init__(self, path: Path | str = DEFAULT_DB, read_only: bool = False) -> None:
        self.path = Path(path)
        self.folder = self.path.parent  # audio paths in records are relative to this
        self.read_only = read_only
        self._local = threading.local()
        if read_only:
            self._missing = _missing_columns(self._conn())
            return
        self._missing = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._tx() as db:
            db.executescript(SCHEMA)
            _add_missing_columns(db)
            db.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema', ?)",
                (str(SCHEMA_VERSION),),
            )

    # connections

    def _conn(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self.read_only:
                uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
                conn = sqlite3.connect(uri, uri=True, timeout=30)
            else:
                conn = sqlite3.connect(self.path, timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _tx(self):
        """`with store._tx() as db:` commits on success and rolls back on error."""
        return self._conn()

    def close(self) -> None:
        """Close this thread's connection. Other threads keep theirs."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # calls

    def save(self, record: dict) -> None:
        """Write a whole call record (call row and all its turns) in one transaction.

        A new call gets the record's review block, but saving an existing call never changes
        its review: a generator that saves a stale record must not undo a human's decision.
        Use set_review to change it."""
        turns = record["generation"]["turns"]
        timed = record.get("speaker_turns") or []
        with self._tx() as db:
            db.execute(_UPSERT_CALL, _call_values(record, now()))
            for i, turn in enumerate(turns):
                timing = timed[i] if i < len(timed) else {}
                values = [_turn_value(turn, c) for c in _TURN_COLUMNS]
                db.execute(
                    f"""
                    INSERT OR REPLACE INTO turns (call_id, idx, {", ".join(_TURN_COLUMNS)},
                        start_s, end_s)
                    VALUES (?, ?, {", ".join("?" * len(_TURN_COLUMNS))}, ?, ?)
                    """,
                    (record["id"], i, *values, timing.get("start"), timing.get("end")),
                )
            # A rerun of the same call can only grow, but keep the table exact anyway.
            db.execute(
                "DELETE FROM turns WHERE call_id = ? AND idx >= ?", (record["id"], len(turns))
            )

    def load(self, call_id: str) -> dict | None:
        """The call record in the same shape simulate.new_record makes, or None."""
        db = self._conn()
        row = db.execute(
            f"SELECT {self._columns('calls')} FROM calls WHERE id = ?", (call_id,)
        ).fetchone()
        if row is None:
            return None
        turns = db.execute(
            f"SELECT {self._columns('turns')} FROM turns WHERE call_id = ? ORDER BY idx",
            (call_id,),
        ).fetchall()
        return _record_from_rows(row, turns)

    def has_column(self, table: str, column: str) -> bool:
        """Whether the file has this column. Only a read-only store can lack one."""
        return column not in self._missing.get(table, {})

    def _columns(self, table: str) -> str:
        """The SELECT list for a table: every column, with a stand-in for each missing one."""
        stand_ins = [
            f"{value} AS {column}" for column, value in self._missing.get(table, {}).items()
        ]
        return ", ".join(["*", *stand_ins])

    def load_call(self, call_id: str) -> CallRecord | None:
        """The call as a CallRecord, or None."""
        record = self.load(call_id)
        return None if record is None else CallRecord.from_dict(record)

    def save_call(self, record: CallRecord) -> None:
        """Write a CallRecord the way save() writes the record's dict."""
        self.save(record.to_dict())

    def calls(self, call_type: str = "") -> Iterator[CallRecord]:
        """Every call as a CallRecord, oldest id first, optionally of one call type."""
        for record in self.call_dicts(call_type):
            yield CallRecord.from_dict(record)

    def exists(self, call_id: str) -> bool:
        """Whether a call with this id is stored."""
        return (
            self._conn().execute("SELECT 1 FROM calls WHERE id = ?", (call_id,)).fetchone()
            is not None
        )

    def call_dicts(self, call_type: str = "") -> Iterator[dict]:
        """Every call as the dict load() gives, oldest id first, optionally of one call type."""
        db = self._conn()
        sql, args = "SELECT id FROM calls", ()
        if call_type:
            sql, args = sql + " WHERE call_type = ?", (call_type,)
        for (call_id,) in db.execute(sql + " ORDER BY id", args).fetchall():
            record = self.load(call_id)
            if record is not None:
                yield record

    def summaries(self) -> list[sqlite3.Row]:
        """One light row per call for list pages: no turns, no reasoning."""
        return (
            self._conn()
            .execute(
                """
            SELECT c.id, c.call_type, c.seed, c.label_scam, c.status, c.review_status,
                   c.reference_summary, COUNT(t.idx) AS turns, MAX(t.end_s) AS seconds
            FROM calls c LEFT JOIN turns t ON t.call_id = c.id
            GROUP BY c.id ORDER BY c.id
            """
            )
            .fetchall()
        )

    def set_review(self, call_id: str, status: str, note: str) -> bool:
        """Record a human review (pending, approved or rejected). False if there is no such call."""
        reviewed_at = None if status == "pending" else now()
        with self._tx() as db:
            cur = db.execute(
                """UPDATE calls SET review_status = ?, review_note = ?, reviewed_at = ?,
                   updated_at = ? WHERE id = ?""",
                (status, note, reviewed_at, now(), call_id),
            )
        return cur.rowcount == 1

    def reject_after_model_error(self, call_id: str, exc: BaseException) -> bool:
        """Mark an unfinished call `rejected` with the reason `llm_error:<error name>`.

        Used when a model error survived the retries. Returns False, and changes nothing, when
        there is no such call or it is already finished."""
        record = self.load_call(call_id)
        if record is None or record.generation.status != "in_progress":
            return False
        reasons = [f"llm_error:{type(exc).__name__}"]
        generation = replace(record.generation, status="rejected", reasons=reasons)
        self.save_call(replace(record, generation=generation))
        return True

    def delete(self, call_id: str) -> None:
        """Delete a call and, through the foreign key, its turns."""
        with self._tx() as db:
            db.execute("DELETE FROM calls WHERE id = ?", (call_id,))

    # pattern notes

    def add_note(
        self,
        scam_type: str,
        summary: str,
        tactics: list[str],
        phrases: list[str],
        source: str = "",
        published: str | None = None,
    ) -> int:
        """Store one pattern note and return its id."""
        with self._tx() as db:
            cur = db.execute(
                """INSERT INTO pattern_notes
                   (scam_type, source, published, summary, tactics, phrases, added_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    scam_type,
                    source,
                    published,
                    summary,
                    json.dumps(tactics, ensure_ascii=False),
                    json.dumps(phrases, ensure_ascii=False),
                    now(),
                ),
            )
        return int(cur.lastrowid)

    def notes(self, scam_type: str, limit: int = 8) -> list[dict]:
        """Active notes for one scam type: dated notes newest first, then undated ones (such as
        the wiki's seed examples), so real recent reports reach the planner first."""
        rows = (
            self._conn()
            .execute(
                """SELECT * FROM pattern_notes WHERE scam_type = ? AND active = 1
               ORDER BY published IS NULL, published DESC, id DESC LIMIT ?""",
                (scam_type, limit),
            )
            .fetchall()
        )
        return [
            {
                "id": r["id"],
                "source": r["source"],
                "published": r["published"],
                "summary": r["summary"],
                "tactics": json.loads(r["tactics"]),
                "phrases": json.loads(r["phrases"]),
            }
            for r in rows
        ]

    def replace_notes(self, scam_type: str, notes: list[dict]) -> None:
        """Make the stored notes of one scam type exactly `notes`, in order, in one transaction.

        Each note is a dict with summary and tactics, and optionally source, published, phrases
        and active (default True). Inactive notes are kept with active = 0.
        """
        stamp = now()
        with self._tx() as db:
            db.execute("DELETE FROM pattern_notes WHERE scam_type = ?", (scam_type,))
            for n in notes:
                db.execute(
                    """INSERT INTO pattern_notes
                       (scam_type, source, published, summary, tactics, phrases, active, added_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        scam_type,
                        n.get("source") or "",
                        n.get("published") or None,
                        n["summary"],
                        json.dumps(n["tactics"], ensure_ascii=False),
                        json.dumps(n.get("phrases", []), ensure_ascii=False),
                        int(n.get("active", True)),
                        stamp,
                    ),
                )

    def note_counts(self, scam_type: str) -> tuple[int, int]:
        """(all notes, active notes) stored for one scam type."""
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*), COALESCE(SUM(active), 0) FROM pattern_notes WHERE scam_type = ?",
                (scam_type,),
            )
            .fetchone()
        )
        return int(row[0]), int(row[1])

    def set_note_active(self, note_id: int, active: bool) -> None:
        """Turn a note on or off for the planner. The note stays in the store."""
        with self._tx() as db:
            db.execute("UPDATE pattern_notes SET active = ? WHERE id = ?", (int(active), note_id))


# row <-> record


def _call_values(record: dict, stamp: str) -> tuple:
    """The values of the calls row for this record, in the order of _UPSERT_CALL."""
    generation, review = record["generation"], record.get("review", {})
    return (
        record["id"],
        record.get("label_type") or "benign",
        generation["call"]["seed"],
        int(bool(record["label_scam"])),
        record.get("label_type"),
        generation["status"],
        json.dumps(generation.get("reasons", []), ensure_ascii=False),
        generation.get("config_name", ""),
        generation.get("config_hash", ""),
        json.dumps(generation.get("models", {}), ensure_ascii=False),
        json.dumps(generation["call"], ensure_ascii=False),
        json.dumps(generation.get("voices", {}), ensure_ascii=False),
        _dump_or_none(generation.get("briefing")),
        _dump_or_none(generation.get("judge")),
        record.get("scam_start_turn"),
        generation.get("scam_start_from"),
        json.dumps(generation.get("receiver_leaked", []), ensure_ascii=False),
        generation.get("first_leak_turn"),
        generation.get("speech_style", ""),
        record.get("reference_summary", ""),
        record.get("audio_path"),
        review.get("status", "pending"),
        review.get("note", ""),
        review.get("reviewed_at"),
        generation.get("created_at") or stamp,
        stamp,
    )


def _dump_or_none(value: Any) -> str | None:
    """JSON text for a column that may be NULL."""
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _turn_value(turn: dict, column: str) -> Any:
    """The turn's value for one turns column, in the form the column stores."""
    value = turn.get(column)
    if column in _JSON_TURN_FIELDS:
        empty: Any = {} if column == "usage" else []
        return json.dumps(value or empty, ensure_ascii=False)
    if column in _BOOL_TURN_FIELDS:
        return int(bool(value))
    if column in ("agent", "raw", "text", "spoken_text", "reasoning"):
        return value or ""
    return value


def _turn_from_row(row: sqlite3.Row) -> dict:
    """A turns row as the turn dict of a record."""
    turn = {c: row[c] for c in _TURN_COLUMNS}
    for c in _JSON_TURN_FIELDS:
        turn[c] = json.loads(turn[c])
    for c in _BOOL_TURN_FIELDS:
        turn[c] = bool(turn[c])
    return turn


def _dialogue(turns: list[sqlite3.Row]) -> tuple[str, list[dict]]:
    """The spec's `text` ("A: ...\\nB: ...") and entity spans shifted into it."""
    parts, spans, offset = [], [], 0
    for t in turns:
        prefix = f"{t['speaker']}: "
        line = prefix + t["text"]
        for e in json.loads(t["entities"]):
            start = offset + len(prefix) + e["start"]
            spans.append({**e, "start": start, "end": start + len(e["text"])})
        parts.append(line)
        offset += len(line) + 1
    return "\n".join(parts), spans


def _record_from_rows(row: sqlite3.Row, turns: list[sqlite3.Row]) -> dict:
    """A calls row and its turns rows as the call record the web pages and the export read."""
    text, entities = _dialogue(turns)
    return {
        "id": row["id"],
        "channel": "call",
        "text": text,
        "audio_path": row["audio_path"],
        "speaker_turns": [
            {"speaker": t["speaker"], "start": t["start_s"], "end": t["end_s"], "text": t["text"]}
            for t in turns
        ],
        "label_scam": bool(row["label_scam"]),
        "label_type": row["label_type"],
        "entities": entities,
        "reference_summary": row["reference_summary"],
        "scam_start_turn": row["scam_start_turn"],
        "source": "synthetic",
        "generation": {
            "schema": RECORD_SCHEMA,
            "status": row["status"],
            "reasons": json.loads(row["reasons"]),
            "config_name": row["config_name"],
            "config_hash": row["config_hash"],
            "models": json.loads(row["models"]),
            "created_at": row["created_at"],
            "call": json.loads(row["call"]),
            "voices": json.loads(row["voices"]),
            "briefing": json.loads(row["briefing"]) if row["briefing"] else None,
            "turns": [_turn_from_row(t) for t in turns],
            "receiver_leaked": json.loads(row["receiver_leaked"]),
            "first_leak_turn": row["first_leak_turn"],
            "speech_style": row["speech_style"],
            "scam_start_from": row["scam_start_from"],
            "judge": json.loads(row["judge"]) if row["judge"] else None,
        },
        "review": {
            "status": row["review_status"],
            "note": row["review_note"],
            "reviewed_at": row["reviewed_at"],
        },
    }


def import_json_folder(store: Store, folder: Path) -> tuple[int, int]:
    """Load the per-call JSON files the simulator wrote before the store existed.

    Returns (imported, skipped). A call already in the store is skipped, not overwritten.
    """
    imported = skipped = 0
    for path in sorted(folder.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "generation" not in record or store.exists(record["id"]):
            skipped += 1
            continue
        store.save(record)
        imported += 1
    return imported, skipped
