"""Write the approved simulated calls as the csv that notebooks/scam_classifier_phayathaibert.ipynb
reads (issue #54).

    python -m data_pipeline.synthesize --export-notebook out.csv
    python -m data_pipeline.synthesize --export-notebook out.csv --store data/synthetic/calls.db
    python -m data_pipeline.synthesize --export-notebook out.csv --text asr

The notebook's raw format is one row per call with `dialogue` (turns tagged `ผู้โทร:` and
`ผู้รับ:`), `scam_class` ("Scam" / "Not a Scam") and `scam_type`. This file adds `source`,
`group_hint` (call type and scenario, so the notebook keeps one scenario in one split group),
`scam_start_turn`, `call_id` and, when the store has them, the leak columns.

Only calls the simulator accepted and a person approved on the /dataset page are written. A call
whose text the notebook's cleaning would change or cut is skipped and counted under a reason,
because the model would then train on something other than what was reviewed.

The store is opened read-only, so this never changes calls.db.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from functools import partial
from pathlib import Path

from nlp_core.text import normalize_text
from nlp_core.types import SCAM_TYPES

ENCODING = "utf-8-sig"  # what the notebook reads and writes: Excel shows the Thai text correctly
SOURCE = "simulated"
SCAM, NOT_SCAM = "Scam", "Not a Scam"

# The store's speakers -> the tags the hand-written dataset uses.
TAGS = {"A": "ผู้โทร", "B": "ผู้รับ"}
ROLES = {"ผู้โทร": "caller", "ผู้รับ": "receiver"}

# Columns an older calls.db may not have yet (the store adds them when it opens the file). They
# are exported when the file has them.
LEAK_COLUMNS = ("receiver_leaked", "first_leak_turn")
CSV_FIELDS = (
    "dialogue",
    "scam_class",
    "scam_type",
    "source",
    "group_hint",
    "scam_start_turn",
    "call_id",
)

# These copy the cleaning rules of the notebook (cells 3 and 9). A test reads the notebook and
# fails if any of them differs, so a change there cannot go unnoticed here.
_TAG_WORDS = "|".join(re.escape(tag) for tag in ROLES)
SPEAKER_TAG = re.compile(rf"({_TAG_WORDS})\s*(?:\([^)]*\))?\s*:\s*")
BRACKET_NOTE = re.compile(r"\[[^\]]*\]")
STAGE_NOTE = re.compile(r"\((?:เสียง|คนอื่น|ชาย|หญิง)[^)]*\)")
CALLER_TAG = TAGS["A"]
NARRATION = re.compile(rf"{re.escape(CALLER_TAG)}(?!\s*[:(]).*?(?=\s+(?:รบกวน|ถ้าไม่ทำตาม)|$)")
LABEL_NAMES = re.compile(r"Scam|Phishing|Impersonation")

# Why a call is not written, in the order the checks run.
NOT_ACCEPTED = "not_accepted"
REVIEW_PENDING = "review_pending"
REVIEW_REJECTED = "review_rejected"
UNKNOWN_TYPE = "unknown_scam_type"
TOO_FEW_TURNS = "fewer_than_2_turns"
UNKNOWN_SPEAKER = "unknown_speaker"
EMPTY_TURN = "empty_turn"
SPEAKER_TAG_IN_TURN = "speaker_tag_in_turn"
BRACKET_IN_TURN = "bracket_in_turn"
STAGE_NOTE_IN_TURN = "stage_note_in_turn"
CALLER_NOUN = "caller_word_in_caller_turn"
LABEL_NAME = "english_label_name"
CLEANING_CHANGES_CALL = "cleaning_changes_call"
ASR_INCOMPLETE = "asr_incomplete"

# The names the notebook's TYPE_MAP maps (its step 4) for the five spec types.
TYPE_NAMES = {t: t for t in SCAM_TYPES}


class Refused(Exception):
    """The call cannot be written; `reason` is the code the summary counts it under."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def turn_problem(speaker: str, text: str) -> str | None:
    """The first reason the notebook's cleaning would damage this turn, or None.

    The checks look at the turn as written and again after the notebook's own normalization,
    because normalizing can join pieces into a pattern that was not there before."""
    for seen in (text, normalize_text(text)):
        if SPEAKER_TAG.search(seen):
            return SPEAKER_TAG_IN_TURN  # the call would split into extra turns
        if BRACKET_NOTE.search(seen) or "[" in seen or "]" in seen:
            return BRACKET_IN_TURN  # cut as a hang-up note, or left behind and the call dropped
        if STAGE_NOTE.search(seen):
            return STAGE_NOTE_IN_TURN
        if LABEL_NAMES.search(seen):
            return LABEL_NAME  # the notebook drops a call whose text names a label
        if speaker == "A" and NARRATION.search(seen):
            return CALLER_NOUN  # the rest of the caller's turn is cut as narration
    return None


def render_dialogue(turns: list[tuple[str, str]]) -> str:
    """[(speaker, text), ...] as one line, the layout most of the hand-written file uses."""
    return " ".join(f"{TAGS[s]}: {' '.join(t.split())}" for s, t in turns)


def parse_dialogue(dialogue: str) -> list[tuple[str, str]]:
    """The notebook's parse_dialogue, giving [(role, raw text), ...]."""
    parts = SPEAKER_TAG.split(dialogue)
    return [(ROLES[tag], text) for tag, text in zip(parts[1::2], parts[2::2], strict=True)]


def notebook_cleaning(dialogue: str) -> list[tuple[str, str]]:
    """The notebook's clean_turns: [(role, text), ...] after notes and narration are removed."""
    turns: list[tuple[str, str]] = []
    for role, text in parse_dialogue(dialogue):
        text = BRACKET_NOTE.sub(" ", text)
        text = STAGE_NOTE.sub(" ", text)
        text = normalize_text(text)
        if role == "caller":
            text = normalize_text(NARRATION.sub(" ", text))
        if not text:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, f"{turns[-1][1]} {text}")
        else:
            turns.append((role, text))
    return turns


def build_row(call: dict, turns: list[dict], text_of: Callable[[dict], str]) -> dict:
    """One csv row for an approved call, or raise Refused.

    `call` is the calls row (as a dict), `turns` its turns in order, and `text_of(turn)` the
    text to write for a turn (the script, or its ASR transcript)."""
    if call["label_scam"] and call["label_type"] not in TYPE_NAMES:
        raise Refused(UNKNOWN_TYPE, str(call["label_type"]))
    pairs: list[tuple[str, str]] = []
    for turn in turns:
        if turn["speaker"] not in TAGS:
            raise Refused(UNKNOWN_SPEAKER, f"{call['id']} turn {turn['idx']}")
        text = text_of(turn)
        if not normalize_text(text):
            raise Refused(EMPTY_TURN, f"{call['id']} turn {turn['idx']}")
        problem = turn_problem(turn["speaker"], text)
        if problem:
            raise Refused(problem, f"{call['id']} turn {turn['idx']}")
        pairs.append((turn["speaker"], text))
    if len(pairs) < 2:
        raise Refused(TOO_FEW_TURNS, call["id"])
    dialogue = render_dialogue(pairs)
    # Last guard: run the notebook's cleaning and compare. Anything the checks above did not
    # name (two turns by one speaker, a turn the normalization empties) shows up here.
    expected = [(ROLES[TAGS[s]], normalize_text(t)) for s, t in pairs]
    if notebook_cleaning(dialogue) != expected:
        raise Refused(CLEANING_CHANGES_CALL, call["id"])
    scenario = json.loads(call["call"]).get("scenario", {}).get("key")
    return {
        "dialogue": dialogue,
        "scam_class": SCAM if call["label_scam"] else NOT_SCAM,
        "scam_type": TYPE_NAMES[call["label_type"]] if call["label_scam"] else "",
        "source": SOURCE,
        "group_hint": f"{call['call_type']}/{scenario}" if scenario else call["id"],
        "scam_start_turn": call["scam_start_turn"] if call["label_scam"] else None,
        "call_id": call["id"],
    }


# ASR text


def default_asr_cache() -> Path:
    """Where the export keeps its transcripts: eval/results/asr_cache (or CALLAS_ASR_CACHE)."""
    env = os.environ.get("CALLAS_ASR_CACHE")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "eval/results/asr_cache"


def turn_signature(turn: dict) -> str:
    """Identifies the audio a transcript belongs to: the turn's script text and audio file name.
    A call regenerated under the same id has other text, so its old transcripts no longer match."""
    return hashlib.sha1(f"{turn['text']}\n{turn.get('audio') or ''}".encode()).hexdigest()[:16]


class AsrText:
    """ASR transcript of a turn's audio.

    The cache is one json file per model, `<model>.export.json`, with {"<call id>/<turn idx>":
    {"sig": turn_signature, "text": transcript}}. It is not the file scripts/eval_autocorrect.py
    writes, because that one cannot tell that a call was regenerated. An entry whose signature
    differs is a miss. A miss is transcribed when pythaiasr is installed and the audio file
    exists, and then cached. If that is not possible the turn has no ASR text and the call is
    refused (asr_incomplete): mixing written and spoken text in one call would not match what
    the classifier reads in production."""

    def __init__(self, folder: Path, cache_dir: Path, model: str = "typhoon_asr") -> None:
        self.folder, self.model = folder, model
        self.cache_path = cache_dir / f"{model}.export.json"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.transcribed = 0
        self._asr = None
        self._unavailable = False

    def __call__(self, call_id: str, turn: dict) -> str:
        key, sig = f"{call_id}/{turn['idx']}", turn_signature(turn)
        hit = self.cache.get(key)
        if hit and hit["sig"] == sig:
            return hit["text"]
        text = self._transcribe(turn.get("audio"))
        if text is None:
            raise Refused(ASR_INCOMPLETE, f"{call_id} turn {turn['idx']}")
        self.cache[key] = {"sig": sig, "text": text}
        self.transcribed += 1
        return text

    def _transcribe(self, audio: str | None) -> str | None:
        if not audio or self._unavailable or not (self.folder / audio).is_file():
            return None
        if self._asr is None:
            try:
                from pythaiasr import ASR

                self._asr = ASR(model=self.model, device="auto")
            except ImportError:
                self._unavailable = True
                return None
        return str(self._asr(str(self.folder / audio)))

    def save(self) -> None:
        """Write the cache file, only when this run transcribed something new."""
        if self.transcribed:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")


# export


def _connect(db: Path) -> sqlite3.Connection:
    """A read-only connection, so the export can never change calls.db."""
    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def export_notebook(
    db: Path,
    out: Path,
    text: str = "script",
    asr: Callable[[str, dict], str] | None = None,
) -> tuple[int, Counter]:
    """Write the csv. Returns (rows written, calls skipped by reason).

    `asr(call_id, turn)` gives the ASR text of a turn for text="asr"."""
    columns, calls, turns = _read_store(db)
    leak = [c for c in LEAK_COLUMNS if c in columns]
    rows: list[dict] = []
    skipped: Counter = Counter()
    for call in calls:
        reason = _not_ready(call)
        if reason:
            skipped[reason] += 1
            continue
        text_of = partial(_turn_text, call_id=call["id"], text=text, asr=asr)
        try:
            row = build_row(call, turns.get(call["id"], []), text_of)
        except Refused as exc:
            skipped[exc.reason] += 1
            continue
        for name in leak:
            row[name] = call[name]
        rows.append(row)
    _write_csv(out, [*CSV_FIELDS, *leak], rows)
    return len(rows), skipped


def _read_store(db: Path) -> tuple[list[str], list[dict], dict[str, list[dict]]]:
    """(column names of calls, every call row, the turn rows of each call id)."""
    con = _connect(db)
    try:
        columns = [r["name"] for r in con.execute("PRAGMA table_info(calls)")]
        calls = [dict(r) for r in con.execute("SELECT * FROM calls ORDER BY id")]
        turns: dict[str, list[dict]] = {}
        for r in con.execute("SELECT * FROM turns ORDER BY call_id, idx"):
            turns.setdefault(r["call_id"], []).append(dict(r))
    finally:
        con.close()
    return columns, calls, turns


def _not_ready(call: dict) -> str | None:
    """Why the call is not written before its turns are even looked at, or None if it is ready."""
    if call["status"] != "accepted":
        return NOT_ACCEPTED
    if call["review_status"] == "approved":
        return None
    return REVIEW_REJECTED if call["review_status"] == "rejected" else REVIEW_PENDING


def _turn_text(
    turn: dict, *, call_id: str, text: str, asr: Callable[[str, dict], str] | None
) -> str:
    """The text to write for a turn: its ASR transcript for text="asr", else its script."""
    if text == "asr" and asr is not None:
        return asr(call_id, turn)
    return turn["text"]


def _write_csv(out: Path, fields: list[str], rows: list[dict]) -> None:
    """Write the rows with a header line. A None value becomes an empty cell."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if v is None else v for k, v in row.items()})


def run_export(out: Path, store: Path, text: str = "script") -> int:
    """The --export-notebook command. Prints a summary and returns the exit code."""
    if store.is_dir():
        store = store / "calls.db"
    if not store.is_file():
        print(f"no store at {store}; pass --store <calls.db>")
        return 2
    asr = AsrText(store.parent, default_asr_cache()) if text == "asr" else None
    written, skipped = export_notebook(store, out, text, asr)
    if asr is not None:
        asr.save()
        print(f"ASR text from {asr.cache_path} ({asr.transcribed} turns transcribed now)")
    print(f"wrote {written} rows to {out} (text: {text})")
    if skipped:
        print(f"skipped {sum(skipped.values())} calls:")
        for reason, n in skipped.most_common():
            print(f"  {reason:28s} {n}")
    if not written:
        print("nothing written: approve calls on the /dataset page first")
    return 0
