"""Calls from the scam classifier's test split, for a person to read aloud on the live page.

The test split is the call ids in data/processed_balanced/test.csv, written by step 7 of
notebooks/scam_classifier_phayathaibert.ipynb. The calls are in dataset/*.jsonl, the files of
the Zenithgeeks/antifakenews_scam dataset. Train and valid calls are never offered: the
fine-tuned classifier learned from them, so a live demo on them would overstate it.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .dataset import GENERATED_CALLER

ROOT = Path(__file__).resolve().parents[1]
TEST_SPLIT = ROOT / "data" / "processed_balanced" / "test.csv"
DATASET_DIR = ROOT / "dataset"


@dataclass(frozen=True)
class TestCall:
    """One test-split call: what the caller says, and the label the classifier should give."""

    __test__ = False  # not a pytest test class, despite the name

    call_id: str
    label: str  # the scam type, or "normal"
    caller_lines: tuple[str, ...]


def random_test_call() -> TestCall | None:
    """A random call from the test split, or None when test.csv or its calls are missing."""
    if not TEST_SPLIT.exists():
        return None
    jsonl = tuple(sorted(DATASET_DIR.glob("*.jsonl")))
    mtimes = tuple(path.stat().st_mtime_ns for path in (TEST_SPLIT, *jsonl))
    calls = _test_calls(jsonl, mtimes)
    return random.choice(calls) if calls else None


@lru_cache(maxsize=1)
def _test_calls(jsonl: tuple[Path, ...], mtimes: tuple[int, ...]) -> tuple[TestCall, ...]:
    """Every test-split call in the JSONL files. `mtimes` is in the key so a new export reloads."""
    with TEST_SPLIT.open(encoding="utf-8-sig", newline="") as split:
        test_ids = {row["call_id"] for row in csv.DictReader(split)}
    calls = []
    for path in jsonl:
        for line in path.read_text(encoding="utf-8").splitlines():
            call = json.loads(line) if line.strip() else None
            if call and call["id"] in test_ids:
                calls.append(_test_call(call))
    return tuple(calls)


def _test_call(call: dict) -> TestCall:
    """One call of a JSONL file as a TestCall."""
    return TestCall(
        call_id=call["id"],
        label=(call.get("label_type") or "scam") if call["label_scam"] else "normal",
        caller_lines=tuple(
            t["text"] for t in call["speaker_turns"] if t["speaker"] == GENERATED_CALLER
        ),
    )
