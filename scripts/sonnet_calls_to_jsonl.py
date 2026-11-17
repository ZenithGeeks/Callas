"""Turn calls written by Claude Sonnet into dataset rows like dataset/*.jsonl.

Each raw file (dataset/sonnet_pilot/raw/*.json, one per batch and theme) holds calls whose turns
mark values inline as {{LABEL:value}}. The tags are removed and each value becomes an entity with
its character offsets in `text`. Baht amounts the writer left untagged are tagged MONEY, so the
notebook masks them like every other amount. The rows go to dataset/sonnet_<type>.jsonl, next to
the existing files and never over them.

    python scripts/sonnet_calls_to_jsonl.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "sonnet_pilot" / "raw"
LABELS = {
    "PERSON", "ORG", "PHONE", "BANK_ACCOUNT", "MONEY", "OTP", "URL", "LINE_ID", "CARD_NUMBER",
    "ID_CARD", "BIRTH_DATE", "ADDRESS",
}  # fmt: skip
TAG = re.compile(r"\{\{([A-Z_]+):([^{}]+?)\}\}")
TAG_OR_BAHT = re.compile(r"(\{\{[A-Z_]+:[^{}]+?\}\})|([\d๐-๙][\d๐-๙,]*\s?บาท)")
TURN = re.compile(r"^([AB]): (.*)$", re.DOTALL)
ID_PREFIX = {"normal": "benign"}


def tag_baht(body: str) -> str:
    """Tag the baht amounts outside existing tags as MONEY."""
    return TAG_OR_BAHT.sub(lambda m: m.group(1) or f"{{{{MONEY:{m.group(2)}}}}}", body)


def to_row(raw: dict, call_id: str) -> dict:
    """Build one dataset row from one raw call, checking turns, labels and offsets."""
    turns, entities, position = [], [], 0
    for index, line in enumerate(raw["turns"]):
        match = TURN.match(line.strip())
        assert match, f"{call_id} turn {index}: no 'A: ' or 'B: '"
        speaker, body = match.groups()
        assert speaker == "AB"[index % 2], f"{call_id} turn {index}: speakers do not alternate"
        words_start = position + len(speaker) + 2
        text, cursor = "", 0
        body = tag_baht(body)
        for tag in TAG.finditer(body):
            label, value = tag.group(1), tag.group(2).strip()
            assert label in LABELS, f"{call_id} turn {index}: unknown label {label}"
            text += body[cursor : tag.start()]
            begin = words_start + len(text)
            text += value
            entities.append(
                {"text": value, "label": label, "start": begin, "end": begin + len(value)}
            )
            cursor = tag.end()
        text += body[cursor:]
        assert "{{" not in text and "}}" not in text, f"{call_id} turn {index}: unclosed tag"
        turns.append({"speaker": speaker, "start": None, "end": None, "text": text})
        position = words_start + len(text) + 1

    full_text = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in turns)
    for entity in entities:
        assert full_text[entity["start"] : entity["end"]] == entity["text"], f"{call_id}: offset"
    is_scam = raw["label_type"] is not None
    start, leak = raw["scam_start_turn"], raw["first_leak_turn"]
    if is_scam:
        assert start is not None and turns[start]["speaker"] == "A", f"{call_id}: bad scam start"
    if leak is not None:
        assert turns[leak]["speaker"] == "B", f"{call_id}: first_leak_turn is not a receiver turn"
    return {
        "id": call_id,
        "channel": "call",
        "text": full_text,
        "audio_path": None,
        "speaker_turns": turns,
        "label_scam": is_scam,
        "label_type": raw["label_type"],
        "scenario": raw["scenario"],
        "scam_outcome": raw["scam_outcome"],
        "scam_outcome_source": "locked",
        "entities": entities,
        "reference_summary": raw["reference_summary"],
        "scam_start_turn": start if is_scam else None,
        "source": "synthetic",
        "receiver_leaked": raw["receiver_leaked"],
        "first_leak_turn": leak,
    }


def main() -> None:
    # Files sort pilot first (fake_authority.json before fake_authority_b2.json), so the ids of
    # earlier batches stay the same when a batch is added.
    calls_by_type: dict[str, list[dict]] = defaultdict(list)
    for raw_file in sorted(RAW.glob("*.json")):
        for raw in json.loads(raw_file.read_text(encoding="utf-8")):
            calls_by_type[raw["label_type"] or "normal"].append(raw)
    for call_type, calls in sorted(calls_by_type.items()):
        prefix = ID_PREFIX.get(call_type, call_type)
        # A scenario name used twice is fine: the notebook groups by it, so those calls share a split.
        rows = [to_row(raw, f"{prefix}_sonnet_{i:05d}") for i, raw in enumerate(calls)]
        out = ROOT / "dataset" / f"sonnet_{call_type}.jsonl"
        out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        entities = sum(len(row["entities"]) for row in rows)
        print(f"{out.relative_to(ROOT)}: {len(rows)} calls, {entities} entities")


if __name__ == "__main__":
    main()
