"""How a romance scam caller talks: docs/scam-wiki/romance_voice.md.

The page has `## name` sections. A romance caller gets four of them as its `voice` (instead of
agents.yaml speakers.voice): `all`, the section named by its scenario's `relation` (lover,
friend, family, ...), `threat` when the scenario has `threatens: true`, and `endings`. Text
before the first section and <!-- comments --> are for people and never reach the model.

A section may hold a table whose first two columns are `Caller | Receiver` (genders, or
`any`). The caller then sees only one row of it, picked by the call's seed among the rows that
fit both genders, so it has no other words to mix in.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Scenario, ScenariosFile

PATH = Path(__file__).resolve().parents[2] / "docs" / "scam-wiki" / "romance_voice.md"
ALWAYS = ("all", "endings")  # sections every romance caller gets, around its relation's
THREAT = "threat"  # the section only a scenario with `threatens: true` gets

_HEADING = re.compile(r"^##\s+([a-z_]+)")
_COMMENT = re.compile(r"<!--.*?-->\s*", re.DOTALL)
_TABLE_HEADER = "| Caller | Receiver |"


def load() -> dict[str, str]:
    """The page's sections, name -> text with its heading. Raises ValueError if the page is
    missing or empty."""
    if not PATH.exists():
        raise ValueError(f"missing {PATH}")
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in _COMMENT.sub("", PATH.read_text(encoding="utf-8")).splitlines():
        if heading := _HEADING.match(line):
            current = sections.setdefault(heading.group(1), [])
        if current is not None:
            current.append(line)
    if not sections:
        raise ValueError(f"{PATH} has no `## name` sections")
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def check(sections: dict[str, str], scenarios: ScenariosFile) -> None:
    """Raise ValueError unless every romance scenario has a relation with a section, the
    sections it needs exist, and every Caller | Receiver table has a row for each gender pair."""
    romance = scenarios.scams.get("romance", [])
    needed = list(ALWAYS) + [THREAT] * any(s.threatens for s in romance)
    if missing := [n for n in needed if n not in sections]:
        raise ValueError(f"{PATH.name} has no section {missing}")
    relations = sorted(set(sections) - set(ALWAYS) - {THREAT})
    for scenario in romance:
        if scenario.relation not in relations:
            raise ValueError(
                f"scenarios.yaml romance scenario {scenario.key} needs `relation:` set to one of "
                f"{relations} (the sections of {PATH.name}), got {scenario.relation!r}"
            )
    genders = scenarios.receivers.genders
    for name, text in sections.items():
        if (table := _table(text)) is None:
            continue
        for caller in genders:
            for receiver in genders:
                if not _fitting(table[1], caller, receiver):
                    raise ValueError(
                        f"{PATH.name} section {name}: no table row for a {caller} caller and a "
                        f"{receiver} receiver"
                    )


def voice(
    sections: dict[str, str],
    gender: str,
    receiver_gender: str,
    scenario: Scenario,
    seed_key: str,
) -> str:
    """The caller's voice: its gender, then the all, relation, threat (if any) and endings
    sections, each table cut to one row."""
    first, last = ALWAYS
    names = [first, scenario.relation] + [THREAT] * scenario.threatens + [last]
    rng = random.Random(f"voice-row:{seed_key}")
    parts = [_one_row(sections[n], gender, receiver_gender, rng) for n in names]
    return "\n\n".join([f"You are {gender}.", *parts])


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table(text: str) -> tuple[int, list[str]] | None:
    """(index of the header line, data rows) of the Caller | Receiver table in `text`, if any."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(_TABLE_HEADER):
            rows = []
            for row in lines[i + 2 :]:  # after the header and its |---| line
                if not row.startswith("|"):
                    break
                rows.append(row)
            return i, rows
    return None


def _fitting(rows: list[str], gender: str, receiver_gender: str) -> list[str]:
    def fits(cell: str, want: str) -> bool:
        return cell in (want, "any")

    return [r for r in rows if fits(_cells(r)[0], gender) and fits(_cells(r)[1], receiver_gender)]


def _one_row(section: str, gender: str, receiver_gender: str, rng: random.Random) -> str:
    """The section with its Caller | Receiver table cut to one fitting row, or as it is."""
    table = _table(section)
    if table is None:
        return section
    start, rows = table
    lines = section.splitlines()
    chosen = rng.choice(_fitting(rows, gender, receiver_gender))
    # Spell the row's two words out, so the caller never reads another word it could use for
    # itself (a ผม caller that sees หนู anywhere in its prompt drifts into it).
    row = dict(zip(_cells(lines[start]), _cells(chosen)))
    words = (
        f"In every line you call yourself {row['You call yourself']} and the receiver "
        f"{row['You call the receiver']}, and nothing else."
    )
    return "\n".join(lines[: start + 2] + [chosen, "", words] + lines[start + 2 + len(rows) :])
