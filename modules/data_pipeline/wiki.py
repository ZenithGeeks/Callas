"""The scam wiki: one Markdown page per scam type, the source of truth for pattern notes.

Pages live in docs/scam-wiki/<scam_type>.md so people can read, edit and review them in
GitHub. `sync()` copies them into the SQLite store, which is what the planner reads.

Page format (see docs/scam-wiki/README.md):

    # Phishing / OTP

    Free text here is ignored.

    ## One-line summary of the pattern
    - source: URL or name (may be empty)
    - published: 2026-01-31 (may be empty)
    - active: yes

    Tactics:
    - first tactic

    Phrases:
    - Thai phrase with {SLOTS}
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

from nlp_core.types import SCAM_TYPES

from .store import Store

DEFAULT_WIKI = Path("docs/scam-wiki")  # relative to the repo root, like the store's DEFAULT_DB

TITLES = {
    "fake_authority": "Fake authority (police, DSI, revenue department)",
    "romance": "Romance",
    "investment": "Investment",
    "phishing_otp": "Phishing / OTP",
    "lottery": "Lottery / prize",
}

_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_KEY = re.compile(r"^(source|published|active)\s*:\s*(.*)$", re.IGNORECASE)
_SECTION = re.compile(r"^(tactics|phrases)\s*:\s*$", re.IGNORECASE)
_ACTIVE_LINE = re.compile(r"^(\s*[-*]\s*active\s*:).*$", re.IGNORECASE)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ENCODING = "utf-8-sig"  # pages saved by Windows editors may start with a BOM
_YES = {"yes", "true", "1", "y"}
_NO = {"no", "false", "0", "n"}


class WikiError(ValueError):
    """A page cannot be read. The message names the file and the line."""


def _fail(name: str, line: int, message: str) -> WikiError:
    """A WikiError that starts with `file:line`, the way compilers write it."""
    return WikiError(f"{name}:{line}: {message}")


def parse_page(text: str, name: str = "<page>") -> list[dict]:
    """Notes of one page, in page order.

    Each note has summary, source, published, active, tactics and phrases.
    """
    notes: list[dict] = []
    note: dict | None = None
    heading_line = 0  # line of the current note's heading
    section = "meta"  # meta | tactics | phrases

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "##":
            raise _fail(name, number, "note has no summary after '##'")
        if line.startswith("## "):
            _keep(notes, note, heading_line, name)
            note, heading_line, section = _new_note(line[3:].strip()), number, "meta"
        elif note is None or not line:
            continue  # title, intro and blank lines
        elif header := _SECTION.match(line):
            section = header.group(1).lower()
        else:
            _read_bullet(note, section, raw, number, name)
    _keep(notes, note, heading_line, name)
    return notes


def _new_note(summary: str) -> dict:
    """An empty note under this heading."""
    return {
        "summary": summary,
        "source": "",
        "published": None,
        "active": True,
        "tactics": [],
        "phrases": [],
    }


def _keep(notes: list[dict], note: dict | None, heading_line: int, name: str) -> None:
    """Add a finished note to `notes`. A note needs at least one tactic."""
    if note is None:
        return
    if not note["tactics"]:
        raise _fail(name, heading_line, f"note '{note['summary']}' has no tactics")
    notes.append(note)


def _read_bullet(note: dict, section: str, raw: str, number: int, name: str) -> None:
    """Put one bullet line in the note: a tactic, a phrase or one of source, published, active."""
    bullet = _BULLET.match(raw)
    if not bullet:
        raise _fail(name, number, f"expected a '- ' bullet or 'Tactics:'/'Phrases:': {raw.strip()}")
    item = bullet.group(1).strip()
    if section in ("tactics", "phrases"):
        note[section].append(item)
        return
    key = _KEY.match(item)
    if not key:
        raise _fail(name, number, f"unknown line before 'Tactics:': {item}")
    field, value = key.group(1).lower(), key.group(2).strip()
    if field == "source":
        note["source"] = value
    elif field == "published":
        if value and not _is_date(value):
            raise _fail(name, number, f"published must be YYYY-MM-DD or empty, got: {value}")
        note["published"] = value or None
    else:
        word = value.lower()
        if word not in _YES | _NO:
            raise _fail(name, number, f"active must be yes or no, got: {value}")
        note["active"] = word in _YES


def _is_date(value: str) -> bool:
    """A real calendar date written YYYY-MM-DD (2026-02-30 is not)."""
    if not _DATE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _one_line(value: str) -> str:
    """Collapse line breaks and repeated spaces, so a value stays on its page line."""
    return " ".join(value.split())


def render_page(title: str, notes: list[dict], intro: str = "") -> str:
    """A page as text. parse_page(render_page(...)) gives the notes back."""
    out = [f"# {_one_line(title)}", ""]
    if intro.strip():
        out += [intro.strip(), ""]
    out += [_render_note(n) for n in notes]
    return "\n".join(out).rstrip("\n") + "\n"


def _render_note(note: dict) -> str:
    """One note as the page text that parse_page reads."""
    lines = [
        f"## {_one_line(note['summary'])}",
        f"- source: {_one_line(note.get('source') or '')}".rstrip(),
        f"- published: {note.get('published') or ''}".rstrip(),
        f"- active: {'yes' if note.get('active', True) else 'no'}",
        "",
        "Tactics:",
        *[f"- {_one_line(t)}" for t in note["tactics"]],
        "",
    ]
    if note.get("phrases"):
        lines += ["Phrases:", *[f"- {_one_line(p)}" for p in note["phrases"]], ""]
    return "\n".join(lines)


def page_path(folder: Path | str, scam_type: str) -> Path:
    """The file of one scam type's page."""
    if scam_type not in SCAM_TYPES:
        raise ValueError(f"unknown scam type: {scam_type}")
    return Path(folder) / f"{scam_type}.md"


def _write(path: Path, text: str) -> None:
    """Write a page with LF line endings, so a Windows run does not rewrite every line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_wiki(folder: Path | str = DEFAULT_WIKI) -> dict[str, list[dict]]:
    """Notes per scam type. A type with no page has an empty list."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"wiki folder not found: {folder}")
    wiki = {}
    for scam_type in SCAM_TYPES:
        path = page_path(folder, scam_type)
        text = path.read_text(encoding=ENCODING) if path.exists() else ""
        wiki[scam_type] = parse_page(text, path.as_posix())
    return wiki


def add_note_to_page(folder: Path | str, scam_type: str, note: dict) -> Path:
    """Append a note to the type's page, creating the page with a title if it is missing."""
    path = page_path(folder, scam_type)
    section = _render_note(note)
    parse_page(section, "<new note>")  # refuse a note the parser would reject later
    if path.exists():
        text = path.read_text(encoding=ENCODING).rstrip("\n") + "\n\n" + section
    else:
        text = render_page(TITLES[scam_type], []) + "\n" + section
    _write(path, text.rstrip("\n") + "\n")
    return path


def set_active(folder: Path | str, scam_type: str, number: int, active: bool) -> str:
    """Set the 'active:' line of the note at 1-based position `number`. Returns its summary.

    Only that one line changes; the rest of the page is left as it was.
    """
    path = page_path(folder, scam_type)
    name = path.as_posix()
    text = path.read_text(encoding=ENCODING)
    notes = parse_page(text, name)
    if not 1 <= number <= len(notes):
        raise ValueError(f"{name} has {len(notes)} notes; there is no note {number}")
    lines = text.splitlines()
    value = "yes" if active else "no"
    headings = [i for i, line in enumerate(lines) if line.strip().startswith("## ")]
    first = headings[number - 1]
    end = headings[number] if number < len(headings) else len(lines)
    for j in range(first + 1, end):
        found = _ACTIVE_LINE.match(lines[j])
        if found:
            lines[j] = f"{found.group(1)} {value}"
            break
    else:
        lines.insert(first + 1, f"- active: {value}")
    _write(path, "\n".join(lines) + "\n")
    return notes[number - 1]["summary"]


def sync(
    store: Store, folder: Path | str = DEFAULT_WIKI, scam_types: tuple[str, ...] = SCAM_TYPES
) -> dict[str, dict[str, int]]:
    """Make the store's notes for each type match its page exactly.

    Every page is parsed before the store is touched, so a mistake in one page changes
    nothing. A folder with none of the five pages is refused (it is probably the wrong
    folder, and syncing would delete every note). A type whose page is missing keeps the
    notes it has, with a warning, instead of losing them.
    Returns {scam_type: {"notes": total, "active": active}}.
    """
    folder = Path(folder)
    wiki = read_wiki(folder)
    if not any(page_path(folder, t).exists() for t in SCAM_TYPES):
        names = ", ".join(f"{t}.md" for t in SCAM_TYPES)
        raise WikiError(f"{folder} has none of the wiki pages ({names}); is this the right folder?")
    counts = {}
    for scam_type in scam_types:
        if not page_path(folder, scam_type).exists() and store.note_counts(scam_type)[0]:
            kept = store.note_counts(scam_type)[0]
            print(
                f"warning: {page_path(folder, scam_type).as_posix()} is missing; "
                f"kept the {kept} stored {scam_type} notes",
                file=sys.stderr,
            )
        else:
            store.replace_notes(scam_type, wiki[scam_type])
        total, active = store.note_counts(scam_type)
        counts[scam_type] = {"notes": total, "active": active}
    return counts
