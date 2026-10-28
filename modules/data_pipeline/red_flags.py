"""Common scam warning signs for the receiver, built from the scam wiki (docs/scam-wiki/).

Every receiver reads the same short digest, whatever the call is about. It is the union of the
active notes across all five scam-type pages, so it never names this call's own type: the
receiver stays blind to the label (see the spec) and only gets general "things people get warned
about". The persona's scam_awareness still decides whether the receiver recognises or acts on any
of it — an alert person may, a trusting or unaware one may not.

Advisory, not essential: unlike the caller's romance_voice.md, a receiver works fine without it,
so a missing or malformed wiki drops the digest (returns "") instead of breaking generation.
"""

from __future__ import annotations

from pathlib import Path

from .wiki import WikiError, read_wiki

WIKI_DIR = Path(__file__).resolve().parents[2] / "docs" / "scam-wiki"
PER_TYPE = 1  # active note summaries taken per page: one per scam type keeps it general and short
MAX_FLAGS = 8  # hard cap: this goes into every receiver prompt, so keep it short


def digest(folder: Path | str = WIKI_DIR) -> str:
    """A short, type-agnostic list of scam warning signs for the receiver prompt.

    Up to PER_TYPE active note summaries from each page, deduplicated, then sorted and capped at
    MAX_FLAGS so the result is the same for every call and bounded in size. Returns "" when the
    wiki is missing or malformed (the digest is advisory; it must not break a run).
    """
    try:
        pages = read_wiki(folder)
    except (WikiError, FileNotFoundError):
        return ""
    flags: list[str] = []
    seen: set[str] = set()
    for scam_type in sorted(pages):
        taken = 0
        for note in pages[scam_type]:
            if not note["active"]:
                continue
            summary = " ".join(note["summary"].split())
            key = summary.lower()
            if key in seen:
                continue
            seen.add(key)
            flags.append(summary)
            taken += 1
            if taken >= PER_TYPE:
                break
    return "\n".join(f"- {flag}" for flag in sorted(flags)[:MAX_FLAGS])
