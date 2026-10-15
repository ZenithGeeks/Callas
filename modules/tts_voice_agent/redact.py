"""Deterministic redaction of call text before any LLM writes a reply.

The reply LLM must never see what the caller said that is sensitive: entities seen in the
call, digits in any script, spoken numbers, links. They become `[LABEL]`, `[NUMBER]` and
`[LINK]`. The Thai text around them stays as it was (only matching is done on a normalised
copy; the output is cut from the original string).

Property (tested): for the output, `leaks_sensitive` never fires for the reasons `digit`,
`digit_word_run`, `url_or_email` or `entity:*`. Keyword, name, organisation and Latin-word
reasons can still fire; those are the reply's problem, not the context's.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from nlp_core.types import Entity

from .leak import (
    SPOKEN_ENGLISH_RUN,
    SPOKEN_THAI_RUN,
    STRIP_FOR_COMPARE,
    URL_OR_EMAIL,
    leak_reasons,
    normalize,
)

NUMBER = "[NUMBER]"
LINK = "[LINK]"
FALLBACK = "[REDACTED]"
_MAX_PASSES = 3
_DIGIT_SEPARATORS = " \t\r\n-.,/"
_THAI_RANGE = re.compile(r"[฀-๿]")
_BLOCKING = ("digit", "digit_word_run", "url_or_email", "entity:")

Span = tuple[int, int, str]  # start, end (exclusive) in the original text, replacement


def _view(text: str, compact: bool) -> tuple[str, list[int]]:
    """Normalised, lower-cased copy of `text` and, for each of its characters, the index of
    the original character it came from. `compact` also drops whitespace, hyphens and dots."""
    chars: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        piece = normalize(ch).lower()
        if compact:
            piece = STRIP_FOR_COMPARE.sub("", piece)
        for c in piece:
            chars.append(c)
            origin.append(i)
    return "".join(chars), origin


def _regex_spans(
    text: str, rx: re.Pattern[str], placeholder: str, compact: bool, expand: bool = False
) -> list[Span]:
    view, origin = _view(text, compact)
    spans: list[Span] = []
    for m in rx.finditer(view):
        if m.end() == m.start():
            continue
        start, end = origin[m.start()], origin[m.end() - 1] + 1
        if expand:  # a link is a whole token: take the Latin characters around the match
            while start > 0 and _is_token_char(text[start - 1]):
                start -= 1
            while end < len(text) and _is_token_char(text[end]):
                end += 1
        spans.append((start, end, placeholder))
    return spans


def _is_token_char(ch: str) -> bool:
    return not ch.isspace() and not _THAI_RANGE.match(ch)


def _entity_spans(text: str, entities: Iterable[Entity]) -> list[Span]:
    view, origin = _view(text, compact=True)
    spans: list[Span] = []
    ordered = sorted(entities, key=lambda e: len(compact_text(e.text)), reverse=True)
    for ent in ordered:  # longest first, so a short name inside a long one cannot split it
        needle = compact_text(ent.text)
        if len(needle) < 2:  # same floor as the leak check
            continue
        pos = view.find(needle)
        while pos != -1:
            spans.append((origin[pos], origin[pos + len(needle) - 1] + 1, f"[{ent.label}]"))
            pos = view.find(needle, pos + len(needle))
    return spans


def compact_text(text: str) -> str:
    return STRIP_FOR_COMPARE.sub("", normalize(text)).lower()


def _has_digit(ch: str) -> bool:
    return any(unicodedata.decimal(c, None) is not None for c in normalize(ch))


def _digit_spans(text: str) -> list[Span]:
    """Every run of digit characters (any script), joined across separators like - . /"""
    spans: list[Span] = []
    i, n = 0, len(text)
    while i < n:
        if not _has_digit(text[i]):
            i += 1
            continue
        end = i + 1
        j = end
        while j < n:
            ch = text[j]
            if _has_digit(ch):
                j += 1
                end = j
            elif ch in _DIGIT_SEPARATORS or not normalize(ch):  # not normalize: zero-width
                j += 1
            else:
                break
        spans.append((i, end, NUMBER))
        i = end
    return spans


def _apply(text: str, spans: list[Span]) -> str:
    """Replace spans (overlaps merged; the first placeholder of a merged group wins)."""
    if not spans:
        return text
    spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[list] = []
    for start, end, ph in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end, ph])
    out, pos = [], 0
    for start, end, ph in merged:
        out.append(text[pos:start])
        out.append(ph)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _one_pass(text: str, entities: list[Entity]) -> str:
    text = _apply(text, _entity_spans(text, entities))
    text = _apply(text, _digit_spans(text))
    spoken: list[Span] = []
    for compact in (False, True):
        spoken += _regex_spans(text, SPOKEN_THAI_RUN, NUMBER, compact)
        spoken += _regex_spans(text, SPOKEN_ENGLISH_RUN, NUMBER, compact)
    text = _apply(text, spoken)
    links: list[Span] = []
    for compact in (False, True):
        links += _regex_spans(text, URL_OR_EMAIL, LINK, compact, expand=True)
    return _apply(text, links)


def _still_unsafe(text: str, entities: list[Entity]) -> bool:
    return any(r.startswith(_BLOCKING) for r in leak_reasons(text, entities))


def redact_context(text: str, entities_seen: Iterable[Entity]) -> str:
    """Return `text` with sensitive details replaced by placeholders. Pure and deterministic."""
    entities = list(entities_seen)
    out = text or ""
    for _ in range(_MAX_PASSES):
        out = _one_pass(out, entities)
        if not _still_unsafe(out, entities):
            return out
    return FALLBACK  # should not happen; never hand the model text we could not clean
