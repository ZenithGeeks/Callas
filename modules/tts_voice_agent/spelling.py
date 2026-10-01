"""Spelling gate for spoken replies. The voice reads the reply aloud, so garbled Thai
(a word outside the dictionary, or a tone mark written before an upper vowel) falls back
to a fixed template. False alarms only cost a template, never a leak."""

from __future__ import annotations

import re
from functools import cache

_THAI = re.compile(r"[฀-๿]")
# A tone mark (่ ้ ๊ ๋) must come after an upper vowel (ั ิ ี ึ ื), never before it.
_TONE_BEFORE_VOWEL = re.compile(r"[่-๋][ัิ-ื]")


@cache
def _words() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


def spelling_problems(text: str) -> list[str]:
    """Thai tokens that look misspelled (empty list = clean)."""
    if _TONE_BEFORE_VOWEL.search(text):
        return ["tone_mark_order"]
    from pythainlp.tokenize import word_tokenize

    words = _words()
    return [
        t for t in word_tokenize(text, engine="newmm") if _THAI.search(t) and t.strip() not in words
    ]


def looks_misspelled(text: str) -> bool:
    return bool(spelling_problems(text))
