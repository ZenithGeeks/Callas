"""Spelling gate for spoken replies. The voice reads the reply aloud, so garbled Thai
(a word outside the dictionary, or a tone mark written before an upper vowel) falls back
to a fixed template. False alarms only cost a template, never a leak."""

from __future__ import annotations

import re
from functools import cache

_THAI = re.compile(r"[฀-๿]")
# Garbled marks: a tone mark (่ ้ ๊ ๋) before an upper vowel (ั ิ ี ึ ื) instead of after it, or
# two tone marks or two upper vowels in a row (เต้้น, กััน, นั้้น). None of these are valid Thai.
_BAD_MARKS = re.compile(r"[่-๋][ัิ-ื]|[่-๋]{2,}|[ัิ-ื]{2,}")


@cache
def _words() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


def spelling_problems(text: str) -> list[str]:
    """Thai tokens that look misspelled (empty list = clean)."""
    if _BAD_MARKS.search(text):
        return ["tone_mark_order"]
    from pythainlp.tokenize import word_tokenize

    words = _words()
    # ๆ (mai yamok) repeats the word before it (สู้ๆ, เยอะๆ); it is not a word, so it is
    # dropped before the dictionary lookup, whether newmm splits it off or keeps it attached.
    return [
        t
        for t in word_tokenize(text, engine="newmm")
        if _THAI.search(w := t.strip().rstrip("ๆ").strip()) and w not in words
    ]


def looks_misspelled(text: str) -> bool:
    """True if `text` has a word outside the Thai dictionary or a misplaced tone mark."""
    return bool(spelling_problems(text))
