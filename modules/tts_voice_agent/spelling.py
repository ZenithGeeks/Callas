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
# Spoken exclamations Thai people say that the PyThaiNLP dictionary lacks (โถ่ in romance_00015).
_SPOKEN = frozenset({"โถ่", "เฮ้", "เฮ่", "เห้ย", "โห้", "เอ้า", "เอิ่ม", "หืม", "ห๊ะ", "ห้ะ", "งื้อ", "อุ๊ต๊ะ"})
# Isan words the dictionary lacks, for receivers whose persona speaks Isan (จังได๋ in
# phishing_otp_00015). ได๋ and ใด๋ alone also cover compounds such as ปานใด๋ and ผู้ได๋.
_ISAN = frozenset(
    {
        "ได๋",
        "ใด๋",
        "จังได๋",
        "จั่งได๋",
        "จังซั่น",
        "จั่งซั่น",
        "จั่งซี่",
        "อิหยัง",
        "เฮา",
        "เพิ่น",
        "จั๊กแม่น",
        "แหน่",
        "เบิ๊ด",
        "น้อ",
    }
)
# Loanwords the dictionary lacks, so newmm split them into non-words (แอ+ป, ค+ริ+ป+โต, คอ+ล).
_LOANWORDS = frozenset({"แอป", "คริปโต", "คอล"})
# ๆ (mai yamok) repeats the word before it (สู้ๆ, เยอะๆ) and ฯ (paiyannoi) shortens it (แอปฯ,
# กรมฯ); neither is a word, so both are dropped before the dictionary lookup, whether newmm
# splits them off or keeps them attached.
_MARKS_AFTER_WORD = "ๆฯ"


@cache
def _words() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words()) | _SPOKEN | _ISAN | _LOANWORDS


@cache
def _trie():
    # newmm splits by its own dictionary, so it needs ours to keep แอป or โถ่ whole
    from pythainlp.util import dict_trie

    return dict_trie(_words())


def spelling_problems(text: str) -> list[str]:
    """Thai tokens that look misspelled (empty list = clean)."""
    if _BAD_MARKS.search(text):
        return ["tone_mark_order"]
    from pythainlp.tokenize import word_tokenize

    words = _words()
    # a dictionary entry can itself end in ๆ (เล็กๆ น้อยๆ), so the token is looked up as is first
    return [
        t
        for t in word_tokenize(text, engine="newmm", custom_dict=_trie())
        if t.strip() not in words
        and _THAI.search(w := t.strip().rstrip(_MARKS_AFTER_WORD).strip())
        and w not in words
    ]


def looks_misspelled(text: str) -> bool:
    """True if `text` has a word outside the Thai dictionary or a misplaced tone mark."""
    return bool(spelling_problems(text))
