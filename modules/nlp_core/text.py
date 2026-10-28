"""Transcript text cleanup shared by the web system and the classifier notebook.

normalize_text is the exact function the notebook trains with, so a model sees the same form
of text in training and in the web system. autocorrect_text is optional (AUTOCORRECT setting).
"""

from __future__ import annotations

import re
import unicodedata
from functools import cache, lru_cache

from pythainlp.util import normalize as thai_normalize

_TITLES = frozenset({"คุณ", "นาย", "นาง", "นางสาว", "พี่", "น้อง", "ดร", "ผู้กอง", "จ่า"})
# Thai letters, vowels and tone marks only: no digits (Thai or Arabic), Latin, URLs, symbols
# or the repeat mark ๆ (U+0E46).
_THAI_WORD = re.compile("[ก-ฺเ-ๅ็-๎]+")


def normalize_text(text: str) -> str:
    """NFC, PyThaiNLP normalize (fixes doubled tone marks) and single spaces between words."""
    # "。" (ideographic full stop) turns up in some ASR output; treat it as a space
    text = unicodedata.normalize("NFC", text).replace("。", " ")
    return " ".join(thai_normalize(text).split())


@cache
def _spelling_fix_pattern() -> re.Pattern[str]:
    from .spelling_fixes import SPELLING_FIXES

    # longest first, so ใหคณุ is replaced as a whole before its part ใหคณ
    wrong = sorted(SPELLING_FIXES, key=len, reverse=True)
    return re.compile("|".join(map(re.escape, wrong)))


def fix_thai_spelling(text: str) -> str:
    """normalize_text, then the reviewed garbles in spelling_fixes.SPELLING_FIXES.

    Fixes marks (doubled vowels and tone marks, a tone mark before its vowel, เเ for แ) and the
    listed words only, so it never guesses: a word it does not know stays as written. Digits,
    English, {SLOTS} and names are left alone.
    """
    from .spelling_fixes import SPELLING_FIXES

    text = normalize_text(text)
    return _spelling_fix_pattern().sub(lambda m: SPELLING_FIXES[m.group()], text)


@cache
def _dictionary() -> frozenset[str]:
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


@cache
def _names() -> frozenset[str]:
    from pythainlp.corpus import thai_family_names, thai_female_names, thai_male_names

    return frozenset(thai_family_names() | thai_female_names() | thai_male_names())


@lru_cache(maxsize=4096)
def _fix_word(word: str) -> str:
    """PyThaiNLP's best spelling for a word outside the dictionary; kept only if it is a word."""
    if len(word) < 3:  # one or two letters have too many near words to pick from
        return word
    from pythainlp.spell import correct

    fixed = correct(word)
    # Never shorten: correct("แอป") is "แอ", so a loanword would lose its last letter.
    return fixed if len(fixed) >= len(word) and fixed in _dictionary() else word


def autocorrect_text(text: str, mode: str | None = None) -> str:
    """Fix Thai words that are not in the dictionary. Leaves digits, English, names and URLs.

    mode is "pythainlp" or "off"; None reads the AUTOCORRECT setting, which is "off" by default.
    It is off because PyThaiNLP's correct() takes about 0.4 s per unknown word and can pick a
    wrong word ("โอนเงนิ" becomes "ถอนเงิน"). Swap in a context-aware model if the three
    checks in issue #30 do not pass.
    """
    if mode is None:
        from callas_settings import get_settings

        mode = get_settings().autocorrect
    if mode != "pythainlp":
        return text
    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(text, engine="newmm", keep_whitespace=True)
    return "".join(_correct_tokens(tokens))


def _correct_tokens(tokens: list[str]) -> list[str]:
    """Replace each unknown Thai token (with its Thai neighbours) by its best spelling."""
    words, names = _dictionary(), _names()
    # `corrected` is the output so far. The previous Thai token is taken back out of it so it
    # can be corrected together with the unknown one; `i` skips the next token only when the
    # corrected word already includes it.
    corrected: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # a name the lists do not know is cut into several pieces after its title
        after_title = any(t in _TITLES for t in corrected[-4:])
        is_unknown_word = (
            bool(_THAI_WORD.fullmatch(token)) and token not in words and token not in names
        )
        if not is_unknown_word or after_title:
            corrected.append(token)
            i += 1
            continue

        # A misspelled word is cut into pieces ("ขอบ" "ค" "นุ"), so correct the unknown
        # piece together with the Thai pieces next to it.
        before = corrected.pop() if corrected and _THAI_WORD.fullmatch(corrected[-1]) else ""
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        after = next_token if _THAI_WORD.fullmatch(next_token) else ""
        candidate = before + token + after
        fixed = _fix_word(candidate)
        if fixed == candidate:
            corrected.extend(piece for piece in (before, token) if piece)
            i += 1
        else:
            corrected.append(fixed)
            i += 2 if after else 1
    return corrected
