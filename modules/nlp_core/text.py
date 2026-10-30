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
# A vowel or tone mark written above or below a letter (ั ิ ี ึ ื ุ ู ฺ ็ ่ ้ ๊ ๋ ์ ...).
_MARK = re.compile("[ัิ-ฺ็-๎]")
# A consonant typed twice (ดด, มม) is a doubled letter, not a misplaced mark; รร is real Thai.
_DOUBLED = re.compile(r"([ก-ฮ])\1")
# Words the call simulator uses that no PyThaiNLP word list has, so they are not "corrected".
_KEEP_WORDS = frozenset({"คริปโต", "คอนเนกชัน", "แอคเคาท์", "พอดแคสต์", "ไฮแลนด์", "แอป", "เฮ้", "ฟิน"})


def normalize_text(text: str) -> str:
    """NFC, PyThaiNLP normalize (fixes doubled tone marks) and single spaces between words."""
    # "。" (ideographic full stop) turns up in some ASR output; treat it as a space
    text = unicodedata.normalize("NFC", text).replace("。", " ")
    return " ".join(thai_normalize(text).split())


@cache
def _dictionary() -> frozenset[str]:
    """The words a fix may produce: PyThaiNLP's main word list."""
    from pythainlp.corpus import thai_words

    return frozenset(thai_words())


@cache
def _known() -> frozenset[str]:
    """Words left alone: the dictionary plus _KEEP_WORDS."""
    return _dictionary() | _KEEP_WORDS


@cache
def _known_trie():
    """The known words for the tokenizer, so a loanword stays one piece (แอป, not แอ ป)."""
    from pythainlp.util import Trie

    return Trie(_known())


@cache
def _names() -> frozenset[str]:
    from pythainlp.corpus import thai_family_names, thai_female_names, thai_male_names

    return frozenset(thai_family_names() | thai_female_names() | thai_male_names())


@lru_cache(maxsize=4096)
def _fix_word(word: str) -> str:
    """The word with one misplaced vowel or tone mark moved back (คณุ to คุณ), else unchanged.

    Only marks move, so a fix never turns a word into a different-sounding one; the result must
    be the one dictionary word made that way, or split into fewer words, all known. Words
    PyThaiNLP's spell() would guess are left as they are: on stored calls its guesses were
    wrong far more often than right (โอนเงนิ to ถอนเงิน, แอป to แผน).
    """
    if len(word) < 3:  # one or two letters have too many near words to pick from
        return word
    if any(m.group(1) != "ร" for m in _DOUBLED.finditer(word)):  # ก็ดดี, not a moved mark
        return word
    from pythainlp.tokenize import word_tokenize

    words = _dictionary()
    pieces = len(word_tokenize(word, engine="newmm", keep_whitespace=False))

    def is_better(spelling: str) -> bool:
        if spelling in words:
            return True
        split = word_tokenize(spelling, engine="newmm", keep_whitespace=False)
        return len(split) < pieces and all(len(p) > 1 and p in words for p in split)

    swaps = {
        word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        for i in range(len(word) - 1)
        if _MARK.match(word[i]) or _MARK.match(word[i + 1])
    }
    found = {s for s in swaps if is_better(s)}
    return found.pop() if len(found) == 1 else word


def autocorrect_text(text: str, mode: str | None = None) -> str:
    """Fix misplaced vowels and tone marks in Thai words that are not known. Leaves digits,
    English, names, URLs and known loanwords.

    mode is "pythainlp" or "off"; None reads the AUTOCORRECT setting, which is "off" by default.
    """
    if mode is None:
        from callas_settings import get_settings

        mode = get_settings().autocorrect
    if mode != "pythainlp":
        return text
    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(text, custom_dict=_known_trie(), engine="newmm", keep_whitespace=True)
    return "".join(_correct_tokens(tokens))


def _correct_tokens(tokens: list[str]) -> list[str]:
    """Fix each unknown Thai token: alone first, else together with its Thai neighbours."""
    known, names = _known(), _names()
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
            bool(_THAI_WORD.fullmatch(token)) and token not in known and token not in names
        )
        if not is_unknown_word or after_title:
            corrected.append(token)
            i += 1
            continue

        # The piece alone first (ให้|คณุ|นะคะ), so its neighbours are never rewritten.
        alone = _fix_word(token)
        if alone != token:
            corrected.append(alone)
            i += 1
            continue

        # A misspelled word can be cut into pieces ("ขอบ" "ค" "นุ"), so try the unknown piece
        # together with the Thai pieces next to it.
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
