"""Thai number words back to digits, for transcripts (issue #53).

The ASR writes numbers the way they are said and the rule NER (entities.py) only reads digits.
Start with `spoken_numbers_to_digits` at the bottom. What counts as a number depends on what
it is (phone, account, money, code); the rules and examples are in README.md.
"""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

MIN_DIGITS = 4  # digit words read one by one that make a number
_DIGITS = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "โท": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
}  # fmt: skip
_MULT = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10_000, "แสน": 100_000, "ล้าน": 1_000_000}
_YEE = "ยี่"  # 20 as in ยี่สิบ
_DASH_WORDS = ("ขีด", "ลบ")  # ลบ is also a real word, so a dash only between two numbers
# Shared with the reply leak check (tts_voice_agent.leak).
THAI_NUMBER_WORDS = (*_DIGITS, *_MULT, _YEE)
ENGLISH_DIGIT_WORDS = "zero one two three four five six seven eight nine oh".split()
_EN_VALUE = {w: str(i) for i, w in enumerate(ENGLISH_DIGIT_WORDS[:10])} | {"oh": "0"}

_UNITS = ("บาท", "ดอลลาร์", "เหรียญ", "หยวน", "เยน", "ยูโร", "ปอนด์", "สตางค์")
_WORD_ALT = "|".join(sorted((*THAI_NUMBER_WORDS, *_DASH_WORDS), key=len, reverse=True))
_NUMBER_WORD = re.compile(_WORD_ALT)
_UNIT_AFTER = re.compile(rf"\s*(?:{'|'.join(_UNITS)})")
_COUNTER_AFTER = re.compile(r"\s*(?:คน|ปี|วัน|เดือน|ครั้ง|ชั้น|ตัว|ชิ้น|ข้อ|ที่)")
_MONEY_BEFORE = re.compile(r"(?:เงิน|โอน|ยอด|ค่า|ราคา|จำนวน)\s*$")
_CODE_BEFORE = re.compile(r"(?:otp|โอทีพี|รหัส|โค้ด|code)[^\d]{0,15}$", re.I)
# How far past the blanks the checks look: a keyword plus up to 15 characters before, a unit after.
_LOOK_BEFORE = 21
_LOOK_AFTER = 8
_EN_ALT = "|".join(ENGLISH_DIGIT_WORDS)
_EN_RUN = re.compile(rf"\b(?:{_EN_ALT})\b(?:[\s-]+\b(?:{_EN_ALT})\b){{{MIN_DIGITS - 1},}}", re.I)


class _Reading(NamedTuple):
    """One group of number words read as digits ("digits") or as a place value ("value")."""

    kind: Literal["digits", "value"]
    text: str


def _place_value(words: list[str]) -> int | None:
    """Value of a place-value reading (หนึ่งร้อยห้าสิบแปด = 158), None if it is not one."""
    if not words or not any(w in _MULT or w == _YEE for w in words):
        return None
    if "ล้าน" in words:
        return _value_with_millions(words)
    return _value_below_million(words)


def _digit_or_none(words: list[str]) -> int | None:
    """The value of a single digit word, None for anything else."""
    return _DIGITS[words[0]] if len(words) == 1 and words[0] in _DIGITS else None


def _number_or_none(words: list[str]) -> int | None:
    """A place-value reading, or a single digit word."""
    return _place_value(words) or _digit_or_none(words)


def _value_with_millions(words: list[str]) -> int | None:
    """Split at the last ล้าน: the words before it count millions, the words after it the rest."""
    i = len(words) - 1 - words[::-1].index("ล้าน")
    head, tail = words[:i], words[i + 1 :]
    millions = _number_or_none(head) if head else 1  # "ล้านบาท" = 1 ล้าน
    rest = _number_or_none(tail) if tail else 0
    if millions is None or rest is None or rest >= 1_000_000:
        return None
    return millions * 1_000_000 + rest


def _value_below_million(words: list[str]) -> int | None:
    """Add up the groups of a reading with no ล้าน: digit word, then its multiplier."""
    total, pending, last = 0, None, 10**9
    for i, w in enumerate(words):
        if w == _YEE:
            if pending is not None or words[i + 1 : i + 2] != ["สิบ"]:
                return None
            pending = 2
        elif w in _DIGITS:
            if pending is not None or w in ("ศูนย์", "โท") or (w == "เอ็ด" and i != len(words) - 1):
                return None
            pending = _DIGITS[w]
        else:
            m = _MULT[w]
            if m >= last:
                return None
            total += (1 if pending is None else pending) * m
            pending, last = None, m
    return total + (pending or 0)


def _group(words: list[str]) -> _Reading | None:
    """Read one group of number words, None if it does not parse."""
    if words and all(w in _DIGITS for w in words):
        return _Reading("digits", "".join(str(_DIGITS[w]) for w in words))
    value = _place_value(words)
    return None if value is None else _Reading("value", str(value))


def _split_on_dashes(words: list[str]) -> list[list[str]] | None:
    """Groups of number words between dashes, None if a dash has no number on one side."""
    groups: list[list[str]] = [[]]
    for w in words:
        if w not in _DASH_WORDS:
            groups[-1].append(w)
        elif groups[-1]:
            groups.append([])
        else:
            return None
    return groups if groups[-1] else None


def _read_run(words: list[str], before: str, after: str) -> str | None:
    """Digits for one run of number words and dashes, None to leave it alone."""
    groups = _split_on_dashes(words)
    if groups is None:
        return None
    readings = [_group(g) for g in groups]
    if None in readings:
        return None
    if len(readings) > 1:
        return _read_dashed(readings)
    reading = readings[0]
    if reading.kind == "digits":
        return _read_digit_words(words, reading.text, after)
    return _read_place_value(reading.text, before, after)


def _read_dashed(readings: list[_Reading]) -> str | None:
    """An identifier with dashes, or the place-value fallback."""
    digits = sum(len(r.text) for r in readings)
    all_digit_words = all(r.kind == "digits" for r in readings)
    if (all_digit_words and digits >= 6) or digits >= 8 or (len(readings) >= 3 and digits >= 6):
        return "-".join(r.text for r in readings)
    return None


def _read_digit_words(words: list[str], digits: str, after: str) -> str | None:
    """Digit words read one by one: convert a long run, or one word before a currency unit."""
    if _UNIT_AFTER.match(after) and len(words) == 1:
        return None if words[0] == "เอ็ด" else digits  # เอ็ด is 1 only after สิบ or in a run
    return digits if len(words) >= MIN_DIGITS else None


def _read_place_value(value: str, before: str, after: str) -> str | None:
    """Convert only when the text around the number says it is money or a code."""
    if _UNIT_AFTER.match(after):
        return value
    if _COUNTER_AFTER.match(after):  # "จำนวนสามสิบคน" is a head count, not money
        return None
    if _MONEY_BEFORE.search(before):
        return value
    return value if int(value) >= 1000 and _CODE_BEFORE.search(before) else None


def _token_edges(text: str) -> set[int]:
    """Offsets where a PyThaiNLP newmm token starts or ends."""
    from pythainlp.tokenize import word_tokenize

    edges, offset = {0}, 0
    for token in word_tokenize(text, engine="newmm", keep_whitespace=True):
        offset += len(token)
        edges.add(offset)
    return edges


def _runs(matches: list[re.Match[str]], text: str) -> list[list[re.Match[str]]]:
    """Group number words that are at most two blank characters apart."""
    runs = [[matches[0]]]
    for previous, current in zip(matches, matches[1:], strict=False):
        gap = text[previous.end() : current.start()]
        if gap.strip() or len(gap) > 2:
            runs.append([])
        runs[-1].append(current)
    return runs


def _trim_dashes(run: list[re.Match[str]]) -> list[re.Match[str]]:
    """Drop a dash at either end: it is an ordinary word ("ลบ" = delete), not part of the number."""
    while run and run[-1].group() in _DASH_WORDS:
        run = run[:-1]
    while run and run[0].group() in _DASH_WORDS:
        run = run[1:]
    return run


def _thai_edits(text: str) -> list[tuple[int, int, str]]:
    """(start, end, replacement) for each run of Thai number words that should become digits."""
    matches = list(_NUMBER_WORD.finditer(text))
    if not matches:
        return []
    word_edges = _token_edges(text)
    edits = (_edit_for_run(text, run, word_edges) for run in _runs(matches, text))
    return [edit for edit in edits if edit is not None]


def _around(text: str, start: int, end: int) -> tuple[str, str]:
    """The text just before and after a run, cut short so that a long text costs nothing extra.

    The checks in _read_run only look a few characters away, past any blanks next to the run.
    """
    first = start
    while first > 0 and text[first - 1].isspace():
        first -= 1
    last = end
    while last < len(text) and text[last].isspace():
        last += 1
    return text[max(0, first - _LOOK_BEFORE) : start], text[end : last + _LOOK_AFTER]


def _edit_for_run(
    text: str, run: list[re.Match[str]], word_edges: set[int]
) -> tuple[int, int, str] | None:
    """The replacement for one run of number words, None to leave the run as it is."""
    run = _trim_dashes(run)
    # only whole words: "หนึ่งเดียว" is one token, so its "หนึ่ง" is not a number
    # (tokens such as "สิบสอง" hold several number words, so only the run's ends are checked)
    if not run or run[0].start() not in word_edges or run[-1].end() not in word_edges:
        return None
    words = [m.group() for m in run]
    start, end = run[0].start(), run[-1].end()
    before, after = _around(text, start, end)
    digits = _read_run(words, before, after)
    if digits is None:
        return None
    # a range such as "ห้าสิบ-หกสิบบาท": both halves convert or neither; only a long digit
    # run keeps converting next to a hyphen
    long_digits = len(words) >= MIN_DIGITS and all(w in _DIGITS or w in _DASH_WORDS for w in words)
    next_to_hyphen = before.rstrip().endswith("-") or after.lstrip().startswith("-")
    if next_to_hyphen and not long_digits:
        return None
    pad_left = " " if before and not before[-1].isspace() else ""
    pad_right = " " if after and not after[0].isspace() else ""
    return start, end, pad_left + digits + pad_right


def spoken_numbers_to_digits(text: str) -> str:
    """Rewrite Thai (and English) number words as digits; see the module docstring for the rule."""
    text = _EN_RUN.sub(
        lambda m: "".join(_EN_VALUE[w.lower()] for w in re.findall(r"[a-z]+", m.group(), re.I)),
        text,
    )
    pieces, position = [], 0
    for start, end, out in _thai_edits(text):  # in order, and they do not overlap
        pieces += [text[position:start], out]
        position = end
    pieces.append(text[position:])
    return "".join(pieces)
