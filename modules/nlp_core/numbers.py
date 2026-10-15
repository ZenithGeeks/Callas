"""Thai number words back to digits, for transcripts. Issue #53.

The ASR writes numbers the way they are said, and the rule NER (entities.py) only reads digits.
spoken_numbers_to_digits turns such words back into digits. How a number is read depends on
what it is, so the rule follows that. Everything that is not a number below is left as it is.

1. Identifiers (phone, bank account, OTP, ID card) are read digit by digit, with ขีด for a
   dash: "หนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสอง" is 158-5-96045-2. Digit words are
   ศูนย์ หนึ่ง สอง/โท สาม สี่ ห้า หก เจ็ด แปด เก้า (เอ็ด as 1). Speakers also pause between
   groups or run everything together, so spaces between digit words are dropped
   ("หนึ่งห้าแปด ห้า" is 1585) and only ขีด/ลบ becomes "-". At least MIN_DIGITS (4) digit
   words in a row, or groups joined by ขีด/ลบ with 6 digits or more in all. "สองคน",
   "สามสี่วัน" and "ห้าลบสอง" stay.
2. Money is read with place value (สิบ ร้อย พัน หมื่น แสน ล้าน, ยี่สิบ, สิบเอ็ด): it is
   converted when a unit follows (บาท, ดอลลาร์, ...) or a money word comes right before it
   (เงิน โอน ยอด ค่า ราคา จำนวน). "โอนหนึ่งร้อยห้าสิบแปดบาท" is "โอน 158 บาท". A single
   digit word before a unit ("ห้าบาท") converts too. When the ASR drops พัน/หมื่น ("สี่สิบเก้า
   ห้าร้อยบาท", "หนึ่ง สองร้อยเก้าสิบบาท") two pieces split by a space become thousands plus
   hundreds (49500, 1290).
3. A place-value number with no money context stays words: "สามสิบคน", "สิบวัน", "ร้อยละห้า".
   The one exception is a code: after รหัส / OTP / โค้ด a place-value value of 1000 or more
   converts, as an ASR may read an OTP like a number.
4. Low priority fallback: place-value groups joined by ลบ/ขีด ("หนึ่งร้อยห้าสิบแปดลบห้าลบ...",
   the form our own TTS produced) need 3 groups and 6 digits, or 8 digits.
5. English digit words ("zero eight one ...") with at least 4 in a row.

A number word only counts when it is a whole word for PyThaiNLP's newmm tokenizer, so
หนึ่งเดียว, ร้อยละ, สามารถ, สามี, ห้าง, ร้อยตำรวจ, สิบเอก, พันธุ์, โทร, ปริญญาโท are never read
as numbers. Text with no number word is returned without being tokenized.
"""

from __future__ import annotations

import re

MIN_DIGITS = 4  # digit words read one by one that make a number
_DIGITS = {
    "ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "โท": 2, "สาม": 3, "สี่": 4,
    "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9,
}  # fmt: skip
_MULT = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10_000, "แสน": 100_000, "ล้าน": 1_000_000}
_YEE = "ยี่"  # 20 as in ยี่สิบ
_DASH = ("ขีด", "ลบ")  # ลบ is also a real word, so a dash only between two numbers
# Shared with the reply leak check (tts_voice_agent.leak).
THAI_NUMBER_WORDS = (*_DIGITS, *_MULT, _YEE)
ENGLISH_DIGIT_WORDS = "zero one two three four five six seven eight nine oh".split()
_EN_VALUE = {w: str(i) for i, w in enumerate(ENGLISH_DIGIT_WORDS[:10])} | {"oh": "0"}

_UNITS = ("บาท", "ดอลลาร์", "เหรียญ", "หยวน", "เยน", "ยูโร", "ปอนด์", "สตางค์")
_ALT = "|".join(sorted((*THAI_NUMBER_WORDS, *_DASH), key=len, reverse=True))
_WORD = re.compile(_ALT)
_UNIT_AFTER = re.compile(rf"\s*(?:{'|'.join(_UNITS)})")
_MONEY_BEFORE = re.compile(r"(?:เงิน|โอน|ยอด|ค่า|ราคา|จำนวน)\s*$")
_CODE_BEFORE = re.compile(r"(?:otp|โอทีพี|รหัส|โค้ด|code)[^\d]{0,15}$", re.I)
_EN_ALT = "|".join(ENGLISH_DIGIT_WORDS)
_EN_RUN = re.compile(rf"\b(?:{_EN_ALT})\b(?:[\s-]+\b(?:{_EN_ALT})\b){{{MIN_DIGITS - 1},}}", re.I)


def _place_value(words: list[str]) -> int | None:
    """Value of a place-value reading (หนึ่งร้อยห้าสิบแปด = 158), None if it is not one."""
    if not words or not any(w in _MULT or w == _YEE for w in words):
        return None
    if "ล้าน" in words:
        i = len(words) - 1 - words[::-1].index("ล้าน")
        head, tail = words[:i], words[i + 1 :]
        millions = _place_value(head) if len(head) > 1 else _digit_or_none(head)
        rest = _place_value(tail) if tail else 0
        if millions is None or rest is None or rest >= 1_000_000:
            return None
        return millions * 1_000_000 + rest
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
            if m >= last or (pending is None and i and words[i - 1] not in _DIGITS):
                return None
            total += (1 if pending is None else pending) * m
            pending, last = None, m
    return total + (pending or 0)


def _digit_or_none(words: list[str]) -> int | None:
    return _DIGITS[words[0]] if len(words) == 1 and words[0] in _DIGITS else None


def _group(words: list[str]) -> tuple[str, str] | None:
    """('digits'|'value', text) for one group of number words, None if it does not parse."""
    if words and all(w in _DIGITS for w in words):
        return "digits", "".join(str(_DIGITS[w]) for w in words)
    value = _place_value(words)
    return None if value is None else ("value", str(value))


def _read_run(words: list[str], before: str, after: str) -> str | None:
    """Digits for one run of number words and dashes, None to leave it alone."""
    groups: list[list[str]] = [[]]
    for w in words:
        if w in _DASH:
            if not groups[-1]:
                return None
            groups.append([])
        else:
            groups[-1].append(w)
    if not groups[-1]:
        return None
    parsed = [_group(g) for g in groups]
    if any(p is None for p in parsed):
        return None
    if len(groups) > 1:  # an identifier with dashes, or the place-value fallback
        digits = sum(len(p[1]) for p in parsed if p)
        all_digit_words = all(p and p[0] == "digits" for p in parsed)
        if (all_digit_words and digits >= 6) or digits >= 8 or (len(groups) >= 3 and digits >= 6):
            return "-".join(p[1] for p in parsed if p)
        return None
    kind, text = parsed[0]  # type: ignore[misc]
    if kind == "digits":
        if _UNIT_AFTER.match(after) and len(words) == 1:
            return text
        return text if len(words) >= MIN_DIGITS else None
    if _UNIT_AFTER.match(after) or _MONEY_BEFORE.search(before):
        return text
    return text if int(text) >= 1000 and _CODE_BEFORE.search(before) else None


def _split_gap_amount(words: list[str], gaps: list[bool]) -> str | None:
    """'สี่สิบเก้า ห้าร้อย' (49 and 500, พัน dropped by the ASR) before a unit: 49500."""
    if gaps.count(True) != 1:
        return None
    cut = gaps.index(True) + 1
    a = _place_value(words[:cut])
    if a is None:
        a = _digit_or_none(words[:cut])
    b = _place_value(words[cut:])
    if a is None or b is None or not (1 <= a < 100 and 100 <= b < 1000) or "ร้อย" not in words[cut:]:
        return None
    return str(a * 1000 + b)


def _thai_edits(text: str) -> list[tuple[int, int, str]]:
    found = list(_WORD.finditer(text))
    if not found:
        return []
    from pythainlp.tokenize import word_tokenize

    edges, pos = {0}, 0
    for tok in word_tokenize(text, engine="newmm", keep_whitespace=True):
        pos += len(tok)
        edges.add(pos)
    edits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(found):
        j = i
        while j + 1 < len(found):
            gap = text[found[j].end() : found[j + 1].start()]
            if gap.strip() or len(gap) > 2:
                break
            j += 1
        run = found[i : j + 1]
        i = j + 1
        # a dash at either end is an ordinary word ("ลบ" = delete), not part of the number
        while run and run[-1].group() in _DASH:
            run = run[:-1]
        while run and run[0].group() in _DASH:
            run = run[1:]
        # only whole words: "หนึ่งเดียว" is one token, so its "หนึ่ง" is not a number
        # (tokens such as "สิบสอง" hold several number words, so only the run's ends are checked)
        if not run or run[0].start() not in edges or run[-1].end() not in edges:
            continue
        words = [m.group() for m in run]
        start, end = run[0].start(), run[-1].end()
        before, after = text[:start], text[end:]
        out = _read_run(words, before, after)
        if out is None and _UNIT_AFTER.match(after):
            gaps = [run[k].end() != run[k + 1].start() for k in range(len(run) - 1)]
            out = _split_gap_amount(words, gaps)
        if out is not None:
            pad_l = " " if before and not before[-1].isspace() else ""
            pad_r = " " if after and not after[0].isspace() else ""
            edits.append((start, end, pad_l + out + pad_r))
    return edits


def spoken_numbers_to_digits(text: str) -> str:
    """Rewrite Thai (and English) number words as digits; see the module docstring for the rule."""
    text = _EN_RUN.sub(
        lambda m: "".join(_EN_VALUE[w.lower()] for w in re.findall(r"[a-z]+", m.group(), re.I)),
        text,
    )
    for start, end, out in reversed(_thai_edits(text)):
        text = text[:start] + out + text[end:]
    return text
