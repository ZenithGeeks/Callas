"""Rule-based Thai entity extraction (interim, until a trained NER exists).

Thai digits are mapped 1:1 to ASCII digits before matching, so offsets stay valid
for the original text. Overlapping matches are resolved: longest span wins, then
the more specific label.
"""

from __future__ import annotations

import re

from .types import Entity

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

_GAP = r"[\s-]?"
_PHONE = re.compile(
    r"(?<![\d+])(?:"
    rf"\+66{_GAP}[689]\d{_GAP}\d{{3}}{_GAP}\d{{4}}"
    rf"|\+66{_GAP}[2-7]\d?{_GAP}\d{{3}}{_GAP}\d{{3,4}}"
    rf"|0[689]\d{_GAP}\d{{3}}{_GAP}\d{{4}}"
    rf"|02{_GAP}\d{{3}}{_GAP}\d{{4}}"
    rf"|0[3-7]\d{_GAP}\d{{3}}{_GAP}\d{{3}}"
    r")(?!\d)"
)
_BANK_DASHED = re.compile(r"(?<!\d)\d{3}-\d-\d{5}-\d(?!\d)|(?<!\d)\d{4}-\d{2}-\d{4,6}(?!\d)")
_BANK_PLAIN = re.compile(r"(?<!\d)\d(?:[- ]?\d){9,11}(?!\d)")
_BANK_CONTEXT = re.compile(r"บัญชี|เลขบัญชี|เลขที่บัญชี|พร้อมเพย์|account", re.I)
_OTP = re.compile(
    r"(?:OTP|รหัส|โค้ด|code)[^\d\n]{0,25}?(?<!\d)(\d{4,8})(?!\d)",
    re.I,
)
_URL = re.compile(
    r"(?:https?://|www\.)[^\s<>\"'฀-๿]+"
    r"|(?<![\w@.])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|co\.th|in\.th|go\.th|or\.th|th"
    r"|me|ly|link|xyz|top|info|cc|vip)(?![\w.])(?:/[^\s<>\"'฀-๿]*)?",
    re.I,
)
_MONEY = re.compile(r"(?<![\d,.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*บาท")

_URL_TRAIL = ".,;:!?)]}"

# Higher wins when two spans have equal length.
_PRIORITY = {"BANK_ACCOUNT": 4, "OTP": 3, "PHONE": 2, "URL": 2, "MONEY": 1}


def extract_entities(text: str) -> list[Entity]:
    ascii_text = text.translate(_THAI_DIGITS)
    candidates: list[tuple[int, int, str]] = []

    for m in _URL.finditer(ascii_text):
        end = m.end()
        while end > m.start() and ascii_text[end - 1] in _URL_TRAIL:
            end -= 1
        candidates.append((m.start(), end, "URL"))
    for m in _PHONE.finditer(ascii_text):
        candidates.append((m.start(), m.end(), "PHONE"))
    for m in _BANK_DASHED.finditer(ascii_text):
        candidates.append((m.start(), m.end(), "BANK_ACCOUNT"))
    for m in _BANK_PLAIN.finditer(ascii_text):
        before = ascii_text[max(0, m.start() - 30) : m.start()]
        if _BANK_CONTEXT.search(before):
            candidates.append((m.start(), m.end(), "BANK_ACCOUNT"))
    for m in _OTP.finditer(ascii_text):
        candidates.append((m.start(1), m.end(1), "OTP"))
    for m in _MONEY.finditer(ascii_text):
        candidates.append((m.start(), m.end(), "MONEY"))

    candidates.sort(key=lambda c: (-(c[1] - c[0]), -_PRIORITY[c[2]], c[0]))
    chosen: list[tuple[int, int, str]] = []
    for start, end, label in candidates:
        if all(end <= other_start or start >= other_end for other_start, other_end, _ in chosen):
            chosen.append((start, end, label))
    chosen.sort()
    return [
        Entity(text=text[start:end], label=label, start=start, end=end)
        for start, end, label in chosen
    ]
