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


Span = tuple[int, int, str]  # start, end, label


def extract_entities(text: str) -> list[Entity]:
    """Find phone numbers, bank accounts, OTPs, URLs and money amounts in Thai text."""
    ascii_text = text.translate(_THAI_DIGITS)
    return [
        Entity(text=text[start:end], label=label, start=start, end=end)
        for start, end, label in _drop_overlaps(_find_spans(ascii_text))
    ]


def _find_spans(ascii_text: str) -> list[Span]:
    """Every regex match, with the label of the pattern that found it."""
    spans: list[Span] = []
    for match in _URL.finditer(ascii_text):
        end = match.end()
        while end > match.start() and ascii_text[end - 1] in _URL_TRAIL:
            end -= 1
        spans.append((match.start(), end, "URL"))
    for match in _PHONE.finditer(ascii_text):
        spans.append((match.start(), match.end(), "PHONE"))
    for match in _BANK_DASHED.finditer(ascii_text):
        spans.append((match.start(), match.end(), "BANK_ACCOUNT"))
    for match in _BANK_PLAIN.finditer(ascii_text):
        # a bare run of digits is an account only when a bank word comes just before it
        before = ascii_text[max(0, match.start() - 30) : match.start()]
        if _BANK_CONTEXT.search(before):
            spans.append((match.start(), match.end(), "BANK_ACCOUNT"))
    for match in _OTP.finditer(ascii_text):
        spans.append((match.start(1), match.end(1), "OTP"))
    for match in _MONEY.finditer(ascii_text):
        spans.append((match.start(), match.end(), "MONEY"))
    return spans


def _drop_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the longest span, then the higher-priority label, and drop what overlaps it."""
    strongest_first = sorted(
        spans, key=lambda span: (-(span[1] - span[0]), -_PRIORITY[span[2]], span[0])
    )
    kept: list[Span] = []
    for start, end, label in strongest_first:
        if all(end <= kept_start or start >= kept_end for kept_start, kept_end, _ in kept):
            kept.append((start, end, label))
    return sorted(kept)
