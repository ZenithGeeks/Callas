"""Leak check for auto-reply text. Safety-critical: false positives are fine, false negatives
are not. Anything that looks like a number, a credential word, a link, or data from the call
is blocked."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from nlp_core.types import Entity

from .constants import MIN_DIGIT_WORD_RUN

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ZERO_WIDTH = dict.fromkeys([*range(0x200B, 0x200E), 0xFEFF, 0x2060])

# โท is how "2" is read aloud in Thai phone numbers.
THAI_NUMBER_WORDS = ("ศูนย์ หนึ่ง สอง โท สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ ร้อย พัน หมื่น แสน ล้าน เอ็ด ยี่").split()
ENGLISH_DIGIT_WORDS = "zero one two three four five six seven eight nine oh".split()

# Anything that is not a Thai or Latin letter may separate spoken digits ("แปด, หนึ่ง / สอง").
_SEP = r"[^ก-๛a-z]*"
_THAI_ALT = "|".join(sorted(THAI_NUMBER_WORDS, key=len, reverse=True))
_THAI_RUN = re.compile(rf"(?:{_THAI_ALT})(?:{_SEP}(?:{_THAI_ALT})){{{MIN_DIGIT_WORD_RUN - 1},}}")
_EN_ALT = "|".join(ENGLISH_DIGIT_WORDS)
_EN_RUN = re.compile(rf"\b(?:{_EN_ALT})\b(?:{_SEP}\b(?:{_EN_ALT})\b){{{MIN_DIGIT_WORD_RUN - 1},}}")
_URL_EMAIL = re.compile(
    r"https?\s*:|www\s*\.|\S\s*@\s*\S|"
    r"[a-z0-9-]+\s*\.\s*(?:com|net|org|co|th|io|me|ly|xyz|info|app|link|biz|cc|in)\b"
)
_KEYWORDS = ("otp", "รหัส", "เลขบัญชี", "บัญชี", "พร้อมเพย์")
# The interim NER finds no PERSON or ORG, so a reply may not address or name anyone:
# no honorifics or name words, no organisations, no Latin words. ขอบคุณ ("thank you") is
# removed before the name check because it contains คุณ.
_NAME_WORDS = ("คุณ", "นาย", "นาง", "ชื่อ", "พี่", "น้อง", "ดร", "ผู้กอง", "จ่า", "ร้อยตำรวจ")
_ORG_WORDS = ("ธนาคาร", "ตำรวจ", "ศาล", "ไปรษณีย์", "ขนส่ง", "กรม", "บริษัท")
_LATIN_WORD = re.compile(r"[a-z]{2,}")
_STRIP_FOR_COMPARE = re.compile(r"[\s\-.]+")


def normalize(text: str) -> str:
    """NFKC, drop zero-width/format characters, Thai digits to ASCII."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_THAI_DIGITS)
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def _compact(text: str) -> str:
    return _STRIP_FOR_COMPARE.sub("", normalize(text)).lower()


def leak_reasons(text: str, entities_seen: Iterable[Entity] = ()) -> list[str]:
    """Why `text` is unsafe, as short codes (empty list = safe). Never includes the text."""
    norm = normalize(text)
    low = norm.lower()
    compact = _compact(text)
    reasons: list[str] = []
    if any(unicodedata.decimal(c, None) is not None for c in norm):
        reasons.append("digit")
    if _THAI_RUN.search(low) or _THAI_RUN.search(compact) or _EN_RUN.search(low):
        reasons.append("digit_word_run")
    if _URL_EMAIL.search(low) or _URL_EMAIL.search(compact):
        reasons.append("url_or_email")
    for word in _KEYWORDS:
        if word in compact:
            reasons.append(f"keyword:{word}")
    no_thanks = compact.replace("ขอบคุณ", "")
    for word in _NAME_WORDS:
        if word in no_thanks:
            reasons.append(f"name_word:{word}")
    for word in _ORG_WORDS:
        if word in compact:
            reasons.append(f"org_word:{word}")
    if _LATIN_WORD.search(low):
        reasons.append("latin_word")
    for ent in entities_seen:
        needle = _compact(ent.text)
        if len(needle) >= 2 and needle in compact:
            reasons.append(f"entity:{ent.label}")
    return reasons


def leaks_sensitive(text: str, entities_seen: Iterable[Entity] = ()) -> bool:
    return bool(leak_reasons(text, entities_seen))
