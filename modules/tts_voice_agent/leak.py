"""Leak check for auto-reply text. Safety-critical: false positives are fine, false negatives
are not. Anything that looks like a number, a credential word, a link, a name, an organisation,
a Latin word or data from the call is blocked."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from nlp_core.numbers import ENGLISH_DIGIT_WORDS, THAI_NUMBER_WORDS
from nlp_core.types import Entity

from .constants import MIN_DIGIT_WORD_RUN


class Reason:
    """Codes leak_reasons returns. KEYWORD, NAME_WORD, ORG_WORD and ENTITY add ":<word|label>"."""

    DIGIT = "digit"
    DIGIT_WORD_RUN = "digit_word_run"
    URL_OR_EMAIL = "url_or_email"
    KEYWORD = "keyword"
    NAME_WORD = "name_word"
    ORG_WORD = "org_word"
    LATIN_WORD = "latin_word"
    ENTITY = "entity"


_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_ZERO_WIDTH = dict.fromkeys([*range(0x200B, 0x200E), 0xFEFF, 0x2060])


# Anything that is not a Thai or Latin letter may separate spoken digits ("แปด, หนึ่ง / สอง").
_SEP = r"[^ก-๛a-z]*"
_THAI_ALT = "|".join(sorted(THAI_NUMBER_WORDS, key=len, reverse=True))
SPOKEN_THAI_RUN = re.compile(
    rf"(?:{_THAI_ALT})(?:{_SEP}(?:{_THAI_ALT})){{{MIN_DIGIT_WORD_RUN - 1},}}"
)
_EN_ALT = "|".join(ENGLISH_DIGIT_WORDS)
SPOKEN_ENGLISH_RUN = re.compile(
    rf"\b(?:{_EN_ALT})\b(?:{_SEP}\b(?:{_EN_ALT})\b){{{MIN_DIGIT_WORD_RUN - 1},}}"
)
URL_OR_EMAIL = re.compile(
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
_DIGIT_RUN = re.compile(r"\d+")
STRIP_FOR_COMPARE = re.compile(r"[\s\-.]+")


def normalize(text: str) -> str:
    """NFKC, drop zero-width/format characters, Thai digits to ASCII."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).translate(_THAI_DIGITS)
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def compact_text(text: str) -> str:
    """Normalized, lower-cased text without spaces, hyphens and dots, for matching."""
    return STRIP_FOR_COMPARE.sub("", normalize(text)).lower()


def leak_reasons(
    text: str, entities_seen: Iterable[Entity] = (), allowed_numbers: Iterable[str] = ()
) -> list[str]:
    """Why `text` is unsafe, as short codes (empty list = safe). Never includes the text.

    Any digit run is blocked unless it is in `allowed_numbers`: the numbers of facts we put in
    the prompt ourselves (today's date and time), never numbers from the call.
    """
    norm = normalize(text)
    low = norm.lower()
    compact = compact_text(text)
    reasons: list[str] = []
    allowed = set(allowed_numbers)
    if any(run not in allowed for run in _DIGIT_RUN.findall(norm)):
        reasons.append(Reason.DIGIT)
    if (
        SPOKEN_THAI_RUN.search(low)
        or SPOKEN_THAI_RUN.search(compact)
        or SPOKEN_ENGLISH_RUN.search(low)
    ):
        reasons.append(Reason.DIGIT_WORD_RUN)
    if URL_OR_EMAIL.search(low) or URL_OR_EMAIL.search(compact):
        reasons.append(Reason.URL_OR_EMAIL)
    for word in _KEYWORDS:
        if word in compact:
            reasons.append(f"{Reason.KEYWORD}:{word}")
    no_thanks = compact.replace("ขอบคุณ", "")
    for word in _NAME_WORDS:
        if word in no_thanks:
            reasons.append(f"{Reason.NAME_WORD}:{word}")
    for word in _ORG_WORDS:
        if word in compact:
            reasons.append(f"{Reason.ORG_WORD}:{word}")
    if _LATIN_WORD.search(low):
        reasons.append(Reason.LATIN_WORD)
    for ent in entities_seen:
        needle = compact_text(ent.text)
        if len(needle) >= 2 and needle in compact:
            reasons.append(f"{Reason.ENTITY}:{ent.label}")
    return reasons


def leaks_sensitive(text: str, entities_seen: Iterable[Entity] = ()) -> bool:
    """True if `text` fails any leak check."""
    return bool(leak_reasons(text, entities_seen))
