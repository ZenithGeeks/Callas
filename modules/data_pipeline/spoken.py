"""How a Thai speaker says a turn's numbers, for the text that goes to the voice.

VachanaTTS reads "158-5-96045-2" as one big number with minus signs. A person reads an account,
phone or ID card number digit by digit, and a code digit by digit; money is read with place
value (49,500 baht is สี่หมื่นเก้าพันห้าร้อยบาท). The stored `text` and its entity spans keep the
digits, because the digits are the labels. Only the string given to the voice changes: it is
built from the turn's own entity spans and kept on the turn as `spoken_text`.

Between the digit groups of a number people either say "ขีด" (dash) or only pause. One style is
chosen per call from its seed (`speech.say_dash_share` in agents.yaml) and used for the whole
call, so a caller does not switch halfway.
"""

from __future__ import annotations

import random
import re

from pythainlp.util import digit_to_text, num_to_thaiword

DASH, PAUSE = "dash", "pause"
DASH_WORD = "ขีด"
# Measured with VachanaTTS (th_f_1 / th_m_1, the same 4 digit groups): joined by a space the audio
# was 3.41 s / 3.17 s, with no separator 3.29 s / 3.11 s, so a space is almost no pause; joined
# by ", " it was 3.55 s / 3.80 s, a real pause (about 0.1 s and 0.2 s per gap). "." and "..."
# were unreliable (the male voice took 5.4 s and 7.4 s).
PAUSE_JOINER = ", "

_DIGIT_GROUPS = ("PHONE", "BANK_ACCOUNT", "ID_CARD")


def pick_speech_style(call_type: str, seed: int, say_dash_share: float) -> str:
    """ "dash" or "pause" for the whole call; the same call type and seed give the same style."""
    rng = random.Random(f"speech:{call_type}:{seed}")
    return DASH if rng.random() < say_dash_share else PAUSE


def _digit_groups(value: str, style: str) -> str:
    """158-5-96045-2 -> หนึ่งห้าแปด ขีด ห้า ขีด ... In the pause style the groups are joined by
    ", ": VachanaTTS barely pauses on a plain space (see PAUSE_JOINER)."""
    joiner = f" {DASH_WORD} " if style == DASH else PAUSE_JOINER
    parts = []
    for token in re.split(r"([-\s]+)", value):
        if not token:
            continue
        if re.fullmatch(r"[-\s]+", token):
            parts.append(joiner if "-" in token or style == PAUSE else " ")
        elif token.isdigit():
            parts.append(digit_to_text(token))
        else:
            parts.append(token)
    return re.sub(r" {2,}", " ", "".join(parts)).strip()


def _money(value: str) -> str:
    match = re.search(r"\d[\d,]*", value)
    if not match:
        return value
    return num_to_thaiword(int(match.group().replace(",", ""))) + "บาท"


def _birth_date(value: str) -> str:
    day, month, year = value.split()  # e.g. "12 มีนาคม 2511"
    return f"{num_to_thaiword(int(day))} {month} {num_to_thaiword(int(year))}"


def _address(value: str) -> str:
    """House and soi numbers by place value ("45" is สี่สิบห้า, "12/3" is สิบสองทับสาม). A
    5-digit number is a postcode and is read digit by digit."""

    def number(match: re.Match) -> str:
        digits = match.group()
        if len(digits) == 5:
            return digit_to_text(digits)
        return num_to_thaiword(int(digits))

    return re.sub(r"\d+", number, value).replace("/", "ทับ")


def say(label: str, value: str, style: str) -> str:
    """How one entity is said. Anything not listed is said as written."""
    try:
        if label in _DIGIT_GROUPS:
            return _digit_groups(value, style)
        if label == "OTP":
            return digit_to_text(re.sub(r"\D", "", value))
        if label == "MONEY":
            return _money(value)
        if label == "BIRTH_DATE":
            return _birth_date(value)
        if label == "ADDRESS":
            return _address(value)
    except ValueError:  # a value in an unexpected shape is better said as written than lost
        return value
    return value


def spoken_text(text: str, entities: list[dict], style: str) -> str:
    """`text` with each entity span replaced by how it is said. Text outside the spans is kept
    exactly as it is."""
    pieces, pos = [], 0
    for entity in sorted(entities, key=lambda e: e["start"]):
        if entity["start"] < pos or text[entity["start"] : entity["end"]] != entity["text"]:
            continue  # overlapping or stale span: leave that part as written
        pieces.append(text[pos : entity["start"]])
        pieces.append(say(entity["label"], entity["text"], style))
        pos = entity["end"]
    pieces.append(text[pos:])
    return "".join(pieces)
