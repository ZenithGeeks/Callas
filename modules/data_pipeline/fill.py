"""Fill the caller's slots ({PHONE}, {URL}, ...) with made-up values and record entity spans.

Values are made up per call from the call's seed and stay the same for the whole call, so a
caller who repeats the account number repeats the same one. Names, banks, link words and
amounts come from config/details.yaml; digits are random. Nothing comes from the model's
memory.
"""

from __future__ import annotations

import random
import re
from datetime import date

from .config import AddressParts, DetailsFile
from .scenarios import CallConfig

SLOT_LABEL = {
    "CALLER_NAME": "PERSON",
    "RECEIVER_NAME": "PERSON",
    "ORG": "ORG",
    "PHONE": "PHONE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "OTP": "OTP",
    "URL": "URL",
    "MONEY": "MONEY",
    "RECEIVER_PHONE": "PHONE",
    "RECEIVER_ACCOUNT": "BANK_ACCOUNT",
    "ID_CARD": "ID_CARD",
    "BIRTH_DATE": "BIRTH_DATE",
    "ADDRESS": "ADDRESS",
}
# The receiver's own details. A receiver turn that writes one of these slots has given that
# detail away; the label is what the leak is recorded as. {RECEIVER_NAME} is not here: saying
# your own name when you answer the phone is not a leak.
RECEIVER_LEAK_SLOTS = {
    "OTP": "OTP",
    "RECEIVER_PHONE": "PHONE",
    "RECEIVER_ACCOUNT": "BANK_ACCOUNT",
    "ID_CARD": "ID_CARD",
    "BIRTH_DATE": "BIRTH_DATE",
    "ADDRESS": "ADDRESS",
}
# Slots whose value belongs to the receiver. The receiver reads them back as slots (see
# simulate._receiver_view) so it recognises its own details.
RECEIVER_OWN_SLOTS = (
    "RECEIVER_NAME",
    "RECEIVER_PHONE",
    "RECEIVER_ACCOUNT",
    "ID_CARD",
    "BIRTH_DATE",
    "ADDRESS",
)
# Slots a caller cannot know: the person who answered owns these. A caller turn that uses one
# is flagged (simulate._next_turn). {RECEIVER_NAME} is not here: a caller may know the name.
RECEIVER_ONLY_SLOTS = ("RECEIVER_PHONE", "RECEIVER_ACCOUNT", "ID_CARD", "BIRTH_DATE", "ADDRESS")
# The date ages are counted at. Fixed, so a seed always gives the same birth date.
AGE_REFERENCE_DATE = date(2025, 7, 1)
THAI_MONTHS = (
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
)
SLOT_RE = re.compile(r"\{([A-Z_]+)\}")
# A number the model wrote itself instead of using a slot. The rule: 4 or more digits in a row
# ("1234", "0812345678"), or digit groups joined by one , . - or space where each later group
# has 3-4 digits, the way thousands, phone and account numbers are grouped ("50,000",
# "5 000", "081.234.5678"). Small numbers ("3", "15") and times ("10.30": the second group has
# 2 digits) are fine. A year such as 2568 counts as a bare number too.
_LOOSE_DIGITS = re.compile(r"\d{4}|\d+(?:[,. -]\d{3,4})+")
_LINE_START = re.compile(r"[AB]\s*:")  # a speaker tag the model copied into its line
_BAHT_AFTER = re.compile(r"^\s*บาท")
BANK_SCENARIO_ORG = "ธนาคาร"  # scenarios.yaml org value that means "pick a bank from details"


class SlotFiller:
    def __init__(self, call: CallConfig, details: DetailsFile) -> None:
        rng = random.Random(f"fill:{call.call_type}:{call.seed}")
        caller = rng.choice(details.names[call.caller_gender])
        other_names = [n for n in details.names[call.receiver.gender] if n != caller]
        receiver = rng.choice(other_names)
        org = call.scenario.org
        if org == BANK_SCENARIO_ORG:
            org = rng.choice(details.banks)
        elif org in ("*", ""):
            org = rng.choice(details.made_up_orgs)
        money = details.money
        amount = rng.choice(money.big if call.call_type in money.big_types else money.small)
        # The order of rng draws below fixes each value for a given seed; do not reorder.
        self.values = {
            "CALLER_NAME": caller,
            "RECEIVER_NAME": receiver,
            "ORG": org,
            "PHONE": _phone(rng),
            "BANK_ACCOUNT": _bank_account(rng),
            "OTP": f"{rng.randint(0, 999999):06d}",
            "URL": _url(rng, details.urls.words, details.urls.tlds),
            "MONEY": f"{amount:,} บาท",
            # the receiver's own details are drawn last so the values above stay what they were
            "RECEIVER_PHONE": _phone(rng),
            "RECEIVER_ACCOUNT": _bank_account(rng),
            "ID_CARD": _id_card(rng),
            "BIRTH_DATE": _birth_date(rng, call.receiver.age),
            "ADDRESS": _address(rng, details.addresses),
        }

    def fill(self, raw: str) -> tuple[str, list[dict], list[str]]:
        """raw text with slots -> (text, entity spans within text, problems)."""
        pieces: list[str] = []
        spans: list[dict] = []
        problems: list[str] = []
        raw_pos = 0  # how much of raw has been consumed
        text_len = 0  # length of the filled text built so far
        for match in SLOT_RE.finditer(raw):
            before = raw[raw_pos : match.start()]
            pieces.append(before)
            text_len += len(before)
            name = match.group(1)
            raw_pos = match.end()
            if name not in self.values:
                problems.append(f"unknown_slot:{name}")
                pieces.append(match.group(0))
                text_len += len(match.group(0))
                continue
            value = self.values[name]
            if name == "MONEY":  # the model often writes "{MONEY} บาท"; keep one unit
                baht = _BAHT_AFTER.match(raw[raw_pos:])
                if baht:
                    raw_pos += baht.end()
            pieces.append(value)
            spans.append(
                {
                    "text": value,
                    "label": SLOT_LABEL[name],
                    "start": text_len,
                    "end": text_len + len(value),
                }
            )
            text_len += len(value)
        pieces.append(raw[raw_pos:])
        if _LOOSE_DIGITS.search(SLOT_RE.sub("|", raw)):  # "|" keeps digits next to a slot apart
            problems.append("number_without_slot")
        text = "".join(pieces)
        if "{" in text or "}" in text:  # {otp}, {OTP2}, {{OTP}}: a slot that did not fill
            problems.append("stray_brace")
        if "\n" in raw or "\r" in raw or _LINE_START.match(raw.lstrip()):
            problems.append("line_break")  # a line is one turn: no second turn, no speaker tag
        return text, spans, problems


def receiver_leaks(raw: str) -> list[str]:
    """Labels of the receiver's own details that a receiver line gives away, in the order they
    first appear. Read from the slots the line uses, so the model never reports them."""
    labels: list[str] = []
    for name in SLOT_RE.findall(raw):
        label = RECEIVER_LEAK_SLOTS.get(name)
        if label and label not in labels:
            labels.append(label)
    return labels


# ---- made-up values --------------------------------------------------------------------------


def _phone(rng: random.Random) -> str:
    """Thai mobile style: 06x, 08x or 09x, then 3 and 4 digits."""
    prefix = f"0{rng.choice('689')}{rng.randint(0, 9)}"
    return f"{prefix}-{rng.randint(0, 999):03d}-{rng.randint(0, 9999):04d}"


def _bank_account(rng: random.Random) -> str:
    first, second = rng.randint(100, 999), rng.randint(0, 9)
    middle, last = rng.randint(10000, 99999), rng.randint(0, 9)
    return f"{first}-{second}-{middle}-{last}"


def _url(rng: random.Random, words: list[str], tlds: list[str]) -> str:
    first, second = rng.choice(words), rng.choice(words)
    return f"https://{first}-{second}{rng.randint(10, 99)}.{rng.choice(tlds)}"


def _id_card(rng: random.Random) -> str:
    """A Thai ID card number in the usual 1-2345-67890-12-3 layout. The digits are random and
    the last digit is never the real check digit (mod 11), so this is not a valid ID number.
    About one draw in ten would pass the check by chance; those get a different last digit."""
    first = rng.randint(1, 8)
    middle = f"{rng.randint(0, 9999):04d}{rng.randint(0, 99999):05d}{rng.randint(0, 99):02d}"
    last = rng.randint(0, 9)
    digits = f"{first}{middle}"
    check = (11 - sum(int(d) * (13 - i) for i, d in enumerate(digits)) % 11) % 10
    if last == check:  # an extra draw only in this case, so other seeds keep their values
        last = (check + 1 + rng.randrange(9)) % 10
    return f"{first}-{middle[:4]}-{middle[4:9]}-{middle[9:]}-{last}"


def _birth_date(rng: random.Random, age: int) -> str:
    """Day, month and Buddhist-era year, e.g. "12 มีนาคม 2511", such that the person is exactly
    `age` years old on AGE_REFERENCE_DATE."""
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    ref = AGE_REFERENCE_DATE
    had_birthday = (month, day) <= (ref.month, ref.day)
    year = ref.year - age - (0 if had_birthday else 1)
    return f"{day} {THAI_MONTHS[month - 1]} {year + 543}"


def _address(rng: random.Random, parts: AddressParts) -> str:
    house, soi = rng.choice(parts.house_numbers), rng.choice(parts.sois)
    return f"{house} ซอย{soi} {rng.choice(parts.districts)} {rng.choice(parts.provinces)}"
