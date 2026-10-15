"""Fill the caller's slots ({PHONE}, {URL}, ...) with made-up values and record entity spans.

Values are made up per call from the call's seed and stay the same for the whole call, so a
caller who repeats the account number repeats the same one. Names, banks, link words and
amounts come from config/details.yaml; digits are random. Nothing comes from the model's
memory.
"""

from __future__ import annotations

import random
import re

from .config import DetailsFile
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
}
SLOT_RE = re.compile(r"\{([A-Z_]+)\}")
# A run of 5+ digits the model wrote itself instead of using a slot.
_LOOSE_DIGITS = re.compile(r"\d[\d\- ]{3,}\d")
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
        if _LOOSE_DIGITS.search(SLOT_RE.sub("", raw)):
            problems.append("number_without_slot")
        return "".join(pieces), spans, problems


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
