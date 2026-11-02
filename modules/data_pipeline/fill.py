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
    "LINE_ID": "LINE_ID",
    "CARD_NUMBER": "CARD_NUMBER",
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
# turns.receiver_view) so it recognises its own details.
RECEIVER_OWN_SLOTS = (
    "RECEIVER_NAME",
    "RECEIVER_PHONE",
    "RECEIVER_ACCOUNT",
    "ID_CARD",
    "BIRTH_DATE",
    "ADDRESS",
)
# The caller's numbers and links. The receiver hears them as slots too, so a receiver that
# repeats one writes the slot, not digits that get the call rejected as number_without_slot
# (#99). Repeating one of these is not a leak: only RECEIVER_LEAK_SLOTS are.
CALLER_NUMBER_SLOTS = ("PHONE", "BANK_ACCOUNT", "OTP", "URL", "MONEY", "LINE_ID", "CARD_NUMBER")
# Slots a caller cannot know: the person who answered owns these. A caller turn that uses one
# is flagged (turns.write_turn). {RECEIVER_NAME} is not here: a caller may know the name.
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
# A LINE ID the model made up instead of writing {LINE_ID} (#143): in a line that names LINE, a
# Latin word of 5 or more characters that is all lowercase or has a digit or "_" (viphost01,
# landdoc, HomeProRefund2025). A name such as "PEA Digital Meter Support" is not caught.
_NAMES_LINE = re.compile(r"ไลน์|ไอดี|LINE", re.IGNORECASE)
_MADE_UP_ID = re.compile(
    r"(?<![A-Za-z0-9_.])(?=[A-Za-z0-9_.]{5})"
    r"(?:[a-z][a-z.]*|[A-Za-z][A-Za-z0-9_.]*[0-9_][A-Za-z0-9_.]*)"
    r"(?![A-Za-z0-9_.])"
)
_BAHT_AFTER = re.compile(r"^\s*บาท")
# "บัตรที่ลงท้ายด้วย {CARD_NUMBER}": only the last 4 digits are said after ลงท้าย (#160)
ENDS_WITH = r"ลงท้าย(?:ด้วย)?\s*(?:เลข(?:ที่)?)?\s*"
_ENDS_WITH_BEFORE = re.compile(ENDS_WITH + "$")
BANK_SCENARIO_ORG = "ธนาคาร"  # scenarios.yaml org value that means "pick a bank from details"
PODCAST_SCENARIO_ORG = "พอดแคสต์"  # scenarios.yaml org value: "pick a podcast from details"
SCHOOL_SCENARIO_ORG = "โรงเรียน"  # scenarios.yaml org value: "pick a school from details"
NICKNAME_RELATIONS = ("friend", "family")  # relations whose two names come from details.nicknames


class SlotFiller:
    """The made-up values of one call, one for each slot, and `fill` to put them in a line."""

    def __init__(self, call: CallConfig, details: DetailsFile) -> None:
        rng = random.Random(f"fill:{call.call_type}:{call.seed}")
        nickname = call.scenario.relation in NICKNAME_RELATIONS and details.nicknames
        names = details.nicknames if nickname else details.names
        caller = rng.choice(names[call.caller_gender])
        other_names = [n for n in names[call.receiver.gender] if n != caller]
        receiver = rng.choice(other_names)
        org = call.scenario.org
        if org == BANK_SCENARIO_ORG:
            org = rng.choice(details.banks)
        elif org == PODCAST_SCENARIO_ORG:
            org = rng.choice(details.podcasts or details.made_up_orgs)
        elif org == SCHOOL_SCENARIO_ORG:
            org = rng.choice(details.schools or details.made_up_orgs)
        elif org in ("*", ""):
            org = rng.choice(details.made_up_orgs)
        money = details.money
        amounts = money.big if call.call_type in money.big_types else money.small
        if call.scenario.money_max is not None:
            amounts = [a for a in amounts if a <= call.scenario.money_max]
        amount = rng.choice(amounts)
        # The order of every rng draw in this method, above and below, fixes each value for a
        # given seed; do not reorder.
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
            # added after the receiver's details (#143) so every value above stays what it was
            "LINE_ID": _line_id(rng, details.line_ids),
            # the card a caller says the receiver has (#160), last for the same reason
            "CARD_NUMBER": _card_number(rng),
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
            if name == "CARD_NUMBER" and _ENDS_WITH_BEFORE.search(raw[: match.start()]):
                value = value[-4:]
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
        text = "".join(pieces)
        return text, spans, problems + _line_problems(raw, text)


def _line_problems(raw: str, text: str) -> list[str]:
    """What is wrong with the line itself, as the model wrote it (`raw`) and as filled (`text`)."""
    problems = []
    if _LOOSE_DIGITS.search(SLOT_RE.sub("|", raw)):  # "|" keeps digits next to a slot apart
        problems.append("number_without_slot")
    if "{" in text or "}" in text:  # {otp}, {OTP2}, {{OTP}}: a slot that did not fill
        problems.append("stray_brace")
    if "\n" in raw or "\r" in raw or _LINE_START.match(raw.lstrip()):
        problems.append("line_break")  # a line is one turn: no second turn, no speaker tag
    if _NAMES_LINE.search(raw) and _MADE_UP_ID.search(SLOT_RE.sub("|", raw)):
        problems.append("line_id_without_slot")
    return problems


def receiver_leaks(raw: str) -> list[str]:
    """Labels of the receiver's own details that a receiver line gives away, in the order they
    first appear. Read from the slots the line uses, so the model never reports them."""
    labels: list[str] = []
    for name in SLOT_RE.findall(raw):
        label = RECEIVER_LEAK_SLOTS.get(name)
        if label and label not in labels:
            labels.append(label)
    return labels


# Gender markers that belong to the other gender. A female speaker using these male markers, or
# a male speaker using these female ones, is a slip worth flagging. ค่า ("value/fee") and คะ
# inside a word (คะแนน) are deliberately not here; bare คะ is only a particle sentence-finally.
# ผม also means "hair", so this stays a soft flag (a reviewer decides), never an auto-reject.
_MALE_MARKERS = ("ครับ", "คับ", "กระผม", "ผม")
_FEMALE_MARKERS = ("ค่ะ", "ดิฉัน")
_KHA_FINAL = re.compile(r"คะ(?=[\s!?.,…]|$)")  # bare คะ as a sentence-final particle, not คะแนน


def wrong_particles(raw: str, gender: str) -> list[str]:
    """Gender markers in a receiver line that do not match the speaker's gender, in order.

    Empty when the line is consistent or the gender is unknown. A soft check: the result is
    recorded as a turn problem (turns.write_turn), it never rewrites the line.
    """
    found: list[str] = []
    if gender == "female":
        found = [m for m in _MALE_MARKERS if m in raw]
    elif gender == "male":
        found = [m for m in _FEMALE_MARKERS if m in raw]
        if _KHA_FINAL.search(raw):
            found.append("คะ")
    # drop a marker already covered by a longer one that contains it (กระผม contains ผม)
    return [m for m in found if not any(m != other and m in other for other in found)]


# Leading interjections the model keeps opening with (โอ้โห, โธ่เอ้ย, ...). Longest first so a
# longer opener (โธ่เอ้ย) is matched before the shorter one inside it (โธ่), and the whole opener
# is dropped, not just its first word.
_LEADING_INTERJECTIONS = ("โธ่เอ้ย", "โอ้โห", "โอ้โฮ", "โหย", "โธ่")
# An opener counts only when the line ends or a non-Thai character (space, ., …, !) follows it,
# never when a Thai letter continues it: โหย is an opener, โหยหา ("to long for") is one real word,
# so the opener must not be chopped off the front of a longer word (U+0E00-U+0E7F is Thai).
_AFTER_OPENER = r"(?=$|[^฀-๿])"


def strip_leading_interjection(raw: str) -> str:
    """Drop a single over-acted interjection at the start of a receiver line (and the filler right
    after it), so the line opens plainly. Leaves the rest untouched; never empties the line."""
    s = raw.lstrip()
    for word in _LEADING_INTERJECTIONS:
        if re.match(re.escape(word) + _AFTER_OPENER, s):
            rest = s[len(word) :].lstrip(" .…,!~ๆ")
            return rest or raw
    return raw


# A receiver asking who or what it itself is (ฉันคือใคร "who am I", ผมชื่ออะไร "what is my name")
# is a broken line: a person knows their own name. Only ฉัน/ผม/ดิฉัน count — หนู and เรา are left
# out because a Thai speaker also uses them for "you" (an older receiver asks a young caller
# "หนูชื่ออะไรจ๊ะ" = what is YOUR name). Matches a self pronoun straight before คือ/เป็นใคร or
# ชื่ออะไร, so "ฉันไม่รู้ว่าคุณคือใคร" and "ฉันเป็นอะไรไป" ("what is wrong with me") are not caught.
_SELF_IDENTITY_QUESTION = re.compile(r"(?:ฉัน|ผม|ดิฉัน)(?:(?:คือ|เป็น)ใคร|ชื่ออะไร)")


def self_identity_question(raw: str) -> bool:
    """True if the receiver asks about its own identity — never sensible for the person who
    answered. Recorded as a turn problem, which rejects the broken call (turns.write_turn)."""
    return bool(_SELF_IDENTITY_QUESTION.search(raw.replace(" ", "")))


# made-up values


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


def _line_id(rng: random.Random, parts: list[str]) -> str:
    """Two different parts joined, e.g. "viphost": lowercase letters, no digits."""
    first, second = rng.sample(parts, 2)
    return first + second


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


def _card_number(rng: random.Random) -> str:
    """A 16-digit card number in the usual 1234-5678-9012-3456 layout. The digits are random and
    the last digit is never the Luhn check digit, so this is not a valid card number. About one
    draw in ten would pass the check by chance; those get a different last digit."""
    body = f"{rng.randint(0, 10**15 - 1):015d}"
    last = rng.randint(0, 9)
    total = sum(sum(divmod(int(d) * 2, 10)) if i % 2 == 0 else int(d) for i, d in enumerate(body))
    check = (10 - total % 10) % 10
    if last == check:  # an extra draw only in this case, so other seeds keep their values
        last = (check + 1 + rng.randrange(9)) % 10
    digits = f"{body}{last}"
    return "-".join(digits[i : i + 4] for i in range(0, 16, 4))


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
