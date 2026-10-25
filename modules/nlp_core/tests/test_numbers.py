import re

import pytest

from asr_diarization.types import Turn
from nlp_core import numbers
from nlp_core.entities import extract_entities
from nlp_core.numbers import spoken_numbers_to_digits
from tts_voice_agent.leak import leak_reasons
from tts_voice_agent.memory import clean_turn

# (spoken, expected)
CONVERTED = [
    # phone numbers read digit by digit, with and without spaces
    ("โทรมาที่ ศูนย์ แปด หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด นะ", "โทรมาที่ 0812345678 นะ"),
    ("เบอร์ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปดครับ", "เบอร์ 0812345678 ครับ"),
    ("ศูนย์แปดเก้า หนึ่งสองสาม สี่ห้าหกเจ็ด", "0891234567"),
    ("เบอร์ ศูนย์ สอง ห้า หนึ่ง หนึ่ง หนึ่ง หนึ่ง หนึ่ง หนึ่ง หนึ่ง", "เบอร์ 0251111111"),
    ("ศูนย์หกโทสามสี่ห้าหกเจ็ดแปดเก้า", "0623456789"),  # โท is 2
    ("ศูนย์แปดหนึ่งขีดสองสามสี่ขีดห้าหกเจ็ดแปด", "081-234-5678"),
    # accounts: with ขีด, with pauses, run together
    ("บัญชี หนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสอง ครับ", "บัญชี 158-5-96045-2 ครับ"),
    ("โอนเข้าบัญชีหนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสองเลย", "โอนเข้าบัญชี 158-5-96045-2 เลย"),
    ("บัญชี หนึ่ง ห้า แปด ขีด ห้า ขีด เก้า หก ศูนย์ สี่ ห้า ขีด สอง", "บัญชี 158-5-96045-2"),
    ("บัญชีเจ็ดสี่สอง ขีด แปด ขีด เจ็ดแปดหนึ่งแปดหนึ่ง ขีด เก้า", "บัญชี 742-8-78181-9"),
    ("บัญชีหนึ่งห้าแปด ห้า เก้าหกศูนย์สี่ห้า สองครับ", "บัญชี 1585960452 ครับ"),
    ("บัญชีหนึ่งห้าแปดห้าเก้าหกศูนย์สี่ห้าสอง", "บัญชี 1585960452"),
    ("เลขบัญชี หนึ่งร้อยห้าสิบแปด ลบ ห้า ลบ เก้าหมื่นหกพันสี่สิบห้า ลบ สอง", "เลขบัญชี 158-5-96045-2"),
    # ID card, OTP
    ("บัตรหนึ่งขีดสองสามสี่ห้าขีดหกเจ็ดแปดเก้าศูนย์ขีดหนึ่งสองขีดสาม", "บัตร 1-2345-67890-12-3"),
    ("รหัสโอทีพีคือ ห้า หนึ่ง เจ็ด แปด สี่ หนึ่ง ค่ะ", "รหัสโอทีพีคือ 517841 ค่ะ"),
    ("รหัสคือห้าหนึ่งเจ็ดแปดสี่หนึ่งค่ะ", "รหัสคือ 517841 ค่ะ"),
    ("OTP ศูนย์เก้าแปดเจ็ด นะ", "OTP 0987 นะ"),
    ("รหัสคือห้าแสนหนึ่งหมื่นเจ็ดพันแปดร้อยสี่สิบเอ็ดค่ะ", "รหัสคือ 517841 ค่ะ"),
    # money
    ("ห้าหมื่นบาท", "50000 บาท"),
    ("โอนหนึ่งร้อยห้าสิบแปดบาท", "โอน 158 บาท"),
    ("ต้องโอนสองพันห้าร้อยบาทครับ", "ต้องโอน 2500 บาทครับ"),
    ("ยอดหนึ่งแสนสองหมื่นบาท", "ยอด 120000 บาท"),
    ("จ่ายยี่สิบเอ็ดบาท", "จ่าย 21 บาท"),
    ("สองล้านห้าแสนบาท", "2500000 บาท"),
    ("สี่หมื่นเก้าพันห้าร้อยบาท", "49500 บาท"),
    # a bare multiplier counts as 1
    ("ล้านบาท", "1000000 บาท"),
    ("สิบล้านบาท", "10000000 บาท"),
    ("ร้อยล้านบาท", "100000000 บาท"),
    ("หมื่นล้านบาท", "10000000000 บาท"),
    ("เงินสิบล้าน", "เงิน 10000000"),
    ("หนึ่งล้านหนึ่งบาท", "1000001 บาท"),
    ("หนึ่งร้อยสิบเอ็ดบาท", "111 บาท"),
    ("สามร้อยสิบบาท", "310 บาท"),
    ("ร้อยสิบเอ็ดบาท", "111 บาท"),
    ("ค่าธรรมเนียมห้าบาท", "ค่าธรรมเนียม 5 บาท"),
    ("ยอดค้างชำระสามหมื่นห้าพัน", "ยอดค้างชำระสามหมื่นห้าพัน"),  # money word is not right before
    ("ยอดสามหมื่นห้าพัน", "ยอด 35000"),
    # mixed sentence
    (
        "โอนห้าพันบาทเข้าบัญชีหนึ่งสองสามขีดสี่ขีดห้าหกเจ็ดแปดเก้าขีดศูนย์ แล้วบอกรหัสหนึ่งสองสามสี่หกห้า",
        "โอน 5000 บาทเข้าบัญชี 123-4-56789-0 แล้วบอกรหัส 123465",
    ),
    ("call me on zero eight one two three four five six seven eight", "call me on 0812345678"),
]

# ordinary words and counts that must not change
UNCHANGED = [
    "หนึ่งเดียวในใจ",
    "สองคนนั้นมาที่นี่",
    "ร้อยละห้าสิบของคน",
    "ร้อยตำรวจเอกสามารถช่วยได้",
    "สามีผมอยู่ที่ห้างสรรพสินค้า",
    "สามสิบคนเข้าร่วมงาน",
    "รอสิบวันเท่านั้น",
    "สิบเอกพันธุ์ศักดิ์โทรมา",
    "จบปริญญาโทแล้วครับ",
    "ไปสี่แยกสามสี่วัน",
    "ลบข้อความแล้วนะ",
    "ห้าลบสองเท่ากับสาม",
    "สองร้อยลบสามสิบ",
    "เก้าอี้ห้าตัว",
    "ซื้อของห้าหกเจ็ดแปดเก้าอี้ให้เขา",  # a long run that ends inside the word เก้าอี้
    # two numbers split by a space are never merged into one amount
    "ผมมีลูกสอง สามร้อยบาท",
    "มีคนสามสิบ ห้าร้อยบาท",
    "โอนยี่สิบ สามร้อยบาท",
    "ภาษีสี่สิบเก้า ห้าร้อยบาท ถึงจะส่งได้",
    # a money word before a head count or a year
    "จำนวนสามสิบคน",
    "โอนเงินสามสิบห้าคน",
    "ค่าสองร้อยคน",
    "ค่าสองพันห้าร้อยหกสิบเจ็ดปี",
    "ยอดสิบวัน",
    "ราคาสองร้อยชิ้น",
    # a range written with a hyphen converts neither half
    "ห้าสิบ-หกสิบบาท",
    "โอนห้าสิบ - หกสิบบาท",
    "สอง-สามบาท",
    # เอ็ด is 1 only after สิบ or in a digit run
    "เอ็ดบาท",
    "I have one or two things",
    "สวัสดีครับ ผมโทรมาจากธนาคาร",
]


@pytest.mark.parametrize(("spoken", "expected"), CONVERTED)
def test_spoken_numbers_become_digits(spoken, expected):
    assert spoken_numbers_to_digits(spoken) == expected


@pytest.mark.parametrize("text", UNCHANGED)
def test_ordinary_words_stay(text):
    assert spoken_numbers_to_digits(text) == text


def _reference_edit_for_run(text, run, word_edges):
    """The first version of numbers._edit_for_run: it slices the whole text before and after."""
    run = numbers._trim_dashes(run)
    if not run or run[0].start() not in word_edges or run[-1].end() not in word_edges:
        return None
    words = [m.group() for m in run]
    start, end = run[0].start(), run[-1].end()
    before, after = text[:start], text[end:]
    digits = numbers._read_run(words, before, after)
    if digits is None:
        return None
    long_digits = len(words) >= numbers.MIN_DIGITS and all(
        w in numbers._DIGITS or w in numbers._DASH_WORDS for w in words
    )
    next_to_hyphen = before.rstrip().endswith("-") or after.lstrip().startswith("-")
    if next_to_hyphen and not long_digits:
        return None
    pad_left = " " if before and not before[-1].isspace() else ""
    pad_right = " " if after and not after[0].isspace() else ""
    return start, end, pad_left + digits + pad_right


def _reference_spoken(text):
    """The first version of spoken_numbers_to_digits: one slice of the whole text per edit."""
    text = numbers._EN_RUN.sub(
        lambda m: "".join(
            numbers._EN_VALUE[w.lower()] for w in re.findall(r"[a-z]+", m.group(), re.I)
        ),
        text,
    )
    matches = list(numbers._NUMBER_WORD.finditer(text))
    if not matches:
        return text
    word_edges = numbers._token_edges(text)
    edits = [_reference_edit_for_run(text, r, word_edges) for r in numbers._runs(matches, text)]
    for start, end, out in reversed([e for e in edits if e is not None]):
        text = text[:start] + out + text[end:]
    return text


# Blanks, hyphens and code words at the distance where the look-around limits matter.
EDGE_CASES = [
    "รหัส" + " " * 40 + "ห้าหนึ่งเจ็ดแปดสี่หนึ่ง",
    "โอทีพี" + "x" * 15 + "สองพันห้าร้อย",
    "โอทีพี" + "x" * 16 + "สองพันห้าร้อย",
    "รหัส" + "x" * 15 + "สองพันห้าร้อย",
    "รหัส" + "x" * 16 + "สองพันห้าร้อย",
    "ยอด" + " " * 60 + "สองพันห้าร้อย",
    "ยอด" + " " * 60 + "สองพันห้าร้อยบาท",
    "ห้าสิบ" + " " * 50 + "-" + " " * 50 + "หกสิบบาท",
    "-" + " " * 30 + "ห้าสิบ",
    "โอนห้าสิบ" + " " * 30 + "บาท",
    "ห้าร้อย" + "\n" * 9 + "คน",
    "สามสิบ",
    "  ห้าสิบบาท  ",
]


@pytest.mark.parametrize("text", [c[0] for c in CONVERTED] + UNCHANGED + EDGE_CASES)
def test_output_matches_the_first_version(text):
    assert spoken_numbers_to_digits(text) == _reference_spoken(text)


def test_output_matches_the_first_version_on_a_long_text():
    text = " ".join([c[0] for c in CONVERTED] + UNCHANGED + EDGE_CASES) * 4
    assert len(text) > 5_000
    assert spoken_numbers_to_digits(text) == _reference_spoken(text)


def test_a_long_digit_run_still_converts_before_a_counter():
    # known trade-off, see the module docstring: an extra entity is safer than a missed one
    assert spoken_numbers_to_digits("สองสามสี่ห้าวัน") == "2345 วัน"


def test_replies_with_number_words_are_still_blocked_by_the_leak_check():
    for reply in ("หนึ่งสองสามสี่", "ห้าพันบาท", "one two three four"):
        assert leak_reasons(reply), reply


def _found(text):
    entities = extract_entities(spoken_numbers_to_digits(text))
    return [(e.label, e.text.replace(" ", "")) for e in entities]


def test_ner_reads_the_same_account_however_it_is_spoken():
    said = [
        "โอนเข้าบัญชี หนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสอง",  # with ขีด
        "โอนเข้าบัญชี หนึ่ง ห้า แปด ขีด ห้า ขีด เก้า หก ศูนย์ สี่ ห้า ขีด สอง",
        "โอนเข้าบัญชี หนึ่งห้าแปด ห้า เก้าหกศูนย์สี่ห้า สอง",  # pauses, no ขีด
        "โอนเข้าบัญชี หนึ่งห้าแปดห้าเก้าหกศูนย์สี่ห้าสอง",  # run together
    ]
    for s in said:
        accounts = [t for label, t in _found(s) if label == "BANK_ACCOUNT"]
        assert len(accounts) == 1, s
        assert accounts[0].replace("-", "") == "1585960452", s


def test_ner_finds_phone_otp_and_money_from_words():
    assert ("PHONE", "0812345678") in _found("โทรมาที่ ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด")
    assert ("OTP", "517841") in _found("รหัสโอทีพีคือ ห้า หนึ่ง เจ็ด แปด สี่ หนึ่ง")
    assert ("OTP", "517841") in _found("รหัสคือห้าแสนหนึ่งหมื่นเจ็ดพันแปดร้อยสี่สิบเอ็ดค่ะ")
    assert ("MONEY", "50000บาท") in _found("โอนห้าหมื่นบาท")
    assert ("MONEY", "49500บาท") in _found("ภาษีสี่หมื่นเก้าพันห้าร้อยบาท")


def test_clean_turn_converts_and_keeps_the_asr_text():
    asr = "โอนเข้าบัญชี หนึ่งห้าแปดขีดห้าขีดเก้าหกศูนย์สี่ห้าขีดสอง"
    turn = clean_turn(Turn("A", 0, 1, asr))
    assert turn.text == "โอนเข้าบัญชี 158-5-96045-2"
    assert turn.raw == asr
    assert [e.label for e in extract_entities(turn.text)] == ["BANK_ACCOUNT"]
