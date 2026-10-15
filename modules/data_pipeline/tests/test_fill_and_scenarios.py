import pytest

from data_pipeline.config import CALL_TYPES, load_config
from data_pipeline.fill import SlotFiller
from data_pipeline.scenarios import sample_config

CONFIG = load_config()


def filler(call_type="phishing_otp", seed=3):
    return SlotFiller(sample_config(call_type, seed, CONFIG), CONFIG.details)


def test_spans_point_at_the_filled_values():
    f = filler()
    text, spans, problems = f.fill("โทร {PHONE} หรือเข้า {URL} แล้วแจ้ง {OTP} ค่ะ")
    assert problems == []
    assert [s["label"] for s in spans] == ["PHONE", "URL", "OTP"]
    for s in spans:
        assert text[s["start"] : s["end"]] == s["text"]


def test_values_stay_the_same_within_a_call():
    f = filler()
    a, _, _ = f.fill("{BANK_ACCOUNT}")
    b, _, _ = f.fill("เลขบัญชี {BANK_ACCOUNT} นะ")
    assert a in b


def test_money_unit_is_not_doubled():
    text, spans, _ = filler().fill("โอน {MONEY} บาท ด่วน")
    assert text.count("บาท") == 1
    assert spans[0]["text"].endswith("บาท")


def test_unknown_slot_and_loose_number_are_flagged():
    _, _, problems = filler().fill("บ้านเลขที่ {HOUSE_NUMBER} โทร 0812345678")
    assert "unknown_slot:HOUSE_NUMBER" in problems
    assert "number_without_slot" in problems


@pytest.mark.parametrize("call_type", CALL_TYPES)
def test_same_seed_same_call(call_type):
    assert sample_config(call_type, 7, CONFIG) == sample_config(call_type, 7, CONFIG)


def test_label_comes_from_the_call_type():
    assert sample_config("benign", 1, CONFIG).label_type is None
    c = sample_config("lottery", 1, CONFIG)
    assert c.label_scam and c.label_type == "lottery"
    assert c.scenario in CONFIG.scenarios.scams["lottery"]


def test_calls_are_long_enough_for_the_reveal():
    lo, hi = CONFIG.scenarios.calls.max_turns
    for seed in range(200):
        c = sample_config("romance", seed, CONFIG)
        assert lo <= c.max_turns <= hi
        assert c.max_turns >= min(hi, 2 * c.reveal_turn + 4)


def test_names_match_gender():
    for seed in range(50):
        c = sample_config("benign", seed, CONFIG)
        f = SlotFiller(c, CONFIG.details)
        assert f.values["CALLER_NAME"] in CONFIG.details.names[c.caller_gender]
        assert f.values["RECEIVER_NAME"] in CONFIG.details.names[c.receiver.gender]


@pytest.mark.parametrize(
    "line",
    [
        "โอน 50,000 บาท",
        "ค่าธรรมเนียม 5,000 บาท",
        "เลข 1234",
        "โทร 081.234.5678",
        "โทร 081-234-5678",
        "โทร 0812345678",
        "ยอด 5 000 บาท",
        "บัญชี 123-4-56789-0",
        "ปี 2568",
        "{PHONE} หรือ 1234 ก็ได้",
    ],
)
def test_bare_numbers_are_flagged(line):
    assert "number_without_slot" in filler().fill(line)[2]


@pytest.mark.parametrize(
    "line",
    [
        "รอ 3 นาทีนะคะ",
        "ภายใน 15 วัน",
        "นัดเวลา 10.30 น.",
        "ตอน 9.05 ค่ะ",
        "มี 120 คน",
        "รหัสคือ {OTP} ค่ะ",
        "โทรมาที่ {PHONE}{OTP}",
    ],
)
def test_small_numbers_times_and_slots_are_not_flagged(line):
    assert "number_without_slot" not in filler().fill(line)[2]


@pytest.mark.parametrize("line", ["รหัส {otp} ค่ะ", "รหัส {OTP2} ค่ะ", "รหัส {{OTP}} ค่ะ", "}", "{ORG"])
def test_unfilled_braces_are_flagged(line):
    text, _, problems = filler().fill(line)
    assert "stray_brace" in problems


def test_filled_slots_leave_no_brace():
    assert filler().fill("รหัส {OTP} จาก {ORG}")[2] == []


@pytest.mark.parametrize(
    "line",
    ["สวัสดีครับ\nมีอะไรให้ช่วยครับ", "A: สวัสดีครับ", "B: ค่ะ", "  A : ค่ะ", "ค่ะ\r\nครับ"],
)
def test_a_line_must_be_one_line_without_a_speaker_tag(line):
    assert "line_break" in filler().fill(line)[2]


@pytest.mark.parametrize("line", ["สวัสดีครับ", "ผม A: ไม่ใช่", "Always ok", "ABC: x"])
def test_ordinary_lines_are_not_line_breaks(line):
    assert "line_break" not in filler().fill(line)[2]
