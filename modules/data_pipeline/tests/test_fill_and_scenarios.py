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
