import pytest

from nlp_core.types import Entity
from tts_voice_agent.leak import Reason, leak_reasons
from tts_voice_agent.redact import REDACTED_REASONS, redact_context

from .test_leak_and_reply import BAD, ENTITIES


def code(reason):
    return reason.partition(":")[0]


def blocking_reasons(text, entities):
    return [r for r in leak_reasons(text, entities) if code(r) in REDACTED_REASONS]


def test_every_reason_code_is_a_reason_constant():
    constants = {v for k, v in vars(Reason).items() if k.isupper()}
    seen = {code(r) for text in BAD for r in leak_reasons(text, ENTITIES)}
    assert seen == constants  # BAD covers every code, and every code is a constant


def test_redacted_reasons_are_the_reasons_redaction_clears():
    cleared = set()
    for text in BAD:
        after = {code(r) for r in leak_reasons(redact_context(text, ENTITIES), ENTITIES)}
        cleared |= {code(r) for r in leak_reasons(text, ENTITIES)} - after
    assert cleared == REDACTED_REASONS


@pytest.mark.parametrize(
    "phone",
    [
        "0812345678",
        "081-234-5678",
        "081 234 5678",
        "+66 81 234 5678",
        "๐๘๑๒๓๔๕๖๗๘",
        "０８１２３４５６７８",
        "08​1234 5678",
        "ศูนย์แปดหนึ่งสองสามสี่ห้าหกเจ็ดแปด",
        "ศูนย์ แปด หนึ่ง สอง",
        "zero eight one two",
    ],
)
def test_phone_formats_become_number(phone):
    out = redact_context(f"A: โทรกลับที่ {phone} นะครับ", [])
    assert "[NUMBER]" in out and out.startswith("A: โทรกลับที่ ") and out.endswith("นะครับ")
    assert not blocking_reasons(out, [])


def test_bank_account_and_money():
    ents = [Entity("123-4-56789-0", "BANK_ACCOUNT", 0, 13), Entity("5,000 บาท", "MONEY", 0, 9)]
    out = redact_context("โอนเข้า 123-4-56789-0 จำนวน 5,000 บาท ด่วน", ents)
    assert out == "โอนเข้า [BANK_ACCOUNT] จำนวน [MONEY] ด่วน"


def test_money_without_entity_is_still_a_number():
    out = redact_context("โอนมา 5,000 บาทด่วน", [])
    assert "5" not in out and "[NUMBER]" in out and "บาทด่วน" in out


def test_url_and_email():
    out = redact_context("เข้า https://evil.example/x ก่อน แล้วส่งไป boss@example.com ด้วย", [])
    assert out == "เข้า [LINK] ก่อน แล้วส่งไป [LINK] ด้วย"
    out = redact_context("ไปที่ evil-bank.com ค่ะ", [])
    assert out == "ไปที่ [LINK] ค่ะ"


def test_person_with_spaces_inserted():
    ents = [Entity("สมชาย", "PERSON", 0, 5)]
    out = redact_context("ผมสม ชาย โทรมา ส-ม-ช-า-ย ด้วย", ents)
    assert "สม" not in out.replace("[PERSON]", "")
    assert out.count("[PERSON]") == 2
    assert not blocking_reasons(out, ents)


def test_longest_entity_first():
    ents = [Entity("สมชาย", "PERSON", 0, 5), Entity("สมชาย ใจดี", "PERSON", 0, 10)]
    assert redact_context("คุยกับสมชาย ใจดี", ents) == "คุยกับ[PERSON]"


def test_plain_thai_unchanged_and_deterministic():
    text = "A: สวัสดีครับ ขอสายหน่อย\nB: กำลังทำอาหารอยู่ค่ะ ขำๆ นำ้"
    assert redact_context(text, []) == text
    assert redact_context("", []) == ""
    assert redact_context("0812345678", ENTITIES) == redact_context("0812345678", ENTITIES)


def test_thai_sara_am_is_not_normalised_away():
    assert redact_context("น้ำ 12 ลิตร", []) == "น้ำ [NUMBER] ลิตร"


@pytest.mark.parametrize("bad", BAD)
def test_property_over_bad_strings(bad):
    for ents in ([], ENTITIES):
        out = redact_context(bad, ents)
        assert not blocking_reasons(out, ents), (out, leak_reasons(out, ents))


def test_property_with_entities_inside_text():
    ents = [Entity("วิชัย", "PERSON", 0, 5), Entity("พรีเมียร์ อินเตอร์", "ORG", 0, 5)]
    text = "A: วิ ชัย จาก พรีเมียร์-อินเตอร์ โทร 08 1.234.5678 หรือ แปดหนึ่งสอง"
    out = redact_context(text, ents)
    assert not blocking_reasons(out, ents)
    assert "[PERSON]" in out and "[ORG]" in out
