import logging

import pytest
from nlp_core.types import Entity, NLPResult

from tts_voice_agent.leak import leak_reasons, leaks_sensitive
from tts_voice_agent.memory import CallState
from tts_voice_agent.reply import safe_reply
from tts_voice_agent.templates import TEMPLATES, pick_template


class FakeLLM:
    def __init__(self, text="", exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def complete(self, job, context):
        self.calls.append((job, context))
        if self.exc:
            raise self.exc
        return self.text


ENTITIES = [
    Entity("สมชาย", "PERSON", 0, 5),
    Entity("123-4-56789-0", "BANK_ACCOUNT", 10, 23),
]

BAD = [
    "โทรมาที่ 0812345678 ได้เลยค่ะ",
    "โทรมาที่ 081-234-5678 ได้เลยค่ะ",
    "โทรมาที่ ๐๘๑๒๓๔๕๖๗๘ ได้เลยค่ะ",
    "โทรมาที่ ０８１ ได้เลยค่ะ",
    "โทรมาที่ 08​1 ได้เลยค่ะ",
    "โทรมาที่ ศูนย์แปดหนึ่ง ได้เลยค่ะ",
    "โทรมาที่ แปด หนึ่ง สอง ได้เลยค่ะ",
    "call me on one two three",
    "zero-eight-one",
    "รหัส OTP คือ หก ค่ะ",
    "ส่ง otp มาได้เลยค่ะ",
    "เข้าไปที่ https://evil.example/x ค่ะ",
    "เข้าไปที่ evil-bank.com ค่ะ",
    "ส่งมาที่ boss@example.com ค่ะ",
    "สวัสดีค่ะ คุณสมชายไม่อยู่ค่ะ",
    "สวัสดีค่ะ คุณ สม ชาย ไม่อยู่ค่ะ",
    "บัญชี ๑๒๓.๔.๕๖๗๘๙.๐",
    "ขอเลขบัญชีหน่อยค่ะ",
    "โอนผ่านพร้อมเพย์ได้ค่ะ",
    "ขอบัญชีหน่อยค่ะ",
    "ขอรหัสหน่อยค่ะ",
    "เลข 1 2 3 4 5 6 7 8 9 0 ค่ะ",
    "١٢٣",  # Arabic-Indic digits
    "โทรมาที่ แปด, หนึ่ง, สอง ค่ะ",
    "โทรมาที่ แปด/หนึ่ง/สอง ค่ะ",
    "ศูนย์แปดโทหนึ่ง",
    "eight, one / two",
    "สวัสดีค่ะ คุณพัชราไม่อยู่ค่ะ",  # a name the regex NER cannot see
    "ดิฉันชื่อมะลิค่ะ",
    "พี่ต้อมไม่อยู่ค่ะ",
    "ทางธนาคารจะติดต่อกลับค่ะ",
    "Thank you, goodbye",
]

GOOD = [
    "ขอบคุณค่ะ ขณะนี้ไม่สะดวกคุยค่ะ",
    "ขออภัยค่ะ ไม่สะดวกรับสายค่ะ",
    "ขอบคุณที่โทรมาค่ะ ไว้ค่อยคุยกันใหม่นะคะ",
    "สามีไม่สะดวกรับสายค่ะ",  # สามี contains สาม but is not a run of number words
]


@pytest.mark.parametrize("text", BAD)
def test_leaky_replies_fall_back_to_template(text):
    state = CallState(entities_seen=list(ENTITIES))
    result = NLPResult()
    out, used = safe_reply(result, state, FakeLLM(text))
    assert used and out == pick_template(result)
    assert out != text


def test_account_with_dots_blocked_by_entity_alone():
    # No digits, keywords, name words or Latin: only the entity rule can catch this.
    ents = [Entity("มะ-ลิ.วัลย์", "PERSON", 0, 11)]
    assert leak_reasons("ฝากถึงมะลิวัลย์ค่ะ", ents) == ["entity:PERSON"]


def test_short_entities_ignored():
    assert not leaks_sensitive("ขอบคุณค่ะ", [Entity("ข", "PERSON", 0, 1)])


@pytest.mark.parametrize("text", GOOD)
def test_clean_replies_pass(text):
    state = CallState(entities_seen=list(ENTITIES))
    out, used = safe_reply(NLPResult(), state, FakeLLM(text))
    assert (out, used) == (text, False)


def test_llm_called_with_reply_job_and_context():
    from asr_diarization.types import Turn

    state = CallState()
    state.add(Turn("A", 0, 1, "สวัสดี"), [])
    llm = FakeLLM("ขอบคุณค่ะ")
    safe_reply(NLPResult(), state, llm)
    assert llm.calls == [("reply", "A: สวัสดี")]


def test_llm_exception_gives_template():
    out, used = safe_reply(NLPResult(), CallState(), FakeLLM(exc=RuntimeError("boom")))
    assert used and out in TEMPLATES


@pytest.mark.parametrize("text", ["", "   ", None, "ก" * 201])
def test_empty_or_long_gives_template(text):
    out, used = safe_reply(NLPResult(), CallState(), FakeLLM(text))
    assert used and out in TEMPLATES


def test_leaked_text_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    safe_reply(NLPResult(), CallState(), FakeLLM("เบอร์ 0812345678 ค่ะ"))
    assert "0812345678" not in caplog.text
    assert "digit" in caplog.text


def test_templates_clean_and_pick_deterministic():
    assert all(not leaks_sensitive(t, []) for t in TEMPLATES)
    for t in ("fake_authority", "romance", "investment", "phishing_otp", "lottery", None, "zzz"):
        r = NLPResult(label_type=t)
        assert pick_template(r) == pick_template(r) and pick_template(r) in TEMPLATES
