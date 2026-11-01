import pytest

from tts_voice_agent.spelling import looks_misspelled, spelling_problems
from tts_voice_agent.templates import TEMPLATES

# Real replies from qwen3.8-27b-fp8 during DSPy tuning.
GARBLED = [
    "สวสดีครับ ขณะน้ีเจ้าของสายไม่สามารถรับสายได้",
    "สวสดีครับ ขณะน้ีไม่สะดวกในการรบัสาย ค่ะ",
]
CLEAN = [
    "ขออภัยครับ ขณะนี้ไม่สามารถรับสายได้ กรุณาติดต่อกลับในภายหลัง",
    "ขออภัยค่ะ ขณะนี้เจ้าของสายไม่ว่างรับสาย กรุณาฝากข้อความหรือโทรกลับในภายหลังค่ะ",
    "ขอบคุณสำหรับข้อมูลค่ะ แต่ตอนนี้ไม่สะดวกคุยเรื่องแพ็กเกจนะคะ",
]


@pytest.mark.parametrize("text", GARBLED)
def test_garbled_thai_is_caught(text):
    assert looks_misspelled(text)


@pytest.mark.parametrize("text", CLEAN + list(TEMPLATES))
def test_clean_thai_passes(text):
    assert not looks_misspelled(text)


def test_doubled_tone_or_vowel_marks_are_caught():
    # garbled Thai with repeated marks (เต้้น, กััน, นั้้น) is rejected like a tone-order error
    assert spelling_problems("ผมก็ตื่นเต้้นเหมือนกัันครับ") == ["tone_mark_order"]
    assert spelling_problems("นั้้นแหละ") == ["tone_mark_order"]
    # correctly written นี้ (vowel then tone) stays clean
    assert spelling_problems("ตอนนี้สบายดี") == []


def test_mai_yamok_is_not_a_misspelling():
    # ๆ repeats the word before it; spaced or not, it must not be flagged (romance_00015)
    assert spelling_problems("ตัวเองสู้ๆ นะครับ พักผ่อนเยอะๆ นะ") == []
    assert spelling_problems("ขอบคุณมาก ๆ เลยนะ") == []
    # a misspelled word in front of ๆ is still caught
    assert looks_misspelled("ไม่สะดวกในการรบัสายๆ")


def test_safe_reply_uses_template_for_garbled_thai():
    from nlp_core.types import NLPResult
    from tts_voice_agent.memory import CallState
    from tts_voice_agent.reply import safe_reply

    class FakeLLM:
        def complete(self, job, context):
            return GARBLED[0]

    text, template_used = safe_reply(NLPResult(), CallState(), FakeLLM())
    assert template_used and text in TEMPLATES
