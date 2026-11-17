"""The PhayaThaiBERT classifier: the text it builds for the model, its answer rules, and (only when
the trained models and torch are on this machine) the real model."""

import importlib.util

import pytest

from nlp_core import phayathaibert as pt
from nlp_core import registry


def test_speakers_become_the_training_letters():
    text = "A: สวัสดีครับ\nCaller: ติดต่อจากธนาคารครับ\nagent: ไม่สะดวกค่ะ\nC: ฮัลโหล"
    assert pt.turns_of(text) == [
        ("A", "สวัสดีครับ"),
        ("B", "ติดต่อจากธนาคารครับ"),
        ("A", "ไม่สะดวกค่ะ"),  # Callas speaks for the person called
        ("B", "ฮัลโหล"),  # any other voice is the caller side
    ]


def test_lines_without_a_speaker_are_left_out():
    text = "ข้อความสรุป: สายเก่า\nB: โอนเงินมาเลย"  # a Thai word before a colon is no speaker
    assert pt.turns_of(text) == [("B", "โอนเงินมาเลย")]


def test_a_text_message_is_one_caller_turn():
    assert pt.turns_of("คุณได้รับรางวัล กดลิงก์") == [("B", "คุณได้รับรางวัล กดลิงก์")]


def test_same_speaker_lines_are_joined_and_empty_lines_dropped():
    assert pt.turns_of("B: หนึ่ง\nB:   \nB: สอง\nA: ครับ") == [("B", "หนึ่ง สอง"), ("A", "ครับ")]


def test_sensitive_values_are_masked_like_the_training_rows():
    row = pt.turns_of("B: โทรกลับที่ 081-234-5678 แล้วโอน 5,000 บาท")[0][1]
    assert row == "โทรกลับที่ [PHONE] แล้วโอน [MONEY]"


def test_the_oldest_lines_go_first_and_the_newest_always_stays():
    text = "A: หนึ่ง\nB: สอง\nA: สาม\nB: สี่"
    lines = lambda row: len(row.splitlines())  # noqa: E731
    assert pt.model_row(text, lines, 2) == "A: สาม\nB: สี่"
    assert pt.model_row(text, lines, 10) == text
    assert pt.model_row(text, lines, 0) == "B: สี่"


def test_to_model_text_uses_the_space_token():
    assert pt.to_model_text("A: สวัสดี ครับ\nB: ค่ะ", "<_>") == "A:<_>สวัสดี<_>ครับ<_>B:<_>ค่ะ"
    assert pt.to_model_text("A:  x\nB: y", None) == "A: x B: y"


def test_answer_rules():
    assert pt.answer({"not_scam": 0.9, "scam": 0.1}, None) == (False, None, 0.9)
    assert pt.answer({"not_scam": 0.45, "scam": 0.55}, None) == (True, None, 0.0)  # unsure
    assert pt.answer({"not_scam": 0.2, "scam": 0.8}, {"lottery": 0.7, "romance": 0.3}) == (
        True,
        "lottery",
        0.8,
    )
    assert pt.answer({"not_scam": 0.2, "scam": 0.8}, {"lottery": 0.4, "romance": 0.35}) == (
        True,
        None,
        0.8,
    )


def test_registered_under_its_name():
    assert "phayathaibert" in registry.available("classifier")


def test_missing_model_says_how_to_get_it(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    with pytest.raises(FileNotFoundError, match="NLP_MODELS_DIR"):
        pt.PhayaThaiBertClassifier(tmp_path)


HAVE_MODEL = (pt.DEFAULT_MODELS_DIR / pt.GATE_DIR_NAME / "config.json").exists() and all(
    importlib.util.find_spec(m) for m in ("torch", "transformers")
)


@pytest.fixture(scope="module")
def classifier():
    return pt.PhayaThaiBertClassifier()


@pytest.mark.skipif(not HAVE_MODEL, reason="needs models/phayathaibert-scam-gate and the ml extra")
def test_real_model_reads_a_scam_and_a_normal_call(classifier):
    # The amount and the account matter: on the 2026-11-16 model the same call without them
    # scored 0.0003 (see "Known limits" in modules/orchestrator/README.md). This test checks the
    # wiring, which the masked values pass through, not that weakness.
    scam = (
        "B: สวัสดีครับ ติดต่อจากสถานีตำรวจครับ บัญชีของคุณพัวพันกับคดีฟอกเงิน\n"
        "A: หา จริงเหรอครับ\n"
        "B: คุณต้องโอน 50,000 บาท ไปที่บัญชี 123-4-56789-0 ภายในวันนี้ครับ"
    )
    normal = (
        "B: สวัสดีค่ะ โทรจากคลินิกทันตกรรมค่ะ ขอยืนยันนัดขูดหินปูนวันเสาร์สิบโมงนะคะ\n"
        "A: ได้ครับ ไปตามนัดครับ\n"
        "B: ขอบคุณค่ะ ถ้าติดธุระโทรเลื่อนได้นะคะ"
    )
    label_scam, label_type, confidence = classifier.predict(scam)
    assert label_scam and confidence >= pt.GATE_MIN_CONFIDENCE
    assert label_type in (None, "fake_authority")
    assert classifier.predict(normal)[0] is False


@pytest.mark.skipif(not HAVE_MODEL, reason="needs models/phayathaibert-scam-gate and the ml extra")
def test_real_model_keeps_the_newest_line_of_a_long_call(classifier):
    long_call = "\n".join(["A: คุยเรื่องทั่วไปเกี่ยวกับอากาศวันนี้ครับ ฝนตกหนักมาก"] * 60)
    row = classifier.row(long_call + "\nB: บรรทัดล่าสุด")
    assert row.endswith("B: บรรทัดล่าสุด")
    assert classifier.count_tokens(row) <= classifier.max_length
