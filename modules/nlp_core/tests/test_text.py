import pytest

from asr_diarization.types import Turn
from nlp_core import text
from nlp_core.spelling_fixes import SPELLING_FIXES
from nlp_core.text import autocorrect_text, fix_thai_spelling, normalize_text
from tts_voice_agent.memory import clean_turn


def test_normalize_same_as_the_notebook():
    # NFC, pythainlp normalize (drops the doubled tone mark), one space between words
    assert normalize_text("สวัสดี   ครับ。ผม\n\tชื่อ") == "สวัสดี ครับ ผม ชื่อ"
    assert normalize_text("เกิด่่ขึ้น") == "เกิด่ขึ้น"
    assert normalize_text("  ") == ""


@pytest.mark.parametrize(
    ("garbled", "fixed"),
    [
        ("โอนเงิิน", "โอนเงิน"),  # doubled vowel
        ("ส่่งสลิป", "ส่งสลิป"),  # doubled tone mark
        ("เป็็นแค่เอกสาร", "เป็นแค่เอกสาร"),
        ("ของท่ีทางเรา", "ของที่ทางเรา"),  # tone mark before its vowel
        ("เเปลก", "แปลก"),  # two Sara E
        ("อยากให้คณุ เพิ่ม LINE", "อยากให้คุณ เพิ่ม LINE"),  # listed garbles
        ("ซื้อมาใหคณุ", "ซื้อมาให้คุณ"),  # the longer form wins over ใหคณ
        ("คนแปลกหน้าดวยค่ะ", "คนแปลกหน้าด้วยค่ะ"),
    ],
)
def test_fix_thai_spelling(garbled, fixed):
    assert fix_thai_spelling(garbled) == fixed
    assert fix_thai_spelling(fixed) == fixed  # fixing twice changes nothing


@pytest.mark.parametrize(
    "keep",
    [
        "โอน {MONEY} เข้า {BANK_ACCOUNT} นะคะ",  # slots
        "โทร 0812345678 หรือ https://gift-promo12.com",
        "คณะกรรมการ ไม้ ใหญ่ หนุ่ม",  # real words holding unlisted garbles (คณ, ไม, ให, หนุ่)
        "ผมชื่อสมชาย จาก ฮั่วเซ่งเฮง",  # names
    ],
)
def test_fix_thai_spelling_leaves_correct_text(keep):
    assert fix_thai_spelling(keep) == keep


def test_spelling_fixes_never_hit_a_real_word():
    from pythainlp.corpus import thai_words
    from pythainlp.tokenize import word_tokenize

    words = thai_words()
    for wrong, right in SPELLING_FIXES.items():
        assert not [w for w in words if wrong in w], f"{wrong} is part of a real word"
        assert normalize_text(wrong) == wrong, f"{wrong} is changed by normalize first"
        assert all(t in words for t in word_tokenize(right, engine="newmm")), right


@pytest.fixture
def fake_fix(monkeypatch):
    seen = []
    monkeypatch.setattr(text, "_fix_word", lambda w: seen.append(w) or "ขอบคุณ")
    return seen


def test_autocorrect_off_changes_nothing(fake_fix):
    assert autocorrect_text("ขอบคนุ", mode="off") == "ขอบคนุ"
    assert fake_fix == []


def test_autocorrect_fixes_only_unknown_thai_words(fake_fix):
    out = autocorrect_text("ขอบคนุ ที่โทรมา", mode="pythainlp")
    assert out == "ขอบคุณ ที่โทรมา"
    assert fake_fix == ["ขอบคนุ"]  # the misspelled word's pieces go to the corrector together


@pytest.mark.parametrize(
    "keep",
    [
        "โอนเงิน 12345 บาท",  # digits
        "โทร 0812345678",
        "เปิด http://xyzabc.example.com/pqrst",  # URL
        "ใช้ Facebookk ครับ",  # English
        "ผมชื่อสมชาย",  # a name from the PyThaiNLP name lists
        "เลข ๑๒๓๔๕ ครับ",  # Thai digits
        "ขอแจ้งนัดคุณวราภรณ์พรุ่งนี้",  # a name after คุณ that no list has
    ],
)
def test_autocorrect_leaves_digits_english_urls_and_names(fake_fix, keep):
    assert autocorrect_text(keep, mode="pythainlp") == keep
    assert fake_fix == []


def test_clean_turn_keeps_the_raw_text_only_when_it_changed():
    same = Turn("A", 0, 1, "สวัสดีครับ")
    assert clean_turn(same) is same
    messy = clean_turn(Turn("A", 0, 1, "สวัสดี   ครับ"))
    assert (messy.text, messy.raw) == ("สวัสดี ครับ", "สวัสดี   ครับ")
    assert clean_turn(messy).raw == "สวัสดี   ครับ"  # cleaning twice does not lose the ASR text


def test_setting_rejects_unknown_value(monkeypatch):
    from callas_settings import Settings

    monkeypatch.setenv("AUTOCORRECT", "magic")
    with pytest.raises(ValueError):
        Settings()


def test_autocorrect_never_deletes_a_piece():
    # PyThaiNLP: correct("ป") is "ไป" and correct("แอป") is "แอ"; neither may replace the word
    assert text._fix_word("ป") == "ป"
    assert text._fix_word("แอป") == "แอป"
    assert autocorrect_text("ทำไม ๆ ๆ ไม่มา", mode="pythainlp") == "ทำไม ๆ ๆ ไม่มา"
