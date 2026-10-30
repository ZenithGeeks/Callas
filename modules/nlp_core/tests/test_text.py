import pytest

from asr_diarization.types import Turn
from nlp_core import text
from nlp_core.text import autocorrect_text, normalize_text
from tts_voice_agent.memory import clean_turn


def test_normalize_same_as_the_notebook():
    # NFC, pythainlp normalize (drops the doubled tone mark), one space between words
    assert normalize_text("สวัสดี   ครับ。ผม\n\tชื่อ") == "สวัสดี ครับ ผม ชื่อ"
    assert normalize_text("เกิด่่ขึ้น") == "เกิด่ขึ้น"
    assert normalize_text("  ") == ""


@pytest.fixture
def fake_fix(monkeypatch):
    seen = []
    monkeypatch.setattr(
        text, "_fix_word", lambda w: seen.append(w) or ("ขอบคุณ" if w == "ขอบคนุ" else w)
    )
    return seen


def test_autocorrect_off_changes_nothing(fake_fix):
    assert autocorrect_text("ขอบคนุ", mode="off") == "ขอบคนุ"
    assert fake_fix == []


def test_autocorrect_fixes_only_unknown_thai_words(fake_fix):
    out = autocorrect_text("ขอบคนุ ที่โทรมา", mode="pythainlp")
    assert out == "ขอบคุณ ที่โทรมา"
    # a piece is tried alone first, then with its neighbours: the misspelled word's pieces
    # reach the corrector together
    assert "ขอบคนุ" in fake_fix
    assert not any("ที่โทรมา" in w for w in fake_fix)  # known words are never sent


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


@pytest.mark.parametrize(
    ("typo", "fixed"),
    [
        ("คณุ", "คุณ"),  # PyThaiNLP's correct() gives "คุ"
        ("ตอ้ง", "ต้อง"),
        ("แลว้", "แล้ว"),
        ("โอนเงนิ", "โอนเงิน"),  # correct() gives "ถอนเงิน": transfer becomes withdraw
    ],
)
def test_a_mark_on_the_wrong_letter_is_moved_back(typo, fixed):
    assert text._fix_word(typo) == fixed


@pytest.mark.parametrize(
    "typo",
    [
        "ตองจายเทาไร",  # missing marks: no mark to move, so no guess
        "ก็ดดี",  # a doubled letter, not a moved mark (a swap would give ก็ดีด)
        "นี้มมีแค่",
    ],
)
def test_no_guess_when_no_mark_was_moved(typo):
    assert text._fix_word(typo) == typo


def test_a_misplaced_mark_is_fixed_inside_a_sentence():
    # the screenshot's caller line: ให้|คณุ|นะคะ, fixed alone without touching its neighbours
    line = "เพื่อเปิดใช้งานกล้องให้คณุนะคะ"
    assert autocorrect_text(line, mode="pythainlp") == "เพื่อเปิดใช้งานกล้องให้คุณนะคะ"


@pytest.mark.parametrize("keep", ["ติดตั้งแอปใหม่ทันที", "เฮ้ ว่าไง", "เรื่องคริปโตนะ"])
def test_known_loanwords_are_left_alone(keep):
    # spell() used to turn these into แผน, เข้, สิงโต
    assert autocorrect_text(keep, mode="pythainlp") == keep
