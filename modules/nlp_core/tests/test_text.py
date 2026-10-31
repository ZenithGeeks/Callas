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


@pytest.fixture
def no_mlm(monkeypatch):
    monkeypatch.setattr(text, "_mlm", lambda: None)


@pytest.mark.parametrize(
    ("line", "fixed"),
    [
        ("โอนเงิืนเพิ่มเติม", "โอนเงินเพิ่มเติม"),  # the second vowel dropped
        ("ถ้ามืิอยืนยัน", "ถ้ามือยืนยัน"),  # spell() also gives มื้อ, which drops no vowel
        ("นี้มืีจำกัดมาก", "นี้มีจำกัดมาก"),  # the first vowel dropped
        ("ถูกส่งกลัีบไป", "ถูกส่งกลับไป"),  # กลับ or กลีบ: without the MLM, the more common
    ],
)
def test_two_different_stacked_vowels_lose_the_wrong_one(no_mlm, line, fixed):
    assert autocorrect_text(line, mode="pythainlp") == fixed


def test_the_mlm_picks_between_stack_fixes(monkeypatch):
    seen = []

    def score(left, word, right):
        seen.append((left, word, right))
        return 1.0 if word == "กลีบ" else 0.0

    monkeypatch.setattr(text, "_mlm", lambda: score)
    assert autocorrect_text("ถูกส่งกลัีบไป", mode="pythainlp") == "ถูกส่งกลีบไป"
    assert {w for _, w, _ in seen} == {"กลับ", "กลีบ"}
    assert all(right == "ไป" for _, _, right in seen)  # scored in the sentence


def test_one_stack_fix_needs_no_mlm(monkeypatch):
    monkeypatch.setattr(text, "_mlm", lambda: pytest.fail("the MLM was loaded"))
    assert autocorrect_text("โอนเงิืน", mode="pythainlp") == "โอนเงิน"


@pytest.mark.parametrize(
    "keep",
    [
        "ติดตั้งแอปใหม่ทันที",  # loanwords.txt
        "เรื่องคริปโตนะ",
        "ส่งรหัสโอทีพีมา",
        "เฮ้ ว่าไง",  # Volubilis
        "สยามพลัสอะไร",  # Volubilis and ICU: spell() made it พลัง
        "ไปก่อนนะบ๊าย",  # ICU: spell() made it บาย
    ],
)
def test_known_loanwords_are_left_alone(keep):
    # spell() used to turn these into แผน, เข้, สิงโต
    assert autocorrect_text(keep, mode="pythainlp") == keep


def _scorer(scores, seen=None):
    """A fake MLM: the score of each span text, -10 for any other."""

    def score(left, word, right):
        if seen is not None:
            seen.append(word)
        return scores.get(word, -10.0)

    return lambda: score


def test_repair_swaps_a_mark_scored_in_the_sentence(monkeypatch):
    monkeypatch.setattr(text, "_mlm", _scorer({"ไปที่บ้าน": 0.0, "ไปที้บ้าน": -5.0}))
    assert text._repair("ฉันไปที้บ้าน") == "ฉันไปที่บ้าน"


def test_repair_needs_the_fix_to_beat_the_original_by_10x(monkeypatch):
    # 3x over the runner-up is not enough when the unchanged span is close (log 10 = 2.3)
    monkeypatch.setattr(text, "_mlm", _scorer({"ไปที่บ้าน": 0.0, "ไปที้บ้าน": -2.0}))
    assert text._repair("ฉันไปที้บ้าน") == "ฉันไปที้บ้าน"


def test_repair_is_off_without_the_mlm_or_the_setting(monkeypatch):
    monkeypatch.setattr(text, "_mlm", lambda: None)
    assert text._repair("ฉันไปที้บ้าน") == "ฉันไปที้บ้าน"
    monkeypatch.setattr(text, "_mlm", lambda: pytest.fail("the MLM was loaded"))
    assert autocorrect_text("ฉันไปที้บ้าน", mode="pythainlp", repair=False) == "ฉันไปที้บ้าน"


def test_repair_never_scores_a_line_with_no_unknown_word(monkeypatch):
    monkeypatch.setattr(text, "_mlm", lambda: pytest.fail("the MLM was loaded"))
    assert text._repair("ฉันไปที่บ้าน") == "ฉันไปที่บ้าน"


def test_repair_leaves_a_span_with_too_many_candidates(monkeypatch):
    seen = []
    monkeypatch.setattr(text, "_mlm", _scorer({}, seen))
    monkeypatch.setattr(text, "_repair_candidates", lambda span: [f"{span}{k}" for k in range(9)])
    assert text._repair("ฉันไปที้บ้าน") == "ฉันไปที้บ้าน"
    assert seen == []
