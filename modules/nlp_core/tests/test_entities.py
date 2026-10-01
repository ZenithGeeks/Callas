import pytest

from nlp_core.entities import extract_entities


def spans(text):
    return [(e.label, e.text) for e in extract_entities(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("โทร 081-234-5678 ด่วน", [("PHONE", "081-234-5678")]),
        ("เบอร์ 0812345678", [("PHONE", "0812345678")]),
        ("โทรมาที่ +66 81 234 5678 นะ", [("PHONE", "+66 81 234 5678")]),
        ("เบอร์บ้าน 02-123-4567", [("PHONE", "02-123-4567")]),
        ("เบอร์ ๐๘๑-๒๓๔-๕๖๗๘", [("PHONE", "๐๘๑-๒๓๔-๕๖๗๘")]),
        ("โอนเข้า 123-4-56789-0 ทันที", [("BANK_ACCOUNT", "123-4-56789-0")]),
        ("เลขบัญชี 1234567890 ธนาคาร", [("BANK_ACCOUNT", "1234567890")]),
        ("บัญชี 123456789012", [("BANK_ACCOUNT", "123456789012")]),
        ("รหัส OTP คือ 123456 ห้ามบอกใคร", [("OTP", "123456")]),
        ("ขอโค้ด 4821", [("OTP", "4821")]),
        ("OTP ๑๒๓๔๕๖", [("OTP", "๑๒๓๔๕๖")]),
        ("กดลิงก์ https://bit.ly/abc123 เลย", [("URL", "https://bit.ly/abc123")]),
        ("ดูที่ www.example.com.", [("URL", "www.example.com")]),
        ("เว็บ kbank-secure.com/login ด่วน", [("URL", "kbank-secure.com/login")]),
        ("โอน 5,000 บาท", [("MONEY", "5,000 บาท")]),
        ("ค่าธรรมเนียม ๒๕๐๐บาท", [("MONEY", "๒๕๐๐บาท")]),
        ("สวัสดีครับ วันนี้อากาศดี", []),
        ("เลข 12345 เฉย ๆ", []),
    ],
)
def test_extraction(text, expected):
    assert spans(text) == expected


def test_offsets_slice_original_text():
    text = "ติดต่อ ๐๘๑-๒๓๔-๕๖๗๘ โอน 5,000 บาท บัญชี 123-4-56789-0 รหัส OTP 998877 http://a.com/x"
    ents = extract_entities(text)
    assert [e.label for e in ents] == ["PHONE", "MONEY", "BANK_ACCOUNT", "OTP", "URL"]
    for e in ents:
        assert text[e.start : e.end] == e.text


def test_no_overlap_and_phone_not_otp():
    text = "รหัส 0812345678 บัญชี 0812345678"
    ents = extract_entities(text)
    for a, b in zip(ents, ents[1:], strict=False):
        assert a.end <= b.start
    assert ("PHONE", "0812345678") in spans(text) or ("BANK_ACCOUNT", "0812345678") in spans(text)
    assert not any(e.label == "OTP" for e in ents)


def test_bank_context_beats_phone_for_same_digits():
    assert spans("เลขบัญชี 0812345678") == [("BANK_ACCOUNT", "0812345678")]
