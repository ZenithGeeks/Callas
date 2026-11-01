import pytest

from data_pipeline import simulate
from data_pipeline.config import ConfigError, load_config
from data_pipeline.fill import SlotFiller
from data_pipeline.scenarios import sample_config
from data_pipeline.spoken import DASH, PAUSE, pick_speech_style, say, spoken_text
from data_pipeline.store import Store
from data_pipeline.tests.test_simulate import REPLIES, SCAM, FakeVoices, judge, make_agents

CONFIG = load_config()


def entity(text, label, start):
    return {"text": text, "label": label, "start": start, "end": start + len(text)}


def test_an_account_is_read_digit_by_digit_with_the_dash_word():
    assert say("BANK_ACCOUNT", "158-5-96045-2", DASH) == "หนึ่งห้าแปด ขีด ห้า ขีด เก้าหกศูนย์สี่ห้า ขีด สอง"


def test_an_account_in_the_pause_style_is_joined_by_pauses():
    assert say("BANK_ACCOUNT", "158-5-96045-2", PAUSE) == "หนึ่งห้าแปด, ห้า, เก้าหกศูนย์สี่ห้า, สอง"


def test_phone_and_id_card_are_read_digit_by_digit():
    assert say("PHONE", "081-774-2578", DASH) == "ศูนย์แปดหนึ่ง ขีด เจ็ดเจ็ดสี่ ขีด สองห้าเจ็ดแปด"
    assert say("PHONE", "081-774-2578", PAUSE) == "ศูนย์แปดหนึ่ง, เจ็ดเจ็ดสี่, สองห้าเจ็ดแปด"
    assert say("ID_CARD", "1-2345-67890-12-3", PAUSE).split(", ")[0] == "หนึ่ง"
    assert say("ID_CARD", "1-2345-67890-12-3", DASH).count("ขีด") == 4


@pytest.mark.parametrize("style", [DASH, PAUSE])
def test_an_otp_is_digit_by_digit_with_no_dash(style):
    assert say("OTP", "510255", style) == "ห้าหนึ่งศูนย์สองห้าห้า"


def test_money_is_read_with_place_value():
    assert say("MONEY", "49,500 บาท", DASH) == "สี่หมื่นเก้าพันห้าร้อยบาท"
    assert say("MONEY", "350 บาท", PAUSE) == "สามร้อยห้าสิบบาท"


def test_a_birth_date_is_read_as_numbers():
    assert say("BIRTH_DATE", "12 มีนาคม 2511", DASH) == "สิบสอง มีนาคม สองพันห้าร้อยสิบเอ็ด"


def test_other_labels_are_said_as_written():
    assert say("PERSON", "สมชาย", DASH) == "สมชาย"
    assert say("URL", "https://secure-gift42.com", DASH) == "https://secure-gift42.com"


def test_text_outside_the_spans_is_untouched():
    text = "โอนมาที่ 158-5-96045-2 เป็นเงิน 49,500 บาท นะคะ"
    first = text.index("158")
    second = text.index("49,500")
    entities = [
        entity("158-5-96045-2", "BANK_ACCOUNT", first),
        entity("49,500 บาท", "MONEY", second),
    ]
    assert spoken_text(text, entities, DASH) == (
        "โอนมาที่ หนึ่งห้าแปด ขีด ห้า ขีด เก้าหกศูนย์สี่ห้า ขีด สอง เป็นเงิน สี่หมื่นเก้าพันห้าร้อยบาท นะคะ"
    )
    assert spoken_text("ไม่มีตัวเลข", [], DASH) == "ไม่มีตัวเลข"


def test_a_stale_span_is_left_as_written():
    bad = entity("123456", "OTP", 0)
    assert spoken_text("999999 ค่ะ", [bad], DASH) == "999999 ค่ะ"


def test_the_style_is_chosen_per_seed_and_follows_the_share():
    assert all(pick_speech_style("romance", s, 0.0) == PAUSE for s in range(30))
    assert all(pick_speech_style("romance", s, 1.0) == DASH for s in range(30))
    first = [pick_speech_style("romance", s, 0.5) for s in range(60)]
    assert first == [pick_speech_style("romance", s, 0.5) for s in range(60)]
    assert {DASH, PAUSE} == set(first)


def test_the_share_must_be_between_0_and_1(tmp_path):
    import shutil

    import yaml

    from data_pipeline.config import DEFAULT_DIR

    folder = tmp_path / "cfg"
    shutil.copytree(DEFAULT_DIR, folder)
    path = folder / "agents.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["speech"]["say_dash_share"] = 1.5
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="say_dash_share"):
        load_config(folder)


def test_the_voice_is_given_the_spoken_text_and_the_record_keeps_both(tmp_path):
    caller = [dict(SCAM[0]), dict(SCAM[1]), dict(SCAM[2], line="โอนมาที่ {BANK_ACCOUNT} นะครับ")]
    replies = [dict(REPLIES[0]), dict(line="รหัส {OTP} ค่ะ", end_call=False)]
    agents = make_agents(caller, replies, judge(start=2))
    voices = FakeVoices()
    rec = simulate.run_call(
        "phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG, voices=voices
    ).to_dict()
    turns = rec["generation"]["turns"]
    assert rec["generation"]["speech_style"] in (DASH, PAUSE)
    filler = SlotFiller(sample_config("phishing_otp", 1, CONFIG), CONFIG.details)
    otp, account = filler.values["OTP"], filler.values["BANK_ACCOUNT"]
    assert otp in turns[3]["text"] and otp not in turns[3]["spoken_text"]  # text keeps digits
    assert account in turns[4]["text"] and account not in turns[4]["spoken_text"]
    assert [said for said, _ in voices.spoken] == [t["spoken_text"] for t in turns]


def test_the_pause_style_never_says_the_dash_word():
    assert "ขีด" not in say("BANK_ACCOUNT", "158-5-96045-2", PAUSE)


def test_house_and_soi_numbers_are_read_by_place_value():
    said = say("ADDRESS", "45 ซอยสุขสันต์ 4 อำเภอบ้านใหม่ จังหวัดขอนแก่น", DASH)
    assert said == "สี่สิบห้า ซอยสุขสันต์ สี่ อำเภอบ้านใหม่ จังหวัดขอนแก่น"
    assert say("ADDRESS", "12/3", DASH) == "สิบสองทับสาม"
    assert say("ADDRESS", "88/21 ซอยร่มเย็น 12", DASH) == "แปดสิบแปดทับยี่สิบเอ็ด ซอยร่มเย็น สิบสอง"


def test_a_postcode_stays_digit_by_digit():
    assert say("ADDRESS", "จังหวัดขอนแก่น 40000", DASH) == "จังหวัดขอนแก่น สี่ศูนย์ศูนย์ศูนย์ศูนย์"


def test_a_line_id_is_spelled_letter_by_letter_in_thai():
    assert say("LINE_ID", "viphost", DASH) == "วี ไอ พี เอช โอ เอส ที"
    assert say("LINE_ID", "luxwin", PAUSE) == "แอล ยู เอ็กซ์ ดับเบิลยู ไอ เอ็น"


def test_a_card_number_is_said_digit_by_digit():
    assert say("CARD_NUMBER", "1234-5678-9012-3456", DASH) == say(
        "ID_CARD", "1234-5678-9012-3456", DASH
    )
    assert "ขีด" in say("CARD_NUMBER", "1234-5678-9012-3456", DASH)
