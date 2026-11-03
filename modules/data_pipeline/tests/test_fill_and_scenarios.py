import re
from datetime import date

import pytest

from data_pipeline.config import CALL_TYPES, load_config
from data_pipeline.fill import (
    AGE_REFERENCE_DATE,
    THAI_MONTHS,
    SlotFiller,
    receiver_leaks,
)
from data_pipeline.scenarios import sample_config
from nlp_core.types import ENTITY_LABELS

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


def test_podcast_org_comes_from_the_podcast_list():
    calls = [sample_config("investment", s, CONFIG) for s in range(200)]
    podcast = [c for c in calls if c.scenario.org == "พอดแคสต์"]
    assert podcast
    for c in podcast:
        assert SlotFiller(c, CONFIG.details).values["ORG"] in CONFIG.details.podcasts


def test_school_org_comes_from_the_school_list():
    calls = [sample_config("benign", s, CONFIG) for s in range(300)]
    school = [c for c in calls if c.scenario.org == "โรงเรียน"]
    assert school
    for c in school:
        org = SlotFiller(c, CONFIG.details).values["ORG"]
        assert org in CONFIG.details.schools  # a school's name, not the word โรงเรียน
        assert SlotFiller(c, CONFIG.details).values["ORG"] == org  # same seed, same school
    # an older config folder without schools still gets an organisation's name
    old_details = CONFIG.details.model_copy(update={"schools": []})
    assert SlotFiller(school[0], old_details).values["ORG"] in CONFIG.details.made_up_orgs


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


@pytest.mark.parametrize(
    "line",
    [
        "โอน 50,000 บาท",
        "ค่าธรรมเนียม 5,000 บาท",
        "เลข 1234",
        "โทร 081.234.5678",
        "โทร 081-234-5678",
        "โทร 0812345678",
        "ยอด 5 000 บาท",
        "บัญชี 123-4-56789-0",
        "ปี 2568",
        "{PHONE} หรือ 1234 ก็ได้",
    ],
)
def test_bare_numbers_are_flagged(line):
    assert "number_without_slot" in filler().fill(line)[2]


@pytest.mark.parametrize(
    "line",
    [
        "รอ 3 นาทีนะคะ",
        "ภายใน 15 วัน",
        "นัดเวลา 10.30 น.",
        "ตอน 9.05 ค่ะ",
        "มี 120 คน",
        "รหัสคือ {OTP} ค่ะ",
        "โทรมาที่ {PHONE}{OTP}",
    ],
)
def test_small_numbers_times_and_slots_are_not_flagged(line):
    assert "number_without_slot" not in filler().fill(line)[2]


@pytest.mark.parametrize("line", ["รหัส {otp} ค่ะ", "รหัส {OTP2} ค่ะ", "รหัส {{OTP}} ค่ะ", "}", "{ORG"])
def test_unfilled_braces_are_flagged(line):
    text, _, problems = filler().fill(line)
    assert "stray_brace" in problems


def test_filled_slots_leave_no_brace():
    assert filler().fill("รหัส {OTP} จาก {ORG}")[2] == []


@pytest.mark.parametrize(
    "line",
    ["สวัสดีครับ\nมีอะไรให้ช่วยครับ", "A: สวัสดีครับ", "B: ค่ะ", "  A : ค่ะ", "ค่ะ\r\nครับ"],
)
def test_a_line_must_be_one_line_without_a_speaker_tag(line):
    assert "line_break" in filler().fill(line)[2]


@pytest.mark.parametrize("line", ["สวัสดีครับ", "ผม A: ไม่ใช่", "Always ok", "ABC: x"])
def test_ordinary_lines_are_not_line_breaks(line):
    assert "line_break" not in filler().fill(line)[2]


# ---- the receiver's own details ----------------------------------------------------------

RECEIVER_SLOTS = {
    "RECEIVER_PHONE": "PHONE",
    "RECEIVER_ACCOUNT": "BANK_ACCOUNT",
    "ID_CARD": "ID_CARD",
    "BIRTH_DATE": "BIRTH_DATE",
    "ADDRESS": "ADDRESS",
    "OTP": "OTP",
    "RECEIVER_NAME": "PERSON",
}


@pytest.mark.parametrize("slot,label", RECEIVER_SLOTS.items())
def test_receiver_slot_spans_are_exact(slot, label):
    f = filler()
    text, spans, problems = f.fill("ก่อน {" + slot + "} หลัง")
    value = f.values[slot]
    assert problems == [] and value
    assert text == f"ก่อน {value} หลัง"
    assert spans == [
        {"text": value, "label": label, "start": 5, "end": 5 + len(value)},
    ]


def test_new_entity_labels_exist():
    assert {"ID_CARD", "BIRTH_DATE", "ADDRESS"} <= set(ENTITY_LABELS)


def test_id_card_has_the_usual_layout():
    for seed in range(30):
        assert re.fullmatch(r"[1-8]-\d{4}-\d{5}-\d{2}-\d", filler(seed=seed).values["ID_CARD"])


def test_birth_date_matches_the_age():
    for seed in range(60):
        call = sample_config("phishing_otp", seed, CONFIG)
        day, month, year = SlotFiller(call, CONFIG.details).values["BIRTH_DATE"].split()
        born = date(int(year) - 543, THAI_MONTHS.index(month) + 1, int(day))
        ref = AGE_REFERENCE_DATE
        age = ref.year - born.year - ((ref.month, ref.day) < (born.month, born.day))
        assert age == call.receiver.age


def test_receiver_details_do_not_change_the_old_values():
    """The receiver's values are drawn last, so a seed keeps the values it always had."""
    values = filler("romance", 5).values
    assert [
        values[k] for k in ("CALLER_NAME", "RECEIVER_NAME", "PHONE", "BANK_ACCOUNT", "OTP")
    ] == (["พิชชาภา", "ชัยวัฒน์", "081-774-2578", "825-0-44920-4", "510255"])


def test_receiver_leaks_are_read_from_the_slots_in_order():
    assert receiver_leaks("รหัส {OTP} บัตร {ID_CARD} รหัส {OTP}") == ["OTP", "ID_CARD"]
    assert receiver_leaks("ผม{RECEIVER_NAME} โทรหา {PHONE} {BANK_ACCOUNT}") == []
    assert receiver_leaks("{RECEIVER_PHONE} {RECEIVER_ACCOUNT} {BIRTH_DATE} {ADDRESS}") == [
        "PHONE",
        "BANK_ACCOUNT",
        "BIRTH_DATE",
        "ADDRESS",
    ]


def test_no_made_up_id_card_passes_the_real_checksum():
    """A real Thai ID card ends in a mod-11 check digit; a made-up one must never validate."""
    for seed in range(1000):
        digits = filler("phishing_otp", seed).values["ID_CARD"].replace("-", "")
        check = (11 - sum(int(d) * (13 - i) for i, d in enumerate(digits[:12])) % 11) % 10
        assert int(digits[12]) != check, seed


def test_id_card_is_the_same_for_the_same_seed():
    assert filler(seed=9).values["ID_CARD"] == filler(seed=9).values["ID_CARD"]


def test_friend_calls_use_nicknames():
    friend = [
        c
        for c in (sample_config("romance", s, CONFIG) for s in range(300))
        if c.scenario.relation == "friend"
    ]
    assert friend
    for c in friend:
        f = SlotFiller(c, CONFIG.details)
        assert f.values["CALLER_NAME"] in CONFIG.details.nicknames[c.caller_gender]
        assert f.values["RECEIVER_NAME"] in CONFIG.details.nicknames[c.receiver.gender]


def test_line_id_is_filled_from_the_part_list_and_labelled():
    f = filler("investment", 5)
    text, spans, problems = f.fill("แอดไลน์ไอดี {LINE_ID} ไว้นะครับ")
    value = f.values["LINE_ID"]
    assert problems == []
    assert spans == [{"text": value, "label": "LINE_ID", "start": 12, "end": 12 + len(value)}]
    assert text[12 : 12 + len(value)] == value
    assert f.fill("ไอดี {LINE_ID} นะครับ")[0].count(value) == 1  # the same ID all call
    assert "LINE_ID" in ENTITY_LABELS


def test_line_ids_are_two_different_parts_in_lowercase_letters():
    parts = CONFIG.details.line_ids
    for seed in range(50):
        value = filler("investment", seed).values["LINE_ID"]
        assert re.fullmatch(r"[a-z]{6,8}", value)
        assert any(value == a + b for a in parts for b in parts if a != b)


@pytest.mark.parametrize(
    "line",
    [
        "แอดไลน์ไอดี viphost01 ไว้นะครับ",
        "ไลน์ landdoc นะคะ",
        "แอด LINE kx92b4c1 ได้เลยค่ะ",
        "ไอดีไลน์ HomeProRefund2025 นะคะ",
        "ไลน์ไอดี land_doc_24 ค่ะ",
    ],
)
def test_a_made_up_line_id_is_flagged(line):
    assert "line_id_without_slot" in filler().fill(line)[2]


@pytest.mark.parametrize(
    "line",
    [
        "แอดไลน์ไอดี {LINE_ID} ไว้นะครับ",
        'แอด LINE ชื่อ "PEA Digital Meter Support" นะคะ',
        "ส่งสลิปทาง LINE ได้เลยค่ะ",
        "เดี๋ยวส่ง SMS ไปนะ verify ที่ลิงก์",  # no LINE named: not checked
    ],
)
def test_line_slots_and_names_are_not_flagged(line):
    assert "line_id_without_slot" not in filler().fill(line)[2]


def test_the_paid_like_scenario_is_the_last_investment_one():
    scenario = CONFIG.scenarios.scams["investment"][-1]
    assert scenario.key == "paid_like_line_admin"
    assert "LINE" in scenario.goal and "{MONEY}" not in scenario.goal  # no money on the call


def test_only_task_and_job_scenarios_call_the_receiver_khun_phi():
    tasks = {"task_job", "free_cctv_task", "online_job_onboarding", "paid_like_line_admin"}
    for call_type, scenarios in CONFIG.scenarios.scams.items():
        for scenario in scenarios:
            says_phi = "คุณพี่" in scenario.tactics
            assert says_phi == (call_type == "investment" and scenario.key in tasks), scenario.key


def _passes_luhn(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()][::-1]
    return sum(d if i % 2 == 0 else sum(divmod(d * 2, 10)) for i, d in enumerate(digits)) % 10 == 0


def test_a_card_number_is_16_digits_that_are_never_a_valid_card():
    for seed in range(300):
        value = filler("fake_authority", seed).values["CARD_NUMBER"]
        assert re.fullmatch(r"\d{4}-\d{4}-\d{4}-\d{4}", value)
        assert not _passes_luhn(value)


def test_a_card_number_is_labelled_and_drawn_last():
    f = filler("fake_authority", 5)
    text, spans, problems = f.fill("บัตรหมายเลข {CARD_NUMBER} ค่ะ")
    value = f.values["CARD_NUMBER"]
    assert problems == []
    assert spans == [{"text": value, "label": "CARD_NUMBER", "start": 12, "end": 12 + len(value)}]
    assert "CARD_NUMBER" in ENTITY_LABELS
    # drawn after every other value, so the values a seed gave before #160 stay the same
    assert list(f.values)[-1] == "CARD_NUMBER"


def test_a_card_number_written_as_digits_is_flagged():
    assert "number_without_slot" in filler().fill("บัตรหมายเลข 4111 1111 1111 1111 ค่ะ")[2]


def test_the_credit_card_scenario_is_the_last_fake_authority_one():
    scenario = CONFIG.scenarios.scams["fake_authority"][-1]
    assert scenario.key == "credit_card_unpaid_statement"
    assert "{CARD_NUMBER}" in scenario.tactics and "{BANK_ACCOUNT}" in scenario.goal


@pytest.mark.parametrize(
    "line",
    ["บัตรที่ลงท้ายด้วย {CARD_NUMBER} ค่ะ", "บัตรลงท้าย {CARD_NUMBER} ค่ะ", "ลงท้ายด้วยเลข {CARD_NUMBER}"],
)
def test_a_card_said_by_its_ending_is_only_the_last_4_digits(line):
    f = filler("fake_authority", 5)
    text, spans, problems = f.fill(line)
    last4 = f.values["CARD_NUMBER"][-4:]
    assert problems == []
    assert [s["text"] for s in spans] == [last4] and spans[0]["label"] == "CARD_NUMBER"
    assert f.values["CARD_NUMBER"] not in text


def test_police_org_comes_from_the_police_station_list():
    calls = [sample_config("benign", s, CONFIG) for s in range(400)]
    police = [c for c in calls if c.scenario.org == "สถานีตำรวจ"]
    assert police
    for c in police:
        org = SlotFiller(c, CONFIG.details).values["ORG"]
        assert org in CONFIG.details.police_stations  # a station's name, not the word สถานีตำรวจ
        assert SlotFiller(c, CONFIG.details).values["ORG"] == org  # same seed, same station
    # an older config folder without stations still gets an organisation's name
    old_details = CONFIG.details.model_copy(update={"police_stations": []})
    assert SlotFiller(police[0], old_details).values["ORG"] in CONFIG.details.made_up_orgs


def test_police_station_names_are_stations():
    assert len(set(CONFIG.details.police_stations)) >= 2
    assert all(name.startswith("สถานีตำรวจ") for name in CONFIG.details.police_stations)


def test_a_police_station_does_not_change_the_calls_other_values():
    c = next(
        c
        for s in range(400)
        if (c := sample_config("benign", s, CONFIG)).scenario.org == "สถานีตำรวจ"
    )
    with_station = SlotFiller(c, CONFIG.details).values
    without = SlotFiller(c, CONFIG.details.model_copy(update={"police_stations": []})).values
    assert {k: v for k, v in with_station.items() if k != "ORG"} == {
        k: v for k, v in without.items() if k != "ORG"
    }


def test_a_url_is_the_site_only_without_https():
    # on a call a person says the site, not "https://" (#157)
    for seed in range(20):
        url = filler("phishing_otp", seed).values["URL"]
        assert not url.startswith(("http://", "https://", "www."))
        assert re.fullmatch(r"[a-z]+-[a-z]+\d{2}\.[a-z]+", url)


def test_no_scenario_puts_a_link_in_a_line_account_name():
    # a site said as part of a LINE display name is not something a caller would say (#175)
    named = re.compile(r"named '[^']*\{URL\}[^']*'")
    for scenarios in CONFIG.scenarios.scams.values():
        for scenario in scenarios:
            assert not named.search(scenario.goal + " " + scenario.tactics), scenario.key
    land = {s.key: s for s in CONFIG.scenarios.scams["fake_authority"]}[
        "fake_land_dept_line_contact"
    ]
    assert "named 'กรมที่ดิน เอกสาร'" in land.tactics
