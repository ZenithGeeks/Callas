"""The receiver must speak with its own gender's markers: a woman uses ค่ะ / คะ and refers to
herself with ฉัน / เรา (not ผม / ครับ); a man uses ครับ. The model mostly obeys the voice hint,
but slips (a female line that says ผม, seen in testing). wrong_particles flags the slip so it is
recorded as a turn problem — it does not rewrite the line. Soft flag: a reviewer decides.
"""

from data_pipeline.fill import self_identity_question, wrong_particles


def test_flags_male_particle_in_a_female_line():
    assert "ครับ" in wrong_particles("ได้เลยครับ เดี๋ยวดูให้", "female")


def test_flags_male_pronoun_in_a_female_line():
    # the real slip seen in A/B testing: a woman saying ผม
    assert "ผม" in wrong_particles("อ้าว โทรผิดเบอร์รึเปล่าคะ? ผมไม่ใช่คุณยายนะ", "female")


def test_flags_female_particle_in_a_male_line():
    assert "ค่ะ" in wrong_particles("อ๋อ ได้ค่ะ", "male")


def test_flags_dichan_in_a_male_line():
    assert "ดิฉัน" in wrong_particles("ดิฉันไม่ทราบครับ", "male")


def test_correct_female_line_is_clean():
    assert wrong_particles("เดี๋ยวนะคะ รหัสคือ {OTP} ค่ะ", "female") == []


def test_correct_male_line_is_clean():
    assert wrong_particles("ไม่สะดวกครับ ขอวางนะครับ", "male") == []


def test_khanaen_is_not_mistaken_for_the_particle_kha():
    # คะแนน ("score") contains คะ but it is not a sentence-final particle
    assert wrong_particles("สอบได้คะแนนดีเลยค่ะ", "female") == []


def test_fee_word_khaa_is_not_mistaken_for_a_female_particle():
    # ค่าธรรมเนียม ("fee") contains ค่า, which is the word "value/fee", not the particle ค่ะ
    assert wrong_particles("ค่าธรรมเนียมเท่าไรครับ", "male") == []


def test_unknown_gender_flags_nothing():
    assert wrong_particles("ได้เลยครับ", "") == []


def test_receiver_asking_who_it_itself_is_is_flagged():
    assert self_identity_question("ฉันคือใคร")  # "who am I" — broken
    assert self_identity_question("ผมชื่ออะไรนะครับ")  # "what is my name"
    assert self_identity_question("อ้าว เอ่อ ฉัน คือ ใคร")  # spaces between


def test_asking_about_the_caller_is_not_flagged():
    assert not self_identity_question("คุณคือใครคะ")  # asking the caller — fine
    assert not self_identity_question("ฉันไม่รู้ว่าคุณคือใครนะคะ")  # about the caller
    assert not self_identity_question("ฉันเป็นอะไรไปเนี่ย")  # "what's wrong with me", not identity
