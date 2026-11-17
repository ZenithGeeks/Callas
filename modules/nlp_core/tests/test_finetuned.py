import json

import pytest

from nlp_core import registry
from nlp_core.finetuned import mask, model_text, read_preprocessing

# What the current notebook writes to preprocessing.json.
PRE = {
    "row_text": "call_so_far",
    "speaker_letters": {"receiver": "A", "caller": "B"},
    "masked_labels": ["PHONE", "BANK_ACCOUNT", "MONEY"],
    "space_token": "<_>",
    "max_length": 256,
}


def word_count(text: str) -> int:
    return len(text.split("<_>"))


def test_live_speakers_get_the_training_role_letters():
    text = "Caller: สวัสดีครับ\nagent: ค่ะ ใครคะ"
    assert model_text(text, PRE, word_count) == "B:<_>สวัสดีครับ<_>A:<_>ค่ะ<_>ใครคะ"


def test_other_speaker_names_pass_through():
    assert model_text("A: ฮัลโหล", PRE, word_count) == "A:<_>ฮัลโหล"


def test_sensitive_values_are_masked_like_the_training_rows():
    row = model_text("Caller: โทรกลับเบอร์ 081-234-5678 นะครับ", PRE, word_count)
    assert row == "B:<_>โทรกลับเบอร์<_>[PHONE]<_>นะครับ"


def test_only_the_trained_labels_are_masked():
    assert mask("เบอร์ 081-234-5678", ["MONEY"]) == "เบอร์ 081-234-5678"


def test_lines_without_a_speaker_are_left_out():
    text = "สรุปบทสนทนาก่อนหน้า\nler: ส่วนที่ถูกตัด\nCaller: โอนเงินมาเลย"
    assert model_text(text, PRE, word_count) == "B:<_>โอนเงินมาเลย"


def test_oldest_lines_are_cut_and_the_newest_stays():
    text = "Caller: หนึ่ง สอง สาม\nagent: สี่ ห้า\nCaller: หก"
    pre = {**PRE, "max_length": 5}
    assert model_text(text, pre, word_count) == "A:<_>สี่<_>ห้า<_>B:<_>หก"


def test_an_uncased_model_gets_a_lowercased_row():
    pre = {**PRE, "lowercase": True}
    row = model_text("Caller: โทรกลับเบอร์ 081-234-5678 ทาง LINE", pre, word_count)
    assert row == "b:<_>โทรกลับเบอร์<_>[phone]<_>ทาง<_>line"


def test_fine_tuned_names_give_a_gate_and_a_type_encoder():
    assert registry.fine_tuned_pair("best") == registry.BEST_FINE_TUNED
    assert registry.fine_tuned_pair("bge_m3") == ("bge_m3", "bge_m3")
    assert registry.fine_tuned_pair("wangchanberta+bge_m3") == ("wangchanberta", "bge_m3")
    assert registry.fine_tuned_pair("llm") is None
    assert registry.fine_tuned_pair("bge_m3+llm") is None
    assert "bge_m3+phayathaibert" in registry.available("classifier")


def test_a_newest_line_over_the_limit_is_kept():
    pre = {**PRE, "max_length": 1}
    assert model_text("agent: ค่ะ\nCaller: ยาว มาก", pre, word_count) == "B:<_>ยาว<_>มาก"


def write_pre(tmp_path, pre: dict):
    (tmp_path / "preprocessing.json").write_text(json.dumps(pre), encoding="utf-8")
    return tmp_path


def test_reads_a_current_preprocessing_file(tmp_path):
    assert read_preprocessing(write_pre(tmp_path, PRE)) == PRE


def test_a_model_from_an_older_notebook_is_refused(tmp_path):
    old = {"input_format": "...", "space_token": "<_>", "max_length": 256}
    with pytest.raises(ValueError, match="older notebook"):
        read_preprocessing(write_pre(tmp_path, old))


def test_a_one_line_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="call so far"):
        read_preprocessing(write_pre(tmp_path, {**PRE, "row_text": "one_line"}))


def test_a_missing_model_names_the_notebook(tmp_path):
    with pytest.raises(FileNotFoundError, match="scam_classifier_phayathaibert.ipynb"):
        read_preprocessing(tmp_path)
