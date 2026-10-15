"""Pure helpers of scripts/eval_autocorrect.py. No models, GPU or audio."""

import pytest

from scripts.eval_autocorrect import (
    align,
    correctness,
    edit_distance,
    entity_key,
    error_rates,
    mistakes,
    percentile,
    prf,
    prf_by_label,
    score_correction,
)


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["a", "b"], ["a", "c", "b"]) == 1
    assert edit_distance([], ["a"]) == 1


def test_error_rates_use_corpus_totals():
    pairs = [(["ab", "cd"], ["ab", "xx"]), (["ef"], ["ef"])]
    assert error_rates(pairs, chars=False) == pytest.approx(1 / 3)
    assert error_rates(pairs, chars=True) == pytest.approx(2 / 6)
    assert error_rates([], chars=True) == 0.0


def test_align_and_mistakes():
    ref = ["a", "b", "c", "d"]
    hyp = ["a", "x", "d", "e"]
    tags = [t for t, *_ in align(ref, hyp)]
    assert tags == ["equal", "replace", "equal", "insert"]
    found = mistakes(ref, hyp)
    assert [(m["ref"], m["asr"]) for m in found] == [("bc", "x"), ("", "e")]
    assert correctness(ref, hyp) == [True, False, True, False]


def test_score_correction_counts_fixed_left_and_broken():
    ref = ["ก", "ข", "ค", "ง", "จ", "ฉ", "ช"]
    asr = ["ก", "x1", "ค", "ง", "x2", "ฉ", "ช"]
    # x1 becomes ข (fixed), x2 is untouched (left), ง becomes ฌ (a correct word broken)
    fixed = ["ก", "ข", "ค", "ฌ", "x2", "ฉ", "ช"]
    got = score_correction(ref, asr, fixed)
    assert [m["ref"] for m in got["fixed"]] == ["ข"]
    assert got["changed_wrong"] == []
    assert [m["ref"] for m in got["left"]] == ["จ"]
    assert got["broken"] == [{"was": "ง", "now": "ฌ"}]


def test_score_correction_changed_but_still_wrong_and_dropped_words():
    ref = ["a", "b", "c"]
    asr = ["a", "x"]  # b is wrong and c is missing
    fixed = ["a", "y"]
    got = score_correction(ref, asr, fixed)
    assert [m["asr"] for m in got["changed_wrong"]] == ["x"]
    assert got["fixed"] == []
    assert got["broken"] == []


def test_prf_counts_repeats_and_empty_sides():
    gold = [("PHONE", "0812345678"), ("MONEY", "500บาท"), ("MONEY", "500บาท")]
    pred = [("PHONE", "0812345678"), ("MONEY", "500บาท"), ("URL", "a.com")]
    s = prf(gold, pred)
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 1)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert prf([], [])["f1"] == 0.0
    by = prf_by_label(gold, pred)
    assert by["PHONE"]["f1"] == 1.0
    assert by["URL"]["precision"] == 0.0
    assert by["overall"]["tp"] == 2


def test_entity_key_ignores_thai_digits_commas_and_spaces():
    assert entity_key("MONEY", "๕,๐๐๐ บาท") == ("MONEY", "5000บาท")
    assert entity_key("BANK_ACCOUNT", "158-5-96045-2") == ("BANK_ACCOUNT", "1585960452")


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    assert percentile([], 0.95) == 0.0
