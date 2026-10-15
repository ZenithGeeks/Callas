"""The notebook export (export.py, issue #54).

The notebook is the contract, so these tests run the notebook's own cells (its input format,
loading, cleaning, checks and label mapping) on the exported csv instead of a copy of them.
"""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from data_pipeline import export, synthesize
from data_pipeline.store import Store

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("scipy")

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "scam_classifier_phayathaibert.ipynb"

# ---- a store with calls we choose ------------------------------------------------------------

GOOD_A = ["สวัสดีครับ ผมโทรมาจากธนาคาร", "ต้องยืนยันตัวตนก่อนครับ", "บอกรหัสที่ส่งเข้ามือถือหน่อยครับ"]
GOOD_B = ["ใครครับ มีอะไรหรือเปล่า", "ผมไม่แน่ใจครับ", "ไม่บอกครับ ผู้โทรคนนี้โทรมาบ่อยมาก"]


def default_texts():
    return [t for pair in zip(GOOD_A, GOOD_B, strict=True) for t in pair]


def make_call(
    call_id="phishing_otp_00000",
    label_type="phishing_otp",
    texts=None,
    speakers=None,
    status="accepted",
    review="approved",
    scenario="bank_security",
    scam_start=2,
):
    """A call record for Store.save. Turns alternate A, B unless `speakers` says otherwise."""
    texts = texts or default_texts()
    speakers = speakers or ["AB"[i % 2] for i in range(len(texts))]
    scam = label_type not in (None, "benign")
    return {
        "id": call_id,
        "label_scam": scam,
        "label_type": label_type if scam else None,
        "scam_start_turn": scam_start if scam else None,
        "audio_path": None,
        "reference_summary": "",
        "generation": {
            "status": status,
            "reasons": [],
            "call": {"call_type": label_type or "benign", "seed": 0, "scenario": {"key": scenario}},
            "turns": [
                {"speaker": s, "text": t, "audio": f"audio/{call_id}/{i:02d}_{s}.wav"}
                for i, (s, t) in enumerate(zip(speakers, texts, strict=True))
            ],
        },
        "review": {"status": review, "note": "", "reviewed_at": None},
    }


def build_store(tmp_path, *calls, name="calls.db"):
    store = Store(tmp_path / name)
    for call in calls:
        store.save(call)
    store.close()
    return tmp_path / name


def run_export(db, tmp_path, text="script", asr=None):
    out = tmp_path / "out.csv"
    written, skipped = export.export_notebook(db, out, text, asr)
    return out, written, skipped


def read(out):
    return pd.read_csv(out, encoding="utf-8-sig", keep_default_na=False, dtype=str)


# ---- the notebook's own cells ----------------------------------------------------------------


def notebook_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return [("".join(c["source"])) for c in cells if c["cell_type"] == "code"]


def cell(marker):
    found = [s for s in notebook_cells() if marker in s]
    assert len(found) == 1, f"{marker!r} is in {len(found)} notebook cells"
    return found[0]


def run_notebook_cleaning(tmp_path, monkeypatch, generated, original_rows=None):
    """Run the notebook's load, clean, check and label-mapping cells (steps 1 to 4) with the
    hand-written data in `original_rows` and the exported csv `generated`. Returns the
    namespace after each stage as a dict of snapshots."""
    import matplotlib

    matplotlib.use("Agg")
    root = tmp_path / "nbroot"
    (root / "dataset").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    original_rows = original_rows or [
        ("ผู้โทร: สวัสดีครับ จากธนาคาร ผู้รับ: ครับ ผู้โทร: ขอรหัส otp ครับ ผู้รับ: ไม่ให้ครับ",
         "Scam", "Bank OTP Phishing"),
        ("ผู้โทร: สวัสดีค่ะ จากคลินิก ผู้รับ: ค่ะ ผู้โทร: ยืนยันนัดค่ะ ผู้รับ: รับทราบค่ะ",
         "Not a Scam", "Legit Clinic Call"),
    ]  # fmt: skip
    pd.DataFrame(original_rows, columns=["dialogue", "scam_class", "scam_type"]).to_csv(
        root / "dataset" / "scam_dataset.csv", index=False, encoding="utf-8-sig"
    )
    (root / "dataset" / "scam_dataset_generated.csv").write_bytes(generated.read_bytes())
    monkeypatch.chdir(root)
    ns: dict = {}

    def run(marker):
        exec(compile(cell(marker), marker, "exec"), ns)  # noqa: S102

    run("SEED = 42")  # settings and ROOT
    run("ROLE_TAGS =")  # step 1: the input format, load_raw
    ns["EXCEL_SOURCE"] = None
    run("raw = load_raw(DATA_FILE)")
    ns["rows_loaded"] = len(ns["raw"])
    run("def has_narration")
    run("def clean_turns_mapped")
    run("after = pd.DataFrame")  # step 3 checks: drops what fails them
    ns["rows_after_checks"] = len(ns["df"])
    run("TYPE_MAP: dict")
    ns["plt"].close("all")  # the cells draw charts
    return ns


def one_of_each_store(tmp_path):
    # the notebook drops calls with the same text and different labels, so each text differs
    calls = [
        make_call(f"{t}_00000", t, scenario=f"{t}_s", texts=[f"{x} ({t})" for x in default_texts()])
        for t in export.TYPE_NAMES
    ]
    calls.append(make_call("benign_00000", "benign", scenario="family"))
    return build_store(tmp_path, *calls)


# ---- the export passes the notebook ----------------------------------------------------------


def test_exported_csv_passes_the_notebooks_load_clean_and_map_steps(tmp_path, monkeypatch):
    db = one_of_each_store(tmp_path)
    out, written, skipped = run_export(db, tmp_path)
    assert written == 6 and not skipped

    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)

    assert ns["rows_loaded"] == 2 + 6  # nothing dropped at load
    assert ns["rows_after_checks"] == 2 + 6  # 0 rows dropped by the step 3 checks
    df = ns["df"]
    simulated = df[df["source"] == "simulated"].copy()
    assert len(simulated) == 6
    assert (simulated["speaker_turns"].map(len) == 6).all()  # every turn survived the cleaning
    assert set(simulated["scam_class"]) == {"Scam", "Not a Scam"}
    assert set(simulated["label_scam"]) == {True, False}
    # the five spec types map to themselves, normal calls to no type
    typed = simulated[simulated["label_scam"]]
    assert dict(zip(typed["scam_type"], typed["label_type"], strict=True)) == {
        t: t for t in export.TYPE_NAMES
    }
    assert simulated.loc[~simulated["label_scam"], "label_type"].isna().all()
    # the noun in a receiver turn is kept as spoken
    assert simulated["text"].str.contains("ผู้โทรคนนี้").all()


def test_the_exported_text_is_what_was_reviewed(tmp_path, monkeypatch):
    call = make_call()
    db = build_store(tmp_path, call)
    out, _, _ = run_export(db, tmp_path)
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    df = ns["df"]
    got = df[df["source"] == "simulated"]["speaker_turns"].iat[0]
    from nlp_core.text import normalize_text

    assert [t["text"] for t in got] == [
        normalize_text(t["text"]) for t in call["generation"]["turns"]
    ]
    assert [t["speaker"] for t in got] == ["A", "B"] * 3


def test_copied_cleaning_rules_match_the_notebook(tmp_path, monkeypatch):
    db = build_store(tmp_path, make_call())
    out, _, _ = run_export(db, tmp_path)
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    for name in ("SPEAKER_TAG", "BRACKET_NOTE", "STAGE_NOTE", "NARRATION"):
        assert getattr(export, name).pattern == ns[name].pattern, name
    assert export.LABEL_NAMES.pattern == ns["LABEL_NAMES"]
    assert export.TAGS["A"] == ns["CALLER_TAG"]
    assert {export.TAGS["A"]: "caller", export.TAGS["B"]: "receiver"} == ns["ROLE_TAGS"]
    assert set(export.TYPE_NAMES.values()) <= {k for k, v in ns["TYPE_MAP"].items() if k == v}


# ---- refusals --------------------------------------------------------------------------------

HAZARDS = {
    # reason: (speaker whose turn it is, text)
    export.SPEAKER_TAG_IN_TURN: ("B", "เขาพูดว่า ผู้โทร: โอนเงินเดี๋ยวนี้"),
    export.BRACKET_IN_TURN: ("B", "ไม่เอาครับ [วางสาย]"),
    export.STAGE_NOTE_IN_TURN: ("B", "ฮือ (เสียงร้องไห้) ไม่รู้จะทำยังไง"),
    export.CALLER_NOUN: ("A", "ผมเป็นผู้โทรจากธนาคารครับ รบกวนยืนยันตัวตน"),
    export.LABEL_NAME: ("B", "นี่มันคือ Phishing ใช่ไหม"),
}


def damaged_by_notebook(ns, speaker, text):
    """True if the notebook's cleaning changes, or its checks reject, a call with this turn."""
    turns = [("A", text), ("B", "ครับ")] if speaker == "A" else [("A", "สวัสดีครับ"), ("B", text)]
    cleaned = ns["clean_turns"](export.render_dialogue(turns))
    rendered = ns["render"](ns["speaker_turns"](cleaned))
    expected = [(export.ROLES[export.TAGS[s]], export.normalize_text(t)) for s, t in turns]
    return (
        cleaned != expected
        or re.search(ns["LABEL_NAMES"], rendered) is not None
        or re.search(r"[\[\]]", rendered) is not None
        or re.search(ns["ROLE_TAG_LEFT"], rendered) is not None
    )


@pytest.mark.parametrize("reason", list(HAZARDS))
def test_a_turn_the_notebook_would_damage_gets_the_call_skipped(reason, tmp_path, monkeypatch):
    speaker, bad = HAZARDS[reason]
    texts = ["สวัสดีครับ", "ครับ", bad, "ครับ"] if speaker == "A" else ["สวัสดีครับ", bad]
    bad_call = make_call("phishing_otp_00001", texts=texts)
    db = build_store(tmp_path, make_call(), bad_call)
    out, written, skipped = run_export(db, tmp_path)
    assert written == 1
    assert skipped == {reason: 1}
    assert list(read(out)["call_id"]) == ["phishing_otp_00000"]

    # and the hazard is real: the notebook does damage this text
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    assert damaged_by_notebook(ns, speaker, bad)


def test_a_lone_bracket_gets_the_call_skipped(tmp_path, monkeypatch):
    """No pair for the notebook to cut, but its step 3 check drops a text that still has one."""
    out, written, skipped = run_export(
        build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ไม่เอา ] ครับ"])), tmp_path
    )
    assert written == 0 and skipped == {export.BRACKET_IN_TURN: 1}
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    assert damaged_by_notebook(ns, "B", "ไม่เอา ] ครับ")


def test_a_hazard_hidden_by_an_invisible_character_is_found(tmp_path, monkeypatch):
    """The notebook cuts the caller word after normalizing, which removes the invisible mark."""
    bad = "ผมเป็นผู้โท​รจากธนาคารครับ"
    texts = ["สวัสดีครับ", "ครับ", bad, "ครับ"]
    out, written, skipped = run_export(build_store(tmp_path, make_call(texts=texts)), tmp_path)
    assert written == 0 and skipped == {export.CALLER_NOUN: 1}
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    assert damaged_by_notebook(ns, "A", bad)


def test_the_noun_in_a_receiver_turn_is_not_a_hazard(tmp_path, monkeypatch):
    call = make_call(texts=["สวัสดีครับ", "ผู้โทรเป็นใครครับ", "ผมเป็นเจ้าหน้าที่ครับ", "ครับ"])
    out, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 1 and not skipped
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    assert not damaged_by_notebook(ns, "B", "ผู้โทรเป็นใครครับ")


def test_an_empty_turn_gets_the_call_skipped(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "   ", "ครับ", "ครับ"])
    out, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.EMPTY_TURN: 1}


def test_a_turn_that_is_only_invisible_characters_counts_as_empty(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "​​", "ครับ", "ครับ"])
    _, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.EMPTY_TURN: 1}


def test_a_speaker_who_talks_twice_in_a_row_is_caught_by_the_final_check(tmp_path, monkeypatch):
    call = make_call(texts=["สวัสดีครับ", "ครับ", "ผมโทรมาจากธนาคาร", "ต้องยืนยันตัวตน"],
                     speakers=["A", "B", "A", "A"])  # fmt: skip
    out, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.CLEANING_CHANGES_CALL: 1}
    # the notebook would have merged the two turns into one
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    merged = ns["clean_turns"](export.render_dialogue([("A", "x"), ("A", "y"), ("B", "z")]))
    assert len(merged) == 2


def test_a_call_with_one_turn_is_skipped(tmp_path):
    call = make_call(texts=["สวัสดีครับ"])
    _, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.TOO_FEW_TURNS: 1}


def test_an_unknown_speaker_is_skipped(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ"], speakers=["A", "C"])
    _, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.UNKNOWN_SPEAKER: 1}


def test_a_scam_with_a_type_outside_the_five_is_skipped(tmp_path):
    call = make_call(label_type="wire_fraud")
    _, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 0 and skipped == {export.UNKNOWN_TYPE: 1}


# ---- which calls are written ----------------------------------------------------------------


def test_only_accepted_and_approved_calls_are_written(tmp_path):
    db = build_store(
        tmp_path,
        make_call("phishing_otp_00000"),
        make_call("phishing_otp_00001", review="pending"),
        make_call("phishing_otp_00002", review="rejected"),
        make_call("phishing_otp_00003", status="rejected"),
        make_call("phishing_otp_00004", status="in_progress"),
        make_call("phishing_otp_00005", status="rejected", review="approved"),
    )
    out, written, skipped = run_export(db, tmp_path)
    assert written == 1
    assert list(read(out)["call_id"]) == ["phishing_otp_00000"]
    assert skipped == {
        export.REVIEW_PENDING: 1,
        export.REVIEW_REJECTED: 1,
        export.NOT_ACCEPTED: 3,
    }


def test_the_store_is_not_changed(tmp_path):
    db = one_of_each_store(tmp_path)

    def dump():
        con = sqlite3.connect(db)
        try:
            return list(con.iterdump())
        finally:
            con.close()

    before = dump()
    run_export(db, tmp_path)
    assert dump() == before


# ---- the file ---------------------------------------------------------------------------------


def test_the_csv_is_utf8_with_a_bom_and_lf_lines(tmp_path):
    db = build_store(tmp_path, make_call())
    out, _, _ = run_export(db, tmp_path)
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert "ผู้โทร:" in raw.decode("utf-8-sig")
    assert pd.read_csv(out, encoding="utf-8-sig").columns[0] == "dialogue"


def test_columns_and_tags(tmp_path):
    db = one_of_each_store(tmp_path)
    out, _, _ = run_export(db, tmp_path)
    frame = read(out)
    assert list(frame.columns) == [
        "dialogue",
        "scam_class",
        "scam_type",
        "source",
        "group_hint",
        "scam_start_turn",
        "call_id",
    ]
    assert set(frame["source"]) == {"simulated"}
    first = frame["dialogue"].iat[0]
    assert first.startswith("ผู้โทร: ") and " ผู้รับ: " in first
    assert "A:" not in first and "B:" not in first
    benign = frame[frame["call_id"] == "benign_00000"].iloc[0]
    assert benign["scam_class"] == "Not a Scam"
    assert benign["scam_type"] == "" and benign["scam_start_turn"] == ""
    scam = frame[frame["call_id"] == "romance_00000"].iloc[0]
    assert scam["scam_class"] == "Scam" and scam["scam_type"] == "romance"
    assert scam["scam_start_turn"] == "2"


def test_group_hint_is_the_call_type_and_scenario(tmp_path):
    db = build_store(
        tmp_path,
        make_call("phishing_otp_00000", scenario="bank_security"),
        make_call("phishing_otp_00001", scenario="bank_security"),
        make_call("phishing_otp_00002", scenario="parcel_link"),
        make_call("romance_00000", "romance", scenario="bank_security"),
        make_call("benign_00000", "benign", scenario="family"),
    )
    out, _, _ = run_export(db, tmp_path)
    hints = dict(zip(read(out)["call_id"], read(out)["group_hint"], strict=True))
    assert (
        hints["phishing_otp_00000"] == hints["phishing_otp_00001"] == "phishing_otp/bank_security"
    )
    assert hints["phishing_otp_00002"] == "phishing_otp/parcel_link"
    assert hints["romance_00000"] == "romance/bank_security"  # same key, other call type
    assert hints["benign_00000"] == "benign/family"


def test_group_hint_falls_back_to_the_call_id(tmp_path):
    call = make_call()
    del call["generation"]["call"]["scenario"]
    out, _, _ = run_export(build_store(tmp_path, call), tmp_path)
    assert read(out)["group_hint"].iat[0] == "phishing_otp_00000"


def test_rows_sharing_a_group_hint_share_a_notebook_group(tmp_path, monkeypatch):
    """The notebook's step 5 cell, run on rows whose text is too different to be linked."""
    pytest.importorskip("sklearn")
    texts = [
        ["สวัสดีครับ ผมจากธนาคารกสิกร", "ครับ มีอะไร", "ขอรหัสโอทีพี", "ไม่ให้ครับ"],
        ["ฮัลโหล คุณถูกรางวัลที่หนึ่ง", "จริงเหรอคะ", "ต้องโอนค่าธรรมเนียม", "ไม่โอนค่ะ"],
        ["ผมเป็นตำรวจสอบสวนกลาง", "อะไรนะ", "บัญชีคุณเกี่ยวกับฟอกเงิน", "ไม่จริง"],
    ]
    calls = [
        make_call(f"phishing_otp_{i:05d}", texts=t, scenario="same") for i, t in enumerate(texts)
    ]
    calls.append(make_call("romance_00000", "romance", scenario="other"))
    out, _, _ = run_export(build_store(tmp_path, *calls), tmp_path)
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out)
    ns["NEAR_DUPLICATE_SIMILARITY"] = 0.99  # nothing is a near copy of anything
    ns["plt"].show = lambda *a, **k: None
    exec(compile(cell("from scipy.sparse import csr_matrix"), "groups", "exec"), ns)  # noqa: S102
    df = ns["df"]
    same = df[df["group_hint"] == "phishing_otp/same"]
    assert len(same) == 3 and same["group"].nunique() == 1
    assert df[df["group_hint"] == "romance/other"]["group"].iat[0] != same["group"].iat[0]
    # the hand-written rows have no hint and stay apart
    assert df[df["source"] == "original"]["group"].nunique() == 2


# ---- leak columns from another branch ---------------------------------------------------------


def test_leak_columns_are_exported_when_the_store_has_them(tmp_path):
    db = build_store(tmp_path, make_call("phishing_otp_00000"), make_call("benign_00000", "benign"))
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE calls ADD COLUMN receiver_leaked INTEGER")
    con.execute("ALTER TABLE calls ADD COLUMN first_leak_turn INTEGER")
    con.execute("UPDATE calls SET receiver_leaked = 1, first_leak_turn = 3 WHERE id LIKE 'phish%'")
    con.execute("UPDATE calls SET receiver_leaked = 0 WHERE id LIKE 'benign%'")
    con.commit()
    con.close()
    frame = read(run_export(db, tmp_path)[0]).set_index("call_id")
    assert frame.loc["phishing_otp_00000", "receiver_leaked"] == "1"
    assert frame.loc["phishing_otp_00000", "first_leak_turn"] == "3"
    assert frame.loc["benign_00000", "receiver_leaked"] == "0"
    assert frame.loc["benign_00000", "first_leak_turn"] == ""


def test_a_store_without_leak_columns_exports_without_them(tmp_path):
    out, written, _ = run_export(build_store(tmp_path, make_call()), tmp_path)
    assert written == 1
    assert not {"receiver_leaked", "first_leak_turn"} & set(read(out).columns)


# ---- --text asr -------------------------------------------------------------------------------


def asr_with_cache(tmp_path, cached):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "typhoon_asr.json").write_text(json.dumps(cached, ensure_ascii=False), "utf-8")
    return export.AsrText(tmp_path, cache_dir)


def test_asr_text_replaces_the_script_where_the_cache_has_it(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ", "ผมโทรมาจากธนาคาร", "ไม่เอาครับ"])
    db = build_store(tmp_path, call)
    asr = asr_with_cache(
        tmp_path, {"phishing_otp_00000/0": "สวัสดีคับ", "phishing_otp_00000/3": "ไม่เอาคับ"}
    )
    out, written, _ = run_export(db, tmp_path, "asr", asr)
    assert written == 1
    dialogue = read(out)["dialogue"].iat[0]
    assert dialogue == "ผู้โทร: สวัสดีคับ ผู้รับ: ครับ ผู้โทร: ผมโทรมาจากธนาคาร ผู้รับ: ไม่เอาคับ"
    assert asr.fallback == 2  # turns 1 and 2 had no transcript and no audio file


def test_script_text_is_used_unless_asr_is_asked_for(tmp_path):
    db = build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ครับ"]))
    asr = asr_with_cache(tmp_path, {"phishing_otp_00000/0": "สวัสดีคับ"})
    out, _, _ = run_export(db, tmp_path, "script", asr)
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: สวัสดีครับ ผู้รับ: ครับ"


def test_asr_text_goes_through_the_same_refusals(tmp_path):
    db = build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ครับ"]))
    asr = asr_with_cache(tmp_path, {"phishing_otp_00000/1": "ครับ [เสียงรบกวน]"})
    _, written, skipped = run_export(db, tmp_path, "asr", asr)
    assert written == 0 and skipped == {export.BRACKET_IN_TURN: 1}


def test_a_missing_transcript_is_made_from_the_audio_when_pythaiasr_is_there(tmp_path, monkeypatch):
    audio = tmp_path / "audio" / "phishing_otp_00000"
    audio.mkdir(parents=True)
    (audio / "00_A.wav").write_bytes(b"RIFF")
    db = build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ครับ"]))

    class FakeASR:
        def __init__(self, model, device):
            self.model = model

        def __call__(self, path):
            assert path.endswith("00_A.wav")
            return "สวัสดีค่ะ"

    import sys
    import types

    monkeypatch.setitem(sys.modules, "pythaiasr", types.SimpleNamespace(ASR=FakeASR))
    asr = asr_with_cache(tmp_path, {})
    out, _, _ = run_export(db, tmp_path, "asr", asr)
    asr.save()
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: สวัสดีค่ะ ผู้รับ: ครับ"
    assert json.loads(asr.cache_path.read_text("utf-8")) == {"phishing_otp_00000/0": "สวัสดีค่ะ"}


# ---- the command ------------------------------------------------------------------------------


def test_the_command_writes_the_csv_and_prints_a_summary(tmp_path, capsys):
    db = build_store(
        tmp_path,
        make_call("phishing_otp_00000"),
        make_call("phishing_otp_00001", review="pending"),
        make_call("phishing_otp_00002", texts=["ครับ [วางสาย]", "ครับ"]),
    )
    out = tmp_path / "sub" / "notebook.csv"
    assert synthesize.main(["--export-notebook", str(out), "--store", str(db)]) == 0
    text = capsys.readouterr().out
    assert "wrote 1 rows" in text
    assert "review_pending" in text and "bracket_in_turn" in text
    assert len(read(out)) == 1


def test_the_command_reads_the_db_option_when_there_is_no_store_option(tmp_path, capsys):
    db = build_store(tmp_path, make_call())
    out = tmp_path / "o.csv"
    assert synthesize.main(["--export-notebook", str(out), "--db", str(db)]) == 0
    assert len(read(out)) == 1


def test_the_command_says_so_when_there_is_no_store(tmp_path, capsys):
    code = synthesize.main(
        ["--export-notebook", str(tmp_path / "o.csv"), "--store", str(tmp_path / "none.db")]
    )
    assert code == 2 and "no store at" in capsys.readouterr().out
    assert not (tmp_path / "none.db").exists()  # looking must not create an empty store
