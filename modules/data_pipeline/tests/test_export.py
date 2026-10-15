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
    columns = ["dialogue", "scam_class", "scam_type", "group_hint"][: len(original_rows[0])]
    pd.DataFrame(original_rows, columns=columns).to_csv(
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
        "receiver_leaked",
        "first_leak_turn",
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
    assert df[df["source"] == ns["ORIGINAL_SOURCE"]]["group"].nunique() == 2


def test_a_group_hint_on_hand_written_rows_links_nothing(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    bank = export.render_dialogue([("A", "สวัสดีครับ จากธนาคาร"), ("B", "ครับ")])
    clinic = export.render_dialogue([("A", "ยืนยันนัดทันตกรรมค่ะ"), ("B", "ค่ะ")])
    rows = [
        (bank, "Scam", "Bank OTP Phishing", "h"),
        (clinic, "Not a Scam", "Legit Call", "h"),
    ]
    out, _, _ = run_export(build_store(tmp_path, make_call()), tmp_path)
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out, rows)
    ns["plt"].show = lambda *a, **k: None
    exec(compile(cell("from scipy.sparse import csr_matrix"), "groups", "exec"), ns)  # noqa: S102
    original = ns["df"][ns["df"]["source"] == ns["ORIGINAL_SOURCE"]]
    assert len(original) == 2 and original["group"].nunique() == 2


# ---- the notebook's split with generated rows -------------------------------------------------

THEMES = {
    "bank": [
        "สวัสดีครับ ผมโทรจากธนาคารกรุงไทยเรื่องบัตรเครดิต",
        "ครับ มีอะไรครับ",
        "บัตรถูกระงับ ต้องยืนยันรหัสหลังบัตรก่อน",
        "ไม่ให้ครับ",
    ],
    "police": [
        "ผมพันตำรวจเอกจากกองปราบ บัญชีคุณเกี่ยวข้องกับคดีฟอกเงิน",
        "อะไรนะคะ ฉันไม่ได้ทำ",
        "ต้องโอนเงินมาตรวจสอบภายในวันนี้เท่านั้น",
        "ไม่โอนค่ะ",
    ],
    "lottery": [
        "ยินดีด้วยครับ คุณได้รับรางวัลสลากกินแบ่งรัฐบาลรางวัลที่หนึ่ง",
        "จริงหรือคะ",
        "ต้องโอนค่าธรรมเนียมรับรางวัลสามพันบาท",
        "ขอคิดดูก่อนค่ะ",
    ],
    "clinic": [
        "สวัสดีค่ะ โทรจากคลินิกทันตกรรม ยืนยันนัดขูดหินปูนวันเสาร์",
        "ได้ค่ะ ไปตามนัด",
        "ขอบคุณค่ะ แล้วพบกันนะคะ",
        "สวัสดีค่ะ",
    ],
    "delivery": [
        "พัสดุของคุณมาถึงแล้วครับ จะส่งช่วงบ่ายนี้",
        "ผมอยู่บ้านครับ",
        "รบกวนเตรียมเงินสดไว้ด้วยนะครับ",
        "ได้ครับ",
    ],
    "family": ["แม่ครับ ผมเองนะ วันอาทิตย์กลับไปกินข้าวที่บ้านนะ", "ได้จ้ะ จะทำต้มยำให้", "ขอบคุณครับแม่", "เจอกันจ้ะ"],
}


def themed(name, extra=""):
    turns = list(THEMES[name])
    turns[-1] += extra
    return turns


def split_fixture(tmp_path, monkeypatch, capsys):
    """Hand-written rows with a saved split, plus generated calls that are (or are not) near
    copies of them. Runs the notebook's cells up to and including the split."""
    pytest.importorskip("sklearn")
    saved = {"bank": "test", "police": "train", "lottery": "valid", "lottery2": "train",
             "clinic": "valid", "delivery": "train"}  # fmt: skip
    originals, kinds = [], {}
    for key in saved:
        theme = key.rstrip("2")
        turns = themed(theme, " ครับผม" if key == "lottery2" else "")
        scam = theme in ("bank", "police", "lottery", "delivery")
        originals.append(
            (
                " ".join(f"{export.TAGS['AB'[i % 2]]}: {t}" for i, t in enumerate(turns)),
                "Scam" if scam else "Not a Scam",
                "Bank OTP Phishing" if scam else "Legit Clinic Call",
            )
        )
        kinds[key] = turns
    calls = [
        make_call("phishing_otp_00000", texts=themed("bank", " ค่ะ"), scenario="bank"),  # near test
        make_call("phishing_otp_00001", texts=themed("police", " ค่ะ"), scenario="police"),
        make_call("phishing_otp_00002", texts=themed("lottery", " ค่ะ"), scenario="lottery"),
        make_call("benign_00000", "benign", texts=themed("family"), scenario="family"),
    ]
    out, written, skipped = run_export(build_store(tmp_path, *calls), tmp_path)
    assert written == 4 and not skipped
    ns = run_notebook_cleaning(tmp_path, monkeypatch, out, originals)
    ns["plt"].show = lambda *a, **k: None
    exec(compile(cell("from scipy.sparse import csr_matrix"), "groups", "exec"), ns)  # noqa: S102
    df = ns["df"]
    processed = tmp_path / "nbroot" / "data" / "processed"
    processed.mkdir(parents=True)
    by_key = dict(zip(saved, df[df["source"] == ns["ORIGINAL_SOURCE"]]["text"], strict=True))
    for name in ("train", "valid", "test"):
        texts = [by_key[k] for k, v in saved.items() if v == name]
        pd.DataFrame({"text": texts}).to_csv(
            processed / f"{name}.csv", index=False, encoding="utf-8-sig"
        )
    ns["PROCESSED"] = processed
    capsys.readouterr()
    exec(compile(cell("def group_stratified_split"), "split", "exec"), ns)  # noqa: S102
    ns["plt"].close("all")
    return ns, saved, capsys.readouterr().out


def split_of(ns, theme_words):
    df = ns["df"]
    return df[df["text"].str.contains(theme_words, regex=False)][["source", "split"]]


def test_hand_written_rows_keep_their_saved_split_and_generated_rows_follow_them(
    tmp_path, monkeypatch, capsys
):
    ns, saved, shown = split_fixture(tmp_path, monkeypatch, capsys)
    df = ns["df"]
    original = df[df["source"] == ns["ORIGINAL_SOURCE"]]
    assert len(original) == 6
    texts = {k: THEMES[k.rstrip("2")][0][:12] for k in saved}
    for key, expected in saved.items():
        row = original[original["text"].str.contains(texts[key], regex=False)]
        if key == "lottery":  # the lottery pair is one group in two saved splits (below)
            row = row[~row["text"].str.contains("ครับผม")]
        if key == "lottery2":
            row = row[row["text"].str.contains("ครับผม")]
        assert list(row["split"]) == [expected], key
    generated = df[df["source"] == "simulated"]
    # the copy of the police call follows its train row; the copy of the test call is dropped
    assert list(split_of(ns, "ผมพันตำรวจเอก").query("source == 'simulated'")["split"]) == ["train"]
    assert not len(split_of(ns, "ธนาคารกรุงไทย").query("source == 'simulated'"))
    # nothing generated is in test
    assert "test" not in set(generated["split"])
    assert set(generated["split"]) <= {"train", "valid"}
    assert "dropped" in shown


def test_a_generated_row_between_two_saved_splits_is_dropped_with_a_warning(
    tmp_path, monkeypatch, capsys
):
    ns, _, shown = split_fixture(tmp_path, monkeypatch, capsys)
    assert "WARNING: 1 groups link hand-written rows that are in different saved splits" in shown
    # the lottery copy was linked to a valid row and a train row, so it is not placed
    assert not len(split_of(ns, "ยินดีด้วยครับ").query("source == 'simulated'"))


def test_a_scenario_new_to_the_split_goes_to_train_or_valid(tmp_path, monkeypatch, capsys):
    ns, _, _ = split_fixture(tmp_path, monkeypatch, capsys)
    family = split_of(ns, "แม่ครับ ผมเองนะ")
    assert list(family["source"]) == ["simulated"]
    assert family["split"].iat[0] in ("train", "valid")


def test_without_a_generated_file_the_notebook_writes_the_committed_processed_files(
    tmp_path, monkeypatch
):
    """Part A on the repo's own data must give back data/processed byte for byte."""
    pytest.importorskip("sklearn")
    import shutil

    import matplotlib

    matplotlib.use("Agg")
    repo = NOTEBOOK.parents[1]
    root = tmp_path / "root"
    (root / "dataset").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    shutil.copy(repo / "dataset" / "scam_dataset.csv", root / "dataset")
    (root / "data" / "processed").mkdir(parents=True)
    shutil.copy(repo / "data" / "processed" / "test.csv", root / "data" / "processed")
    monkeypatch.chdir(root)
    ns: dict = {}
    for source in notebook_cells():
        exec(compile(source, "part A", "exec"), ns)  # noqa: S102
        if "write_records(path, records[name])" in source:
            break
    ns["plt"].close("all")
    for name in ("train", "valid", "test", "dataset"):
        made = (root / "data" / "processed" / f"{name}.csv").read_bytes()
        assert made == (repo / "data" / "processed" / f"{name}.csv").read_bytes(), name


# ---- leak columns from another branch ---------------------------------------------------------


def test_leak_columns_are_exported_when_the_store_has_them(tmp_path):
    """A store made by the current Store, which has the leak columns, with leaks saved in it."""
    leaky = make_call("phishing_otp_00000")
    leaky["generation"]["receiver_leaked"] = ["PHONE", "BANK_ACCOUNT"]
    leaky["generation"]["first_leak_turn"] = 3
    db = build_store(tmp_path, leaky, make_call("benign_00000", "benign"))
    frame = read(run_export(db, tmp_path)[0]).set_index("call_id")
    assert json.loads(frame.loc["phishing_otp_00000", "receiver_leaked"]) == [
        "PHONE",
        "BANK_ACCOUNT",
    ]
    assert frame.loc["phishing_otp_00000", "first_leak_turn"] == "3"
    assert frame.loc["benign_00000", "receiver_leaked"] == "[]"
    assert frame.loc["benign_00000", "first_leak_turn"] == ""


def test_a_store_without_leak_columns_exports_without_them(tmp_path):
    """A store from before the leak columns existed."""
    db = build_store(tmp_path, make_call())
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE calls DROP COLUMN receiver_leaked")
    con.execute("ALTER TABLE calls DROP COLUMN first_leak_turn")
    con.commit()
    con.close()
    out, written, _ = run_export(db, tmp_path)
    assert written == 1
    assert not {"receiver_leaked", "first_leak_turn"} & set(read(out).columns)


# ---- --text asr -------------------------------------------------------------------------------


def cache_entry(call, idx, text):
    turn = {**call["generation"]["turns"][idx], "idx": idx}
    return {f"{call['id']}/{idx}": {"sig": export.turn_signature(turn), "text": text}}


def asr_with_cache(tmp_path, cached):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "typhoon_asr.export.json").write_text(
        json.dumps(cached, ensure_ascii=False), "utf-8"
    )
    return export.AsrText(tmp_path, cache_dir)


def fake_pythaiasr(monkeypatch, said):
    """A pythaiasr whose ASR answers `said[file name]` and records the files it was asked for."""
    import sys
    import types

    asked = []

    class FakeASR:
        def __init__(self, model, device):
            self.model = model

        def __call__(self, path):
            asked.append(Path(path).name)
            return said[Path(path).name]

    monkeypatch.setitem(sys.modules, "pythaiasr", types.SimpleNamespace(ASR=FakeASR))
    return asked


def test_asr_text_replaces_the_script_where_the_cache_has_it(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ"])
    db = build_store(tmp_path, call)
    cached = {**cache_entry(call, 0, "สวัสดีคับ"), **cache_entry(call, 1, "คับ")}
    out, written, _ = run_export(db, tmp_path, "asr", asr_with_cache(tmp_path, cached))
    assert written == 1
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: สวัสดีคับ ผู้รับ: คับ"


def test_script_text_is_used_unless_asr_is_asked_for(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ"])
    db = build_store(tmp_path, call)
    asr = asr_with_cache(tmp_path, cache_entry(call, 0, "สวัสดีคับ"))
    out, _, _ = run_export(db, tmp_path, "script", asr)
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: สวัสดีครับ ผู้รับ: ครับ"


def test_a_call_with_a_turn_that_has_no_asr_text_is_skipped_not_mixed(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ", "ผมโทรมาจากธนาคาร", "ไม่เอาครับ"])
    db = build_store(tmp_path, call, make_call("phishing_otp_00001", texts=["สวัสดีครับ", "ครับ"]))
    asr = asr_with_cache(tmp_path, cache_entry(call, 0, "สวัสดีคับ"))  # turns 1 to 3 have none
    out, written, skipped = run_export(db, tmp_path, "asr", asr)
    assert written == 0
    assert skipped == {export.ASR_INCOMPLETE: 2}


def test_asr_text_goes_through_the_same_refusals(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ครับ"])
    db = build_store(tmp_path, call)
    cached = {**cache_entry(call, 0, "สวัสดีครับ"), **cache_entry(call, 1, "ครับ [เสียงรบกวน]")}
    _, written, skipped = run_export(db, tmp_path, "asr", asr_with_cache(tmp_path, cached))
    assert written == 0 and skipped == {export.BRACKET_IN_TURN: 1}


def test_a_missing_transcript_is_made_from_the_audio_when_pythaiasr_is_there(tmp_path, monkeypatch):
    audio = tmp_path / "audio" / "phishing_otp_00000"
    audio.mkdir(parents=True)
    for name in ("00_A.wav", "01_B.wav"):
        (audio / name).write_bytes(b"RIFF")
    db = build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ครับ"]))
    asked = fake_pythaiasr(monkeypatch, {"00_A.wav": "สวัสดีค่ะ", "01_B.wav": "ค่ะ"})
    asr = asr_with_cache(tmp_path, {})
    out, _, _ = run_export(db, tmp_path, "asr", asr)
    asr.save()
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: สวัสดีค่ะ ผู้รับ: ค่ะ"
    assert asked == ["00_A.wav", "01_B.wav"]
    saved = json.loads(asr.cache_path.read_text("utf-8"))
    assert saved["phishing_otp_00000/0"]["text"] == "สวัสดีค่ะ"


def test_a_cached_transcript_is_reused_without_asking_the_model(tmp_path, monkeypatch):
    call = make_call(texts=["สวัสดีครับ", "ครับ"])
    db = build_store(tmp_path, call)
    asked = fake_pythaiasr(monkeypatch, {})
    cached = {**cache_entry(call, 0, "สวัสดีคับ"), **cache_entry(call, 1, "คับ")}
    run_export(db, tmp_path, "asr", asr_with_cache(tmp_path, cached))
    assert asked == []


def test_a_call_regenerated_under_the_same_id_is_transcribed_again(tmp_path, monkeypatch):
    first = make_call(texts=["สวัสดีครับ", "ครับ"])
    db = build_store(tmp_path, first)
    cache = asr_with_cache(
        tmp_path, {**cache_entry(first, 0, "เก่า"), **cache_entry(first, 1, "เก่า")}
    )
    # the same call id, made again with other text and other audio
    again = make_call(texts=["ฮัลโหลครับ", "ว่าไงครับ"])
    for i, turn in enumerate(again["generation"]["turns"]):
        turn["audio"] = f"audio/phishing_otp_00000/{i:02d}_{turn['speaker']}_v2.wav"
    Store(db).save(again)
    audio = tmp_path / "audio" / "phishing_otp_00000"
    audio.mkdir(parents=True)
    for name in ("00_A_v2.wav", "01_B_v2.wav"):
        (audio / name).write_bytes(b"RIFF")
    asked = fake_pythaiasr(monkeypatch, {"00_A_v2.wav": "ใหม่หนึ่ง", "01_B_v2.wav": "ใหม่สอง"})
    out, written, _ = run_export(
        db, tmp_path, "asr", export.AsrText(tmp_path, cache.cache_path.parent)
    )
    assert written == 1
    assert read(out)["dialogue"].iat[0] == "ผู้โทร: ใหม่หนึ่ง ผู้รับ: ใหม่สอง"
    assert asked == ["00_A_v2.wav", "01_B_v2.wav"]


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
