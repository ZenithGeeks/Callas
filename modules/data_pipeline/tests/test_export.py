"""The notebook export (export.py, issue #54) and the notebook's part A.

The notebook is the contract, so these tests run the notebook's own cells (its input format,
loading, masking, cleaning, checks, label mapping, class sizes, scenario split and export)
instead of a copy of them. The notebook reads the simulated calls from dataset/*.jsonl.
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

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


def jsonl_call(
    call_id="phishing_otp_00000",
    label_type="phishing_otp",
    texts=None,
    speakers=None,
    outcome="safe",
    outcome_source="locked",
    scam_start=2,
    scenario=None,
    entities=None,
):
    """One call as a line of dataset/*.jsonl. Turns alternate A, B. `scenario` is the name (or a
    dict with "key") the file holds; `entities` are {"text", "label", "start", "end"} in text."""
    texts = texts or default_texts()
    speakers = speakers or ["AB"[i % 2] for i in range(len(texts))]
    scam = label_type not in (None, "benign")
    return {
        "id": call_id,
        "channel": "call",
        "text": "\n".join(f"{s}: {t}" for s, t in zip(speakers, texts, strict=True)),
        "audio_path": None,
        "speaker_turns": [
            {"speaker": s, "start": None, "end": None, "text": t}
            for s, t in zip(speakers, texts, strict=True)
        ],
        "label_scam": scam,
        "label_type": label_type if scam else None,
        "scam_outcome": outcome,
        "scam_outcome_source": outcome_source,
        "scenario": scenario,
        "entities": entities or [],
        "reference_summary": "",
        "scam_start_turn": scam_start if scam else None,
        "source": "synthetic",
        "receiver_leaked": [],
        "first_leak_turn": None,
    }


def as_jsonl(call: dict) -> dict:
    """A store record from make_call as the line --export would write for it."""
    turns = call["generation"]["turns"]
    return jsonl_call(
        call["id"],
        call["label_type"] if call["label_scam"] else None,
        [t["text"] for t in turns],
        [t["speaker"] for t in turns],
    )


def write_jsonl(path, calls):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in calls), encoding="utf-8"
    )


def run_notebook_cleaning(tmp_path, monkeypatch, calls=None, files=None):
    """Run the notebook's load, clean, check and label-mapping cells (steps 1 to 4) on JSONL
    calls in dataset/. `calls` go to one file; `files` maps file names to their calls. Returns
    the notebook's namespace."""
    import matplotlib

    matplotlib.use("Agg")
    root = tmp_path / "nbroot"
    (root / "dataset").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    if files is None:
        files = {
            "calls.jsonl": calls
            or [jsonl_call(), jsonl_call("benign_00000", None, themed_texts("clinic"))]
        }
    for name, rows in files.items():
        write_jsonl(root / "dataset" / name, rows)
    monkeypatch.chdir(root)
    ns: dict = {}

    def run(marker):
        exec(compile(cell(marker), marker, "exec"), ns)  # noqa: S102

    run("SEED = 42")  # settings and ROOT
    run("SPEAKER_ROLES =")  # the input format: jsonl_row, load_jsonl
    run("DATA_FILES = sorted(")  # step 1
    ns["rows_loaded"] = len(ns["raw"])
    run("spoken = raw[")  # step 2
    run("def clean_turns_mapped")
    run("after = pd.DataFrame")  # step 3 checks: drops what fails them
    ns["rows_after_checks"] = len(ns["df"])
    run("TYPE_MAP: dict")
    ns["plt"].close("all")  # the cells draw charts
    return ns


def word_count(text: str) -> int:
    """Stands in for the tokenizer of step 6b: one token per word."""
    return len(text.split())


STEP_6B_CELLS = (
    "def scam_start_of",  # the label rule and "one_line"
    "def call_so_far_rows",  # "call_so_far"
    "call_rows = {",  # build the rows with ROW_TEXT
    "longest = line_tokens",  # the token tables
)


def run_part_a_rest(ns, max_tokens=10_000, row_text=None, **settings):
    """Steps 4b to 7 on a namespace from run_notebook_cleaning. `settings` replace the values the
    class-size cell sets (SKIP_TYPES). Step 5 makes one group per scenario. Step 6b uses the
    notebook's ROW_TEXT unless `row_text` is given, and counts words instead of tokens (its
    tokenizer cell is not run), at most `max_tokens`."""
    pytest.importorskip("sklearn")
    ns["plt"].show = lambda *a, **k: None
    sizes = cell("typed_scam = ")
    for name, value in settings.items():
        line = next(x for x in sizes.splitlines() if x.startswith(f"{name}"))
        sizes = sizes.replace(line, f"{name} = {value!r}")
    exec(compile(sizes, "class sizes", "exec"), ns)  # noqa: S102
    for marker in ('df["group"] = pd.factorize(', "def scenario_split", "ROW_TEXTS = ("):
        exec(compile(cell(marker), marker, "exec"), ns)  # noqa: S102
    if row_text is not None:
        ns["ROW_TEXT"] = row_text
    ns["count_tokens"], ns["MAX_TOKENS"] = word_count, max_tokens
    for marker in (*STEP_6B_CELLS, "RECORD_KEYS = ("):
        exec(compile(cell(marker), marker, "exec"), ns)  # noqa: S102
    ns["plt"].close("all")
    return ns


THEMED = {
    "clinic": ["สวัสดีค่ะ โทรจากคลินิกทันตกรรม", "ค่ะ", "ยืนยันนัดขูดหินปูนวันเสาร์ค่ะ", "ได้ค่ะ ไปตามนัด"],
}


def themed_texts(name):
    return list(THEMED[name])


def one_of_each_store(tmp_path):
    # the notebook drops calls with the same text and different labels, so each text differs
    calls = [
        make_call(f"{t}_00000", t, scenario=f"{t}_s", texts=[f"{x} ({t})" for x in default_texts()])
        for t in export.TYPE_NAMES
    ]
    calls.append(make_call("benign_00000", "benign", scenario="family"))
    return build_store(tmp_path, *calls)


# ---- the export passes the notebook ----------------------------------------------------------


def test_jsonl_calls_pass_the_notebooks_load_clean_and_map_steps(tmp_path, monkeypatch):
    # the notebook drops calls with the same text and different labels, so each text differs;
    # a number marks them, not the type name, which the label-word check would drop
    calls = [
        jsonl_call(f"{t}_00000", t, [f"{x} ({n})" for x in default_texts()])
        for n, t in enumerate(export.TYPE_NAMES)
    ]
    calls.append(jsonl_call("benign_00000", None, themed_texts("clinic")))
    ns = run_notebook_cleaning(tmp_path, monkeypatch, calls)

    assert ns["rows_loaded"] == 6  # nothing dropped at load
    assert ns["rows_after_checks"] == 6  # 0 rows dropped by the step 3 checks
    df = ns["df"]
    assert set(df["source"]) == {"simulated"}
    scams = df[df["label_scam"]]
    assert (scams["speaker_turns"].map(len) == 6).all()  # every turn survived the cleaning
    assert set(df["scam_class"]) == {"Scam", "Not a Scam"}
    # the five spec types map to themselves, normal calls to no type
    assert dict(zip(scams["scam_type"], scams["label_type"], strict=True)) == {
        t: t for t in export.TYPE_NAMES
    }
    assert df.loc[~df["label_scam"], "label_type"].isna().all()
    # the noun in a receiver turn is kept as spoken
    assert scams["text"].str.contains("ผู้โทรคนนี้").all()
    assert set(df["call_id"]) == {c["id"] for c in calls}


def test_the_jsonl_text_is_what_the_notebook_trains_on(tmp_path, monkeypatch):
    call = make_call()
    ns = run_notebook_cleaning(tmp_path, monkeypatch, [as_jsonl(call)])
    got = ns["df"]["speaker_turns"].iat[0]
    from nlp_core.text import normalize_text

    assert [t["text"] for t in got] == [
        normalize_text(t["text"]) for t in call["generation"]["turns"]
    ]
    # letters by role, as in the app: B the caller (who speaks first here), A the receiver
    assert [t["speaker"] for t in got] == ["B", "A"] * 3


def test_a_jsonl_call_keeps_its_turns_and_roles(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch)
    call = jsonl_call(texts=["สวัสดีครับ  จากธนาคาร", "ครับ"])
    assert ns["jsonl_row"](call)["turns"] == [
        ("caller", "สวัสดีครับ  จากธนาคาร"),
        ("receiver", "ครับ"),
    ]
    cleaned = ns["clean_turns"](ns["jsonl_row"](call)["turns"])
    assert cleaned == [("caller", "สวัสดีครับ จากธนาคาร"), ("receiver", "ครับ")]


def test_a_normal_call_has_no_outcome_whatever_its_file_says(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch)
    row = ns["jsonl_row"](jsonl_call("benign_00001", None, outcome="scammed"))
    assert row["scam_class"] == "Not a Scam" and row["scam_type"] == ""
    assert row["scam_outcome"] is None and row["scam_outcome_source"] is None
    scam = ns["jsonl_row"](jsonl_call(outcome="scammed", outcome_source="inferred"))
    assert (scam["scam_outcome"], scam["scam_outcome_source"]) == ("scammed", "inferred")


@pytest.mark.parametrize(
    ("label_type", "scenario", "expected"),
    [
        ("lottery", "gov_lottery", "lottery/gov_lottery"),  # a name
        ("romance", {"key": "engineer_abroad", "role": "..."}, "romance/engineer_abroad"),  # a dict
        (None, "bank_fraud_alert", "normal/bank_fraud_alert"),  # a normal call
        ("lottery", None, "lottery/call_lottery_00007"),  # none: the call is its own scenario
    ],
)
def test_the_scenario_is_read_with_its_type_in_front(
    label_type, scenario, expected, tmp_path, monkeypatch
):
    ns = run_notebook_cleaning(tmp_path, monkeypatch)
    call = jsonl_call("lottery_00007", label_type, scenario=scenario)
    assert ns["jsonl_row"](call)["scenario"] == expected


# ---- masking the sensitive values --------------------------------------------------------------


def entity(texts, turn, value, label):
    """The entity for `value` in turn `turn` of a jsonl_call with these texts and A, B speakers."""
    text = "\n".join(f"{'AB'[i % 2]}: {t}" for i, t in enumerate(texts))
    line_start = sum(len(f"A: {t}") + 1 for t in texts[:turn])
    start = text.index(value, line_start + 3)
    return {"text": value, "label": label, "start": start, "end": start + len(value)}


MASKED_TEXTS = [
    "สวัสดีครับ โทรกลับเบอร์ 081-234-5678 นะครับ",
    "ค่ะ",
    "โอน 5,000 บาท เข้าบัญชี 123-4-56789-0 ครับ คุณสมชาย",
    "ไม่โอนค่ะ",
]


def masked_call():
    entities = [
        entity(MASKED_TEXTS, 0, "081-234-5678", "PHONE"),
        entity(MASKED_TEXTS, 2, "5,000 บาท", "MONEY"),
        entity(MASKED_TEXTS, 2, "123-4-56789-0", "BANK_ACCOUNT"),
        entity(MASKED_TEXTS, 2, "สมชาย", "PERSON"),  # not sensitive: it stays
    ]
    return jsonl_call(texts=MASKED_TEXTS, entities=entities)


def test_sensitive_values_become_their_label(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch, [masked_call()])
    assert ns["rows_after_checks"] == 1  # the placeholders are not notes: the call stays
    turns = [t["text"] for t in ns["df"]["speaker_turns"].iat[0]]
    assert turns == [
        "สวัสดีครับ โทรกลับเบอร์ [PHONE] นะครับ",
        "ค่ะ",
        "โอน [MONEY] เข้าบัญชี [BANK_ACCOUNT] ครับ คุณสมชาย",
        "ไม่โอนค่ะ",
    ]


def test_a_note_in_brackets_is_still_removed_but_a_placeholder_stays(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch)
    turns = ns["clean_turns"]([("caller", "โทรกลับ [PHONE] [วางสาย] ครับ"), ("receiver", "ค่ะ")])
    assert turns == [("caller", "โทรกลับ [PHONE] ครับ"), ("receiver", "ค่ะ")]


def test_an_entity_that_is_not_at_its_position_stops_the_notebook(tmp_path, monkeypatch):
    call = masked_call()
    call["entities"][0]["start"] += 1
    with pytest.raises(ValueError, match=r"calls.jsonl line 1.*PHONE"):
        run_notebook_cleaning(tmp_path, monkeypatch, [call])


def test_a_number_the_masking_missed_is_listed_not_dropped(tmp_path, monkeypatch, capsys):
    """A run of 3+ digits may be a missed value or just a year: the call stays, its id is shown."""
    texts = ["สวัสดีครับ เรื่องภาษีปี 2567 ครับ", "ค่ะ", "ขอรหัสครับ", "ไม่ให้ค่ะ"]
    ns = run_notebook_cleaning(
        tmp_path, monkeypatch, [jsonl_call("phishing_otp_00003", texts=texts)]
    )
    assert ns["rows_after_checks"] == 1
    out = capsys.readouterr().out
    assert "1 calls with a run of 3+ digits left" in out and "phishing_otp_00003" in out


def test_every_jsonl_file_in_dataset_is_read(tmp_path, monkeypatch):
    files = {
        "lottery.jsonl": [jsonl_call("lottery_00000", "lottery")],
        "normal.jsonl": [jsonl_call("benign_00000", None, themed_texts("clinic"))],
    }
    ns = run_notebook_cleaning(tmp_path, monkeypatch, files=files)
    (tmp_path / "nbroot" / "dataset" / "notes.txt").write_text("not a call", encoding="utf-8")
    exec(compile(cell("DATA_FILES = sorted("), "load", "exec"), ns)  # noqa: S102
    assert sorted(ns["raw"]["call_id"]) == ["benign_00000", "lottery_00000"]


def test_rejected_calls_are_dropped_and_counted(tmp_path, monkeypatch):
    by_judge = jsonl_call("romance_00001", "romance", [f"{x} (1)" for x in default_texts()])
    by_judge.update(dataset="real_rejected", review_status="rejected")
    by_person = jsonl_call("romance_00002", "romance", [f"{x} (2)" for x in default_texts()])
    by_person["review_status"] = "rejected"
    approved = jsonl_call("romance_00003", "romance", [f"{x} (3)" for x in default_texts()])
    approved.update(dataset="real_approved", review_status="approved")
    no_fields = jsonl_call("romance_00004", "romance", [f"{x} (4)" for x in default_texts()])
    files = {"romance.jsonl": [by_judge, by_person, approved, no_fields]}
    ns = run_notebook_cleaning(tmp_path, monkeypatch, files=files)
    assert sorted(ns["raw"]["call_id"]) == ["romance_00003", "romance_00004"]
    assert ns["rejected"] == {"romance.jsonl": 2}


def test_an_unknown_speaker_stops_the_notebook_with_the_file_and_line(tmp_path, monkeypatch):
    bad = jsonl_call(speakers=["A", "C", "A", "B", "A", "B"])
    with pytest.raises(ValueError, match=r"calls.jsonl line 2"):
        run_notebook_cleaning(tmp_path, monkeypatch, [jsonl_call("phishing_otp_00009"), bad])


@pytest.mark.parametrize(
    "said",
    ["นี่มันคือ Phishing ใช่ไหม", "scam แน่ๆ", "พวก SCAMMER", "เป็นการ impersonation",
     "fraud ชัดๆ", "แบบ fake_authority", "สแกมแน่นอน", "ฟิชชิ่งค่ะ"],
)  # fmt: skip
def test_a_call_that_names_a_label_is_dropped(said, tmp_path, monkeypatch):
    texts = ["สวัสดีครับ จากธนาคาร", said, "ขอรหัสครับ", "ไม่ให้ครับ"]
    calls = [jsonl_call("phishing_otp_00001", texts=texts), jsonl_call("phishing_otp_00002")]
    ns = run_notebook_cleaning(tmp_path, monkeypatch, calls)
    assert list(ns["df"]["call_id"]) == ["phishing_otp_00002"]


def test_words_people_say_are_not_label_words(tmp_path, monkeypatch):
    texts = ["ช่วงนี้มิจฉาชีพเยอะครับ", "lottery หรือ investment เหรอคะ", "ขอรหัส OTP ครับ", "ไม่ให้ค่ะ"]
    ns = run_notebook_cleaning(tmp_path, monkeypatch, [jsonl_call(texts=texts)])
    assert ns["rows_after_checks"] == 1


def test_shortcut_words_are_counted_by_class(tmp_path, monkeypatch):
    warned = jsonl_call(texts=["ช่วงนี้มิจฉาชีพเยอะครับ", "ครับ", "ขอรหัสครับ", "ไม่ให้ครับ"])
    ns = run_notebook_cleaning(
        tmp_path, monkeypatch, [warned, jsonl_call("benign_00000", None, themed_texts("clinic"))]
    )
    assert ns["shortcuts"].loc["มิจฉาชีพ"].to_dict() == {
        "% of Not a Scam calls": 0.0,
        "% of Scam calls": 100.0,
    }


# ---- refusals --------------------------------------------------------------------------------

HAZARDS = {
    # reason: (speaker whose turn it is, text)
    export.SPEAKER_TAG_IN_TURN: ("B", "เขาพูดว่า ผู้โทร: โอนเงินเดี๋ยวนี้"),
    export.BRACKET_IN_TURN: ("B", "ไม่เอาครับ [วางสาย]"),
    export.STAGE_NOTE_IN_TURN: ("B", "ฮือ (เสียงร้องไห้) ไม่รู้จะทำยังไง"),
    export.CALLER_NOUN: ("A", "ผมเป็นผู้โทรจากธนาคารครับ รบกวนยืนยันตัวตน"),
    export.LABEL_NAME: ("B", "นี่มันคือ Phishing ใช่ไหม"),
}


@pytest.mark.parametrize("reason", list(HAZARDS))
def test_a_turn_the_notebook_would_damage_gets_the_call_skipped(reason, tmp_path):
    speaker, bad = HAZARDS[reason]
    texts = ["สวัสดีครับ", "ครับ", bad, "ครับ"] if speaker == "A" else ["สวัสดีครับ", bad]
    bad_call = make_call("phishing_otp_00001", texts=texts)
    db = build_store(tmp_path, make_call(), bad_call)
    out, written, skipped = run_export(db, tmp_path)
    assert written == 1
    assert skipped == {reason: 1}
    assert list(read(out)["call_id"]) == ["phishing_otp_00000"]


def test_a_lone_bracket_gets_the_call_skipped(tmp_path):
    """No pair to cut, but a text that still has one would fail the notebook's bracket check."""
    out, written, skipped = run_export(
        build_store(tmp_path, make_call(texts=["สวัสดีครับ", "ไม่เอา ] ครับ"])), tmp_path
    )
    assert written == 0 and skipped == {export.BRACKET_IN_TURN: 1}


def test_a_hazard_hidden_by_an_invisible_character_is_found(tmp_path):
    """The caller word shows after normalizing, which removes the invisible mark."""
    bad = "ผมเป็นผู้โท​รจากธนาคารครับ"
    texts = ["สวัสดีครับ", "ครับ", bad, "ครับ"]
    out, written, skipped = run_export(build_store(tmp_path, make_call(texts=texts)), tmp_path)
    assert written == 0 and skipped == {export.CALLER_NOUN: 1}


def test_the_noun_in_a_receiver_turn_is_not_a_hazard(tmp_path):
    call = make_call(texts=["สวัสดีครับ", "ผู้โทรเป็นใครครับ", "ผมเป็นเจ้าหน้าที่ครับ", "ครับ"])
    out, written, skipped = run_export(build_store(tmp_path, call), tmp_path)
    assert written == 1 and not skipped


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
    ns = run_notebook_cleaning(tmp_path, monkeypatch)
    merged = ns["clean_turns"]([("caller", "x"), ("caller", "y"), ("receiver", "z")])
    assert merged == [("caller", "x y"), ("receiver", "z")]


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


# ---- the notebook's class sizes, split and export ---------------------------------------------

SCENARIOS_PER_TYPE = 8  # enough for the split to give every type a scenario in every split


def calls_of(label_type, outcome, n, source="locked", start=0):
    """n calls of one type and outcome, each with its own text, spread over SCENARIOS_PER_TYPE
    scenarios. The text tells them apart with numbers: a type or outcome name in it would be
    dropped by the label-word check."""
    kind = label_type or "normal"
    mark = len(f"{kind}{outcome}{source}") * 1000 + sum(map(ord, f"{kind}{outcome}{source}"))
    return [
        jsonl_call(
            f"{kind}_{outcome}_{source}_{start + i:05d}",
            label_type,
            [f"สวัสดีครับ เรื่องที่ {mark} เลขที่ {start + i}", f"ครับ {i}",
             f"รายละเอียดที่ {start + i} ของ {mark}", f"เข้าใจแล้วครับ {start + i}"],
            outcome=outcome if label_type else None,
            outcome_source=source if label_type else None,
            scenario=f"scene_{(start + i) % SCENARIOS_PER_TYPE}",
        )
        for i in range(n)
    ]  # fmt: skip


def balance_fixture():
    return (
        calls_of("lottery", "scammed", 6)
        + calls_of("lottery", "scammed", 4, source="inferred", start=100)
        + calls_of("lottery", "safe", 9)
        + calls_of("phishing_otp", "scammed", 5)
        + calls_of("phishing_otp", "safe", 7)
        + calls_of("phishing_otp", None, 3, source=None)  # outcome never set: cannot be balanced
        + calls_of(None, None, 20, source=None)
    )


def kept(ns):
    df = ns["df"]
    return df.groupby([df["label_type"].fillna("normal"), df["scam_outcome"].fillna("-")]).size()


def test_every_usable_call_is_kept_and_no_class_is_cut_down(tmp_path, monkeypatch):
    """No downsampling: Part B weights the classes, so the counts stay as they were."""
    ns = run_part_a_rest(run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture()))
    assert kept(ns).to_dict() == {
        ("lottery", "safe"): 9,
        ("lottery", "scammed"): 10,  # locked and inferred alike
        ("normal", "-"): 20,
        ("phishing_otp", "-"): 3,  # an outcome that was never set does not leave a call out
        ("phishing_otp", "safe"): 7,
        ("phishing_otp", "scammed"): 5,
    }
    assert set(ns["df"]["stratum"]) == {"lottery", "phishing_otp", "not_scam"}


def test_a_skipped_type_is_left_out_and_the_others_keep_every_call(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture())
    ns = run_part_a_rest(ns, SKIP_TYPES={"phishing_otp"})
    assert "phishing_otp" not in set(ns["df"]["label_type"])
    assert kept(ns).to_dict() == {
        ("lottery", "safe"): 9,
        ("lottery", "scammed"): 10,
        ("normal", "-"): 20,
    }


def test_a_group_is_one_scenario(tmp_path, monkeypatch):
    ns = run_part_a_rest(run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture()))
    df = ns["df"]
    assert df.groupby("group")["scenario"].nunique().max() == 1
    assert df.groupby("scenario")["group"].nunique().max() == 1
    assert df["group"].nunique() == df["scenario"].nunique()
    assert {s.split("/")[0] for s in df["scenario"]} == {"lottery", "phishing_otp", "normal"}


def test_the_split_is_new_every_run_and_reads_no_saved_split(tmp_path, monkeypatch):
    """An old test.csv that names every call as test, in both processed folders, is ignored."""
    ns = run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture())
    texts = ns["df"]["text"].tolist()
    for folder in ("processed", "processed_balanced"):
        old = tmp_path / "nbroot" / "data" / folder
        old.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"text": texts + ["A: สายเก่า\nB: ครับ"]}).to_csv(
            old / "test.csv", index=False, encoding="utf-8-sig"
        )
    ns = run_part_a_rest(ns)
    df = ns["df"]
    assert set(df["split"]) == {"train", "valid", "test"}
    # a scenario is in one split only, so valid and test hold scenarios train never saw
    assert df.groupby("group")["split"].nunique().max() == 1
    assert df.groupby("scenario")["split"].nunique().max() == 1
    # every stratum (the type, or not_scam) is in every split
    assert (pd.crosstab(df["stratum"], df["split"]) > 0).all().all()
    written = pd.read_csv(ns["PROCESSED"] / "test.csv", encoding="utf-8-sig")
    assert "A: สายเก่า\nB: ครับ" not in set(written["text"])
    assert len(written) < len(ns["rows"])
    # every row of a call is in the call's split
    assert ns["rows"].groupby("call_id")["split"].nunique().max() == 1
    assert (
        ns["rows"]["split"] == ns["rows"]["call_id"].map(df.set_index("call_id")["split"])
    ).all()


def test_the_notebooks_default_row_text_is_the_call_so_far(tmp_path, monkeypatch):
    """The app classifies the call so far after each turn, so the rows hold it by default."""
    ns = run_part_a_rest(run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture()))
    assert ns["ROW_TEXT"] == "call_so_far"
    assert "turns_dropped" in ns["rows"].columns


def test_the_export_has_the_call_id_and_the_outcome(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture())
    ns = run_part_a_rest(ns, row_text="one_line")
    assert ns["PROCESSED"].name == "processed_balanced"
    frame = pd.concat(
        pd.read_csv(ns["PROCESSED"] / f"{n}.csv", encoding="utf-8-sig", keep_default_na=False)
        for n in ("train", "valid", "test")
    )
    # turns_dropped is only written with ROW_TEXT = "call_so_far"
    assert list(frame.columns)[-5:] == [
        "call_id",
        "scam_outcome",
        "scam_outcome_source",
        "scenario",
        "turn_index",
    ]
    # ROW_TEXT = "one_line": one row per turn, the turn's line on its own; of exact copies one
    # is kept (54 calls of 4 turns; "ครับ 0" and the like repeat across calls)
    assert frame["id"].is_unique and frame["text"].is_unique and len(frame) < 54 * 4
    assert not frame["text"].str.contains("\n").any()
    assert set(frame["id"].str.rsplit("_t", n=1).str[1]) == {"00", "01", "02", "03"}
    assert set(frame["scenario"].str.split("/").str[0]) == {"lottery", "phishing_otp", "normal"}
    safe = frame[frame["label_scam"] == False]  # noqa: E712
    assert (safe["scam_outcome"] == "").all() and (safe["scam_outcome_source"] == "").all()
    assert (safe["label_type"] == "").all()
    scams = frame[frame["label_scam"] == True]  # noqa: E712
    # one scam row per scam call (turn 2, the start); the 3 phishing_otp calls whose outcome
    # was never set keep an empty outcome
    assert (scams["turn_index"] == 2).all()
    assert scams["scam_outcome"].value_counts().to_dict() == {"safe": 16, "scammed": 15, "": 3}
    # only the row files are written: the whole calls stay in dataset/*.jsonl
    assert sorted(p.name for p in ns["PROCESSED"].iterdir()) == [
        "dataset.csv",
        "test.csv",
        "train.csv",
        "valid.csv",
    ]


def test_the_export_with_the_call_so_far(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture())
    ns = run_part_a_rest(ns, row_text="call_so_far")
    frame = pd.concat(
        pd.read_csv(ns["PROCESSED"] / f"{n}.csv", encoding="utf-8-sig", keep_default_na=False)
        for n in ("train", "valid", "test")
    )
    # with the call so far, the rows also say how many old lines were cut
    assert list(frame.columns)[-2:] == ["turn_index", "turns_dropped"]
    assert (frame["turns_dropped"].astype(int) == 0).all()  # these short calls all fit
    # every call-so-far text is unique here, so all 54 calls x 4 turns are rows
    assert len(frame) == 54 * 4 and frame["text"].is_unique
    assert set(frame["text"].str.count("\n")) == {0, 1, 2, 3}
    scams = frame[frame["label_scam"] == True]  # noqa: E712
    assert (scams["turn_index"] == 2).all()
    assert (scams["scam_start_turn"].astype(int) == 2).all()


# ---- step 6b: one row per turn: the turn's line, or the call so far --------------------------


@pytest.fixture
def part_a(tmp_path, monkeypatch):
    """The notebook's namespace after part A, so its step 6b functions can be called."""
    return run_part_a_rest(run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture()))


def rows_of(ns, texts, scam_start=None, label_type=None, row_text="one_line", max_tokens=10_000):
    """Step 6b's rows of one call with this ROW_TEXT: the caller speaks first, then the turns
    alternate. Tokens are words (word_count), at most `max_tokens` a row."""
    roles = [("caller", "receiver")[i % 2] for i in range(len(texts))]
    call = SimpleNamespace(
        call_id="lottery_09999",
        speaker_turns=ns["speaker_turns"](list(zip(roles, texts, strict=True))),
        scam_start_clean=scam_start,
        label_scam=label_type is not None,
        label_type=label_type,
        scam_outcome="scammed" if label_type else None,
        scam_outcome_source="locked" if label_type else None,
        source="simulated",
        scam_type=label_type or "",
        group=0,
        split="train",
        scenario=f"{label_type or 'normal'}/test",
    )
    ns["ROW_TEXT"], ns["MAX_TOKENS"] = row_text, max_tokens
    return ns[f"{row_text}_rows"](call)


FIVE_TURNS = [
    "สวัสดีครับ ติดต่อจาก บริษัท",
    "ครับ มีอะไรครับ",
    "ขอเลขบัตร ประชาชน ครับ",
    "ไม่ให้ครับ",
    "ขอเลขบัตร อีกที ครับ",
]


@pytest.mark.parametrize("row_text", ["one_line", "call_so_far"])
def test_a_scam_call_is_labelled_scam_from_its_start_on_the_caller_turns_only(part_a, row_text):
    rows = rows_of(part_a, FIVE_TURNS, scam_start=2, label_type="lottery", row_text=row_text)
    # the same label rule for both row texts: before the start: safe; from it on: the caller
    # turns are scam, the receiver turns safe
    assert [r["label_scam"] for r in rows] == [False, False, True, False, True]
    assert [r["label_type"] for r in rows] == [None, None, "lottery", None, "lottery"]
    assert [r["scam_outcome"] for r in rows] == [None, None, "scammed", None, "scammed"]


@pytest.mark.parametrize("row_text", ["one_line", "call_so_far"])
def test_every_row_of_a_normal_call_is_safe(part_a, row_text):
    rows = rows_of(part_a, FIVE_TURNS, row_text=row_text)
    assert len(rows) == 5 and not any(r["label_scam"] for r in rows)
    assert all(r["label_type"] is None and r["scam_outcome"] is None for r in rows)


def test_a_row_is_the_turns_line_on_its_own_with_the_caller_as_b(part_a):
    rows = rows_of(part_a, FIVE_TURNS, scam_start=2, label_type="lottery")
    assert [r["text"] for r in rows] == [
        "B: สวัสดีครับ ติดต่อจาก บริษัท",
        "A: ครับ มีอะไรครับ",
        "B: ขอเลขบัตร ประชาชน ครับ",
        "A: ไม่ให้ครับ",
        "B: ขอเลขบัตร อีกที ครับ",
    ]
    assert [r["turn_index"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(len(r["speaker_turns"]) == 1 for r in rows)
    assert {r["speaker_turns"][0]["speaker"] for r in rows if r["label_scam"]} == {"B"}
    # 0 on the row where the scam starts, empty on the others; nothing is ever cut
    assert [r["scam_start_turn"] for r in rows] == [None, None, 0, None, None]
    assert not any("turns_dropped" in r for r in rows)


def test_a_call_so_far_row_holds_every_line_up_to_its_turn(part_a):
    rows = rows_of(part_a, FIVE_TURNS, scam_start=2, label_type="lottery", row_text="call_so_far")
    assert rows[2]["text"] == "B: สวัสดีครับ ติดต่อจาก บริษัท\nA: ครับ มีอะไรครับ\nB: ขอเลขบัตร ประชาชน ครับ"
    assert [len(r["speaker_turns"]) for r in rows] == [1, 2, 3, 4, 5]
    # where the start sits inside each row, when the row holds it
    assert [r["scam_start_turn"] for r in rows] == [None, None, 2, 2, 2]


def test_a_long_call_so_far_row_loses_its_oldest_lines_and_keeps_its_own(part_a):
    # word_count counts "B:" as a word: the lines are 4, 3, 4, 2 and 4 words long
    rows = rows_of(
        part_a, FIVE_TURNS, scam_start=2, label_type="lottery", row_text="call_so_far", max_tokens=9
    )
    assert [r["turns_dropped"] for r in rows] == [0, 0, 1, 1, 3]
    for r in rows:
        assert r["text"].rsplit("\n", 1)[-1].endswith(FIVE_TURNS[r["turn_index"]])
        assert word_count(r["text"]) <= 9
    # the start counted inside the row; once it is cut away, it is unknown in that row
    assert [r["scam_start_turn"] for r in rows] == [None, None, 1, 1, None]
    assert [r["label_scam"] for r in rows] == [False, False, True, False, True]


def test_a_single_line_longer_than_the_limit_is_kept_whole_in_the_call_so_far(part_a):
    rows = rows_of(part_a, FIVE_TURNS, row_text="call_so_far", max_tokens=2)
    assert [r["turns_dropped"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["text"] for r in rows][0] == "B: สวัสดีครับ ติดต่อจาก บริษัท"


def test_row_text_must_be_one_of_the_two(part_a):
    settings = cell("ROW_TEXTS = (").replace('ROW_TEXT = "call_so_far"', 'ROW_TEXT = "whole_call"')
    with pytest.raises(ValueError, match="ROW_TEXT must be one of"):
        exec(compile(settings, "settings", "exec"), dict(part_a))  # noqa: S102


def test_a_scam_start_on_a_receiver_turn_stops_the_notebook(part_a):
    with pytest.raises(ValueError, match=r"lottery_09999: the scam start is not a caller turn"):
        rows_of(part_a, FIVE_TURNS, scam_start=1, label_type="lottery")


def test_a_scam_call_with_no_start_is_dropped_but_a_normal_call_is_kept(tmp_path, monkeypatch):
    calls = [
        jsonl_call("phishing_otp_00001", texts=[f"{x} (1)" for x in default_texts()]),
        jsonl_call(
            "phishing_otp_00002", texts=[f"{x} (2)" for x in default_texts()], scam_start=None
        ),
        jsonl_call("benign_00000", None, themed_texts("clinic")),
    ]
    ns = run_notebook_cleaning(tmp_path, monkeypatch, calls)
    assert sorted(ns["df"]["call_id"]) == ["benign_00000", "phishing_otp_00001"]


def test_part_a_writes_one_row_per_turn_in_its_calls_split(part_a):
    rows, df = part_a["rows"], part_a["df"]
    # every turn is a row, less the exact copies (one copy of each text is kept)
    dropped = part_a["copies"].sum() + part_a["two_labels"].sum()
    assert len(rows) == df["speaker_turns"].map(len).sum() - dropped
    assert rows["text"].is_unique
    # the default ROW_TEXT: each row holds one line per turn it keeps
    assert (rows["text"].str.count("\n") == rows["speaker_turns"].map(len) - 1).all()
    assert (rows["split"] == rows["call_id"].map(df.set_index("call_id")["split"])).all()


def test_the_token_table_stops_on_a_line_longer_than_the_limit(tmp_path, monkeypatch):
    ns = run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture())
    with pytest.raises(AssertionError, match="a line is longer than part B reads"):
        run_part_a_rest(ns, max_tokens=3)


# ---- the start of part B: the hand-written outside test and the evaluation helpers -----------


def write_handwritten(root, rows):
    """A data/processed/dataset.csv like the one the earlier notebook wrote: (label_type, scam)."""
    folder = root / "data" / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "text": [f"A: สวัสดีครับ {i}\nB: ครับ" for i in range(len(rows))],
            "speaker_turns": "[]",
            "entities": "[]",
            "label_scam": [scam for _, scam in rows],
            "label_type": [label_type for label_type, _ in rows],
        }
    ).to_csv(folder / "dataset.csv", index=False, encoding="utf-8-sig")


def run_part_b_start(tmp_path, monkeypatch, handwritten_rows=None, run_handwritten=True):
    """Part A on the balance fixture, the first cell of part B, then the setup cell of the added
    evaluation section after step 13 (no GPU needed). Every type of the small fixture counts as
    trained. `run_handwritten` sets the RUN_HANDWRITTEN switch of that cell (off in the
    notebook)."""
    ns = run_part_a_rest(run_notebook_cleaning(tmp_path, monkeypatch, balance_fixture()))
    if handwritten_rows is not None:
        write_handwritten(tmp_path / "nbroot", handwritten_rows)
    ns["MIN_TYPE_SAMPLES"] = ns["MIN_TYPE_EVAL_SAMPLES"] = 1
    exec(compile(cell("def read_records"), "part B", "exec"), ns)  # noqa: S102
    setup = cell("def recall_by_outcome")
    assert setup.count("RUN_HANDWRITTEN = False") == 1
    setup = setup.replace("RUN_HANDWRITTEN = False", f"RUN_HANDWRITTEN = {run_handwritten}")
    exec(compile(setup, "added evaluation", "exec"), ns)  # noqa: S102
    return ns


def test_recall_by_outcome_counts_each_ending_and_leaves_normal_calls_out(tmp_path, monkeypatch):
    ns = run_part_b_start(tmp_path, monkeypatch)
    frame = pd.DataFrame(
        {
            "label_scam": [True, True, True, True, True, False, False],
            "scam_outcome": ["scammed", "scammed", "safe", "safe", "safe", None, None],
        }
    )
    got = ns["recall_by_outcome"](frame, [1, 0, 1, 1, 1, 1, 0])
    assert got.loc["scammed"].to_dict() == {"calls": 2, "caught": 1, "recall": 0.5}
    assert got.loc["safe"].to_dict() == {"calls": 3, "caught": 3, "recall": 1.0}
    assert got.loc["all scams"].to_dict() == {"calls": 5, "caught": 4, "recall": 0.8}


def test_macro_f1_leaves_out_classes_that_are_not_scored(tmp_path, monkeypatch):
    ns = run_part_b_start(tmp_path, monkeypatch)
    # class 2 has no true sample: averaged over 0 and 1 only it does not count as an F1 of 0
    got = ns["macro_f1_over"]([0, 0, 1, 1], [0, 0, 1, 2], [0, 1])
    assert got == pytest.approx((1.0 + 2 / 3) / 2)


def test_the_hand_written_calls_are_read_as_an_outside_test(tmp_path, monkeypatch):
    rows = [("lottery", True)] * 2 + [("fake_authority", True)] * 4 + [(None, False)] * 3
    ns = run_part_b_start(tmp_path, monkeypatch, rows)
    assert len(ns["handwritten"]) == 9 and ns["handwritten"]["gate"].sum() == 6
    # fake_authority is not a type the fixture trains, so only lottery can be scored
    assert ns["TYPE_LABELS"] == ["phishing_otp", "lottery"]
    assert ns["HANDWRITTEN_TYPES"] == ["lottery"]
    typed = ns["handwritten_typed"]
    assert set(typed["label_type"]) == {"lottery"}
    assert set(typed["type"]) == {ns["TYPE_LABELS"].index("lottery")}


def test_without_the_hand_written_file_the_outside_test_is_skipped(tmp_path, monkeypatch, capsys):
    ns = run_part_b_start(tmp_path, monkeypatch)
    assert ns["handwritten"] is None and ns["HANDWRITTEN_TYPES"] == []
    assert "outside test is skipped" in capsys.readouterr().out


def test_the_hand_written_test_is_off_by_default_even_with_the_file(tmp_path, monkeypatch, capsys):
    """RUN_HANDWRITTEN is False in the notebook: the file holds whole calls with the old letters,
    while the models learn from one row per turn."""
    rows = [("lottery", True)] * 2 + [(None, False)] * 2
    ns = run_part_b_start(tmp_path, monkeypatch, rows, run_handwritten=False)
    assert ns["handwritten"] is None and ns["HANDWRITTEN_TYPES"] == []
    assert "RUN_HANDWRITTEN is False" in capsys.readouterr().out


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
