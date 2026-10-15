import re
import wave
from pathlib import Path

import numpy as np
import pytest

from backend import dataset
from data_pipeline.store import Store

HX = {"HX-Request": "true"}


def write_wav(path, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * seconds), dtype="<i2").tobytes())


def record(rid="phishing_otp_00001", status="accepted", scam=True):
    turns = [
        {
            "speaker": "A",
            "text": "ขอรหัส 123456 ด้วยค่ะ",
            "entities": [{"text": "123456", "label": "OTP", "start": 6, "end": 12}],
            "reasoning": "ask for the OTP now",
            "problems": [],
            "spelling": [],
            "audio": f"audio/{rid}/00_A.wav",
            "spoken_text": "ขอรหัส หนึ่งสองสามสี่ห้าหก ด้วยค่ะ",
            "prompt_tokens": 1234,
        },
        {
            "speaker": "B",
            "text": "ไม่ให้นะค่ะ",
            "entities": [],
            "reasoning": "",
            "problems": ["number_without_slot"],
            "leaked": ["OTP", "ID_CARD"],
            "spelling": [],
            "audio": f"audio/{rid}/01_B.wav",
        },
    ]
    return {
        "id": rid,
        "text": "A: ขอรหัส 123456 ด้วยค่ะ\nB: ไม่ให้นะค่ะ",
        "speaker_turns": [
            {"speaker": "A", "start": 0.0, "end": 1.0, "text": turns[0]["text"]},
            {"speaker": "B", "start": 1.45, "end": 2.45, "text": turns[1]["text"]},
        ],
        "label_scam": scam,
        "label_type": "phishing_otp" if scam else None,
        "entities": [{"text": "123456", "label": "OTP", "start": 9, "end": 15}],
        "reference_summary": "ผู้โทรขอรหัส OTP",
        "scam_start_turn": 0 if scam else None,
        "generation": {
            "status": status,
            "reasons": [] if status == "accepted" else ["no_scam_move", "unknown_slot:ADDRESS"],
            "turns": turns,
            "judge": {"label_holds": True, "natural_thai": 4, "issues": "none", "reasoning": "r"},
            "scam_start_from": "caller",
            "speech_style": "dash",
            "voices": {"A": "th_f_1", "B": "th_m_2"},
            "call": {"seed": 1, "scenario": {"key": "bank_security", "role": "bank staff"}},
        },
        "review": {"status": "pending", "note": "", "reviewed_at": None},
    }


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    db = tmp_path / "data" / "calls.db"
    store = Store(db)
    for rec in (record(), record("benign_00002", status="rejected", scam=False)):
        store.save(rec)
    write_wav(db.parent / "audio" / "phishing_otp_00001" / "00_A.wav")
    write_wav(db.parent / "audio" / "phishing_otp_00001.wav", 2.45)
    monkeypatch.setenv("CALLAS_DATASET_DB", str(db))
    return db


def test_page_opens_on_the_to_review_view(client, db_path):
    html = client.get("/dataset").text
    assert html.count("data-row=") == 1  # only the call that passed checks and is unreviewed
    assert 'data-call="phishing_otp_00001"' in html
    assert '>To review<span class="view-count">1</span>' in html
    assert '>Failed checks<span class="view-count">1</span>' in html
    assert '0</span> <span class="text-muted">of 2 reviewed' in html


def test_failed_view_shows_why(client, db_path):
    html = client.get("/dataset?view=failed").text
    assert html.count("data-row=") == 1 and 'data-call="benign_00002"' in html
    assert "Why it failed the checks" in html
    assert "Unknown slot {ADDRESS}" in html and "No scam move found" in html


def test_call_detail_partial(client, db_path):
    html = client.get("/dataset/call/phishing_otp_00001", headers=HX).text
    assert "<html" not in html
    assert 'data-tone="alarm" data-label="OTP">123456<' in html  # entity highlight
    assert "Scam starts" in html and "Flagged by the caller agent" in html
    assert 'data-play-src="/dataset/audio/phishing_otp_00001/00_A.wav"' in html
    assert "/dataset/audio/phishing_otp_00001/01_B.wav" not in html  # file missing: no button
    assert 'data-start="1.45"' in html  # messages carry call timings for follow-along
    assert "Number written without a slot" in html
    assert "ask for the OTP now" in html


def test_detail_without_htmx_redirects_to_page(client, db_path):
    r = client.get("/dataset/call/phishing_otp_00001", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/dataset?id=phishing_otp_00001"


def test_filters_return_the_queue_only(client, db_path):
    r = client.get("/dataset?view=all&type=benign", headers={**HX, "HX-Target": "queue"})
    assert "<html" not in r.text
    assert r.text.count("data-row=") == 1 and "benign_00002" in r.text
    assert "of 1 reviewed" in r.text  # progress follows the type filter


def test_review_is_saved_and_list_row_updates(client, db_path):
    r = client.post(
        "/dataset/call/phishing_otp_00001/review",
        data={"status": "approved", "note": "ดี", "view": "all"},
        headers=HX,
    )
    assert r.status_code == 200
    saved = Store(db_path).load("phishing_otp_00001")
    assert saved["review"]["status"] == "approved" and saved["review"]["note"] == "ดี"
    assert saved["review"]["reviewed_at"]
    assert saved["generation"]["turns"][0]["text"] == "ขอรหัส 123456 ด้วยค่ะ"  # call untouched
    assert 'id="row-phishing_otp_00001" hx-swap-oob="outerHTML"' in r.text
    assert 'id="queue-progress" hx-swap-oob="outerHTML"' in r.text
    assert "1 approved · 0 rejected" in r.text
    assert 'data-review="approved"' in r.text

    undo = client.post(
        "/dataset/call/phishing_otp_00001/review",
        data={"status": "pending", "note": ""},
        headers=HX,
    )
    assert undo.status_code == 200
    saved = Store(db_path).load("phishing_otp_00001")
    assert saved["review"] == {"status": "pending", "note": "", "reviewed_at": None}


def test_undo_without_a_note_keeps_the_stored_one(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    client.post(url, data={"status": "approved", "note": "keep me"}, headers=HX)
    # The toast's Undo sends no note, only the flag.
    client.post(url, data={"status": "pending", "keep_note": "1"}, headers=HX)
    review = Store(db_path).load("phishing_otp_00001")["review"]
    assert review["status"] == "pending" and review["note"] == "keep me"
    assert review["reviewed_at"] is None
    # The flag does nothing for a real decision, and the bar's own Undo sends its note box.
    client.post(url, data={"status": "approved", "note": "", "keep_note": "1"}, headers=HX)
    assert Store(db_path).load("phishing_otp_00001")["review"]["note"] == ""


def test_review_from_a_plain_form_post_is_refused(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    r = client.post(url, data={"status": "approved"})
    assert r.status_code == 403
    assert Store(db_path).load("phishing_otp_00001")["review"]["status"] == "pending"


def test_review_from_another_origin_is_refused(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    for origin in ("http://evil.example", "null", "http://testserver:9999"):
        r = client.post(url, data={"status": "approved"}, headers={**HX, "Origin": origin})
        assert r.status_code == 403, origin
    assert Store(db_path).load("phishing_otp_00001")["review"]["status"] == "pending"
    same = client.post(
        url, data={"status": "approved"}, headers={**HX, "Origin": "http://testserver"}
    )
    assert same.status_code == 200


def test_review_that_leaves_the_view_removes_its_row(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    gone = client.post(url, data={"status": "approved", "view": "todo"}, headers=HX)
    assert 'id="row-phishing_otp_00001" hx-swap-oob="delete"' in gone.text
    kept = client.post(url, data={"status": "approved", "view": "all"}, headers=HX)
    assert 'id="row-phishing_otp_00001" hx-swap-oob="outerHTML"' in kept.text
    assert 'hx-swap-oob="delete"' not in kept.text


def test_progress_follows_the_type_sent_with_the_review(client, db_path):
    r = client.post(
        "/dataset/call/phishing_otp_00001/review",
        data={"status": "rejected", "type": "benign"},
        headers=HX,
    )
    assert "of 1 reviewed" in r.text and "0 approved · 0 rejected" in r.text


def test_review_rejects_bad_input(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    assert client.post(url, data={"status": "maybe"}, headers=HX).status_code == 422
    missing = client.post(
        "/dataset/call/nope_00001/review", data={"status": "approved"}, headers=HX
    )
    assert missing.status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/dataset/audio/phishing_otp_00001/..%2F..%2Fphishing_otp_00001.json",
        "/dataset/audio/..%2F..%2Fsecret",
        "/dataset/audio/phishing_otp_00001/01_B.wav",  # pattern ok, file missing
        "/dataset/audio/phishing_otp_00001/notes.txt",
    ],
)
def test_audio_only_serves_known_files(client, db_path, path):
    assert client.get(path).status_code == 404


def test_audio_serves_turn_and_call(client, db_path):
    turn = client.get("/dataset/audio/phishing_otp_00001/00_A.wav")
    whole = client.get("/dataset/audio/phishing_otp_00001")
    assert turn.status_code == whole.status_code == 200
    assert turn.headers["content-type"] == "audio/wav"


def test_empty_folder_explains_how_to_generate(client, tmp_path, monkeypatch):
    monkeypatch.setenv("CALLAS_DATASET_DB", str(tmp_path / "none" / "calls.db"))
    html = client.get("/dataset").text
    assert "No call to review" in html and "python -m data_pipeline.synthesize" in html
    assert "No database at <code" in html and "CALLAS_DATASET_DB" in html
    assert "CALLAS_DATASET_DIR" not in html and "JSON" not in html


def test_missing_database_is_not_created_by_any_get(client, tmp_path, monkeypatch):
    folder = tmp_path / "none"
    monkeypatch.setenv("CALLAS_DATASET_DB", str(folder / "calls.db"))
    for url in (
        "/dataset",
        "/dataset?view=all&id=phishing_otp_00001",
        "/dataset/call/phishing_otp_00001",
        "/dataset/audio/phishing_otp_00001",
    ):
        client.get(url, headers={**HX, "HX-Target": "queue"})
    post = client.post(
        "/dataset/call/phishing_otp_00001/review", data={"status": "approved"}, headers=HX
    )
    assert post.status_code == 404
    assert not folder.exists()


def test_bogus_view_type_and_id_do_not_break_the_page(client, db_path):
    r = client.get("/dataset?view=nonsense&type=nonsense")
    assert r.status_code == 200
    assert '>To review<span class="view-count">1</span>' in r.text  # fell back to defaults
    assert 'data-call="phishing_otp_00001"' in r.text
    r = client.get("/dataset?id=phishing_otp_09999")
    assert r.status_code == 200 and "No call to review" in r.text
    r = client.get("/dataset?id=../etc/passwd")
    assert r.status_code == 200 and "No call to review" in r.text
    assert client.get("/dataset/call/phishing_otp_09999", headers=HX).status_code == 404


class AnythingGoes:
    """A store that knows every id, on a disk where every file exists. Only the guards in
    dataset.load and dataset.audio_file are left to say no."""

    def __init__(self, folder):
        self.folder = folder

    def load(self, record_id):
        return {"id": record_id}


@pytest.fixture
def everything_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    return AnythingGoes(tmp_path)


BAD_IDS = ["../x", "..\\x", "phishing_otp_00001\n", "..", "", "phishing_otp_00001/../x"]
BAD_NAMES = ["../x", "..\\x", "00_A.wav\n", "../../phishing_otp_00001.wav", "0_A.wav", ""]


@pytest.mark.parametrize("bad", BAD_IDS)
def test_load_and_audio_file_reject_bad_ids(everything_exists, bad):
    assert dataset.load(everything_exists, bad) is None
    assert dataset.audio_file(everything_exists, bad) is None
    assert dataset.audio_file(everything_exists, bad, "00_A.wav") is None


@pytest.mark.parametrize("name", BAD_NAMES)
def test_audio_file_rejects_bad_names(everything_exists, name):
    assert dataset.audio_file(everything_exists, "phishing_otp_00001", name) is None


def test_guards_let_good_ids_and_names_through(everything_exists):
    assert dataset.load(everything_exists, "phishing_otp_00001") == {"id": "phishing_otp_00001"}
    assert dataset.audio_file(everything_exists, "phishing_otp_00001") is not None
    assert dataset.audio_file(everything_exists, "phishing_otp_00001", "00_A.wav") is not None


def test_audio_file_finds_real_files(db_path):
    db = Store(db_path)
    assert dataset.audio_file(db, "phishing_otp_00001", "00_A.wav").is_file()
    assert dataset.audio_file(db, "phishing_otp_00001").is_file()
    assert dataset.load(db, "phishing_otp_00001")["id"] == "phishing_otp_00001"


def test_save_review_rejects_a_trailing_newline_id(db_path):
    db = Store(db_path)
    assert dataset.save_review(db, "phishing_otp_00001\n", "approved", "") is None


def test_text_from_the_database_is_escaped(client, db_path):
    rec = record("scam_00003")
    evil = "<script>alert(1)</script>"
    rec["generation"]["turns"][0]["text"] = f"<b>{evil}</b>"
    rec["generation"]["turns"][0]["reasoning"] = f"<img src=x onerror={evil}>"
    rec["review"]["note"] = '"><svg onload=alert(2)>'
    Store(db_path).save(rec)
    html = client.get("/dataset?id=scam_00003").text
    assert "<script>alert(1)" not in html and "<img src=x" not in html
    assert "<svg onload" not in html and '"><svg' not in html
    assert "&lt;script&gt;alert(1)" in html and "&lt;img src=x" in html
    assert "&gt;&lt;svg onload" in html


@pytest.mark.parametrize("path", ["/message", "/call", "/live", "/dataset"])
def test_static_references_are_versioned_paths(client, db_path, path):
    html = client.get(path).text
    refs = re.findall(r'(?:src|href)="([^"]*static/[^"]*)"', html)
    assert refs, path
    for ref in refs:
        assert ref.startswith("/static/") and "?v=" in ref, (path, ref)


def test_reason_codes_from_the_simulator_have_words():
    from backend import dataset

    for code in ("stray_brace", "line_break", "number_without_slot", "no_scam_move"):
        assert dataset.reason_text(code) != code


def test_a_leaking_receiver_turn_is_tagged(client, db_path):
    html = client.get("/dataset/call/phishing_otp_00001", headers=HX).text
    assert html.count("Leaked: OTP, ID CARD") == 1
    assert 'data-tone="alarm" data-leak lang="en">Leaked: OTP, ID CARD<' in html
    assert dataset._leak_tag({"leaked": []}) == "" and dataset._leak_tag({}) == ""


def test_the_detail_shows_spoken_text_prompt_size_and_speech_style(client, db_path):
    html = client.get("/dataset/call/phishing_otp_00001", headers=HX).text
    assert "Spoken: ขอรหัส หนึ่งสองสามสี่ห้าหก ด้วยค่ะ" in html
    assert html.count("Prompt about 1234 tokens") == 1  # the other turn has no size
    assert "with ขีด between digit groups" in html
    assert dataset.reason_text("too_long_for_context") == "A prompt did not fit the context window"
    assert "only the receiver knows" in dataset.reason_text("caller_used_receiver_slot")
