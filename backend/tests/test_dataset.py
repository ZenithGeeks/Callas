import wave

import numpy as np
import pytest

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
        },
        {
            "speaker": "B",
            "text": "ไม่ให้นะค่ะ",
            "entities": [],
            "reasoning": "",
            "problems": ["number_without_slot"],
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
        "/dataset/call/phishing_otp_00001/review", data={"status": "approved", "note": "ดี"}
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

    undo = client.post("/dataset/call/phishing_otp_00001/review", data={"status": "pending"})
    assert undo.status_code == 200
    saved = Store(db_path).load("phishing_otp_00001")
    assert saved["review"] == {"status": "pending", "note": "", "reviewed_at": None}


def test_progress_follows_the_type_sent_with_the_review(client, db_path):
    r = client.post(
        "/dataset/call/phishing_otp_00001/review", data={"status": "rejected", "type": "benign"}
    )
    assert "of 1 reviewed" in r.text and "0 approved · 0 rejected" in r.text


def test_review_rejects_bad_input(client, db_path):
    url = "/dataset/call/phishing_otp_00001/review"
    assert client.post(url, data={"status": "maybe"}).status_code == 422
    assert (
        client.post("/dataset/call/nope_00001/review", data={"status": "approved"}).status_code
        == 404
    )


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
