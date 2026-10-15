import json
import threading

from data_pipeline.store import Store, import_json_folder


def make_record(call_id="phishing_otp_00001", call_type="phishing_otp", seed=1, turns=2):
    lines = [
        ("A", "ขอรหัส 123456 ด้วยค่ะ", [{"text": "123456", "label": "OTP", "start": 7, "end": 13}]),
        ("B", "ไม่ให้ค่ะ", []),
        (
            "A",
            "โอนมาที่ 111-2-33333-4 นะ",
            [{"text": "111-2-33333-4", "label": "BANK_ACCOUNT", "start": 9, "end": 22}],
        ),
    ]
    gen_turns = []
    for speaker, text, entities in lines[:turns]:
        gen_turns.append(
            {
                "speaker": speaker,
                "agent": "scam_caller" if speaker == "A" else "receiver",
                "raw": text,
                "text": text,
                "spoken_text": "",
                "entities": entities,
                "asks_now": speaker == "A",
                "end_call": False,
                "reasoning": "r",
                "problems": [],
                "leaked": [],
                "spelling": [],
                "latency_s": 1.0,
                "prompt_tokens": 321,
                "audio": None,
                "audio_s": None,
            }
        )
    scam = call_type != "benign"
    return {
        "id": call_id,
        "channel": "call",
        "text": "",
        "audio_path": None,
        "speaker_turns": [
            {"speaker": t["speaker"], "start": None, "end": None, "text": t["text"]}
            for t in gen_turns
        ],
        "label_scam": scam,
        "label_type": call_type if scam else None,
        "entities": [],
        "reference_summary": "สรุป",
        "scam_start_turn": 0 if scam else None,
        "source": "synthetic",
        "generation": {
            "status": "accepted",
            "reasons": [],
            "config_name": "config",
            "config_hash": "abc123",
            "models": {"judge": "m"},
            "created_at": "2026-10-15T00:00:00+00:00",
            "call": {"call_type": call_type, "seed": seed},
            "voices": {"A": "th_f_1", "B": "th_m_1"},
            "briefing": None,
            "turns": gen_turns,
            "receiver_leaked": [],
            "first_leak_turn": None,
            "speech_style": "",
            "scam_start_from": "caller" if scam else None,
            "judge": {"label_holds": True},
        },
        "review": {"status": "pending", "note": "", "reviewed_at": None},
    }


def test_round_trip(tmp_path):
    store = Store(tmp_path / "calls.db")
    record = make_record()
    store.save(record)
    loaded = store.load("phishing_otp_00001")
    assert loaded["generation"]["turns"] == record["generation"]["turns"]
    assert loaded["review"] == record["review"]
    assert loaded["generation"]["call"] == record["generation"]["call"]
    assert loaded["label_type"] == "phishing_otp"
    assert loaded["label_scam"] is True


def test_text_and_entities_are_rebuilt(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record(turns=3))
    loaded = store.load("phishing_otp_00001")
    assert loaded["text"] == "A: ขอรหัส 123456 ด้วยค่ะ\nB: ไม่ให้ค่ะ\nA: โอนมาที่ 111-2-33333-4 นะ"
    assert len(loaded["entities"]) == 2
    for e in loaded["entities"]:
        assert loaded["text"][e["start"] : e["end"]] == e["text"]


def test_saving_again_updates_in_place(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record(turns=2))
    store.save(make_record(turns=3))
    assert len(store.summaries()) == 1
    assert len(store.load("phishing_otp_00001")["generation"]["turns"]) == 3
    store.save(make_record(turns=1))
    assert len(store.load("phishing_otp_00001")["generation"]["turns"]) == 1


def test_load_unknown_returns_none(tmp_path):
    store = Store(tmp_path / "calls.db")
    assert store.load("nope_00001") is None
    assert store.exists("nope_00001") is False


def test_set_review(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record())
    assert store.set_review("phishing_otp_00001", "approved", "ดี่") is True
    loaded = store.load("phishing_otp_00001")
    assert loaded["review"]["status"] == "approved"
    assert loaded["review"]["note"] == "ดี่"
    assert isinstance(loaded["review"]["reviewed_at"], str) and loaded["review"]["reviewed_at"]
    store.set_review("phishing_otp_00001", "pending", "")
    loaded = store.load("phishing_otp_00001")
    assert loaded["review"]["reviewed_at"] is None
    assert loaded["review"]["status"] == "pending"
    assert store.set_review("nope_00001", "approved", "") is False


def test_records_filter_by_type(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record("phishing_otp_00001"))
    store.save(make_record("benign_00002", call_type="benign", seed=2))
    store.save(make_record("phishing_otp_00003", seed=3))
    ids = [r["id"] for r in store.records("phishing_otp")]
    assert ids == ["phishing_otp_00001", "phishing_otp_00003"]
    assert len(list(store.records())) == 3
    benign = store.load("benign_00002")
    assert benign["label_type"] is None
    assert benign["label_scam"] is False


def test_summaries_count_turns(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record(turns=3))
    row = store.summaries()[0]
    assert row["turns"] == 3
    assert row["review_status"] == "pending"
    assert row["call_type"] == "phishing_otp"


def test_delete_removes_turns_too(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.save(make_record())
    store.delete("phishing_otp_00001")
    assert store.load("phishing_otp_00001") is None
    store.save(make_record(turns=1))
    assert len(store.load("phishing_otp_00001")["generation"]["turns"]) == 1


def test_import_json_folder(tmp_path):
    store = Store(tmp_path / "calls.db")
    folder = tmp_path / "json"
    folder.mkdir()
    for record in (
        make_record("phishing_otp_00001"),
        make_record("benign_00002", call_type="benign", seed=2),
    ):
        (folder / f"{record['id']}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
    assert import_json_folder(store, folder) == (2, 0)
    assert import_json_folder(store, folder) == (0, 2)


def test_notes(tmp_path):
    store = Store(tmp_path / "calls.db")
    id1 = store.add_note("romance", "old", ["t1"], ["p1"], published="2026-01-01")
    id2 = store.add_note("romance", "new", ["t2"], ["p2"], published="2026-06-01")
    store.add_note("lottery", "other", [], [])
    assert isinstance(id1, int)
    assert [n["summary"] for n in store.notes("romance")] == ["new", "old"]
    store.set_note_active(id2, False)
    assert [n["summary"] for n in store.notes("romance")] == ["old"]
    assert store.notes("romance")[0]["tactics"] == ["t1"]


def test_threads_can_write_at_once(tmp_path):
    store = Store(tmp_path / "calls.db")

    def worker(k):
        for i in range(10):
            store.save(make_record(f"phishing_otp_{k}{i:04d}", seed=k * 100 + i))
        store.close()

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.summaries()) == 40


def test_dated_notes_come_before_undated(tmp_path):
    store = Store(tmp_path / "calls.db")
    store.add_note("romance", "seed example", ["t"], [])
    store.add_note("romance", "older report", ["t"], [], published="2026-01-01")
    store.add_note("romance", "newer report", ["t"], [], published="2026-06-01")
    assert [n["summary"] for n in store.notes("romance")] == [
        "newer report",
        "older report",
        "seed example",
    ]


def test_saving_a_stale_record_does_not_undo_a_review(tmp_path):
    store = Store(tmp_path / "calls.db")
    stale = make_record()  # the generator's copy, loaded before the review
    store.save(stale)
    store.set_review(stale["id"], "approved", "good call")
    store.save(stale)  # the generator saves its old copy again
    review = store.load(stale["id"])["review"]
    assert review["status"] == "approved" and review["note"] == "good call"
    assert review["reviewed_at"]


def test_a_new_row_keeps_the_review_it_arrives_with(tmp_path):
    store = Store(tmp_path / "calls.db")
    record = make_record()
    record["review"] = {
        "status": "rejected",
        "note": "robotic",
        "reviewed_at": "2026-10-01T00:00:00",
    }
    store.save(record)
    assert store.load(record["id"])["review"] == record["review"]


def test_import_json_folder_keeps_reviews_of_new_rows(tmp_path):
    folder = tmp_path / "old"
    folder.mkdir()
    record = make_record()
    record["review"] = {"status": "approved", "note": "ok", "reviewed_at": "2026-10-01T00:00:00"}
    (folder / f"{record['id']}.json").write_text(json.dumps(record), encoding="utf-8")
    store = Store(tmp_path / "calls.db")
    assert import_json_folder(store, folder) == (1, 0)
    assert store.load(record["id"])["review"]["status"] == "approved"


def test_leak_fields_round_trip(tmp_path):
    store = Store(tmp_path / "calls.db")
    record = make_record(turns=2)
    record["generation"]["turns"][1]["leaked"] = ["OTP", "ID_CARD"]
    record["generation"]["receiver_leaked"] = ["ID_CARD", "OTP"]
    record["generation"]["first_leak_turn"] = 1
    store.save(record)
    loaded = store.load("phishing_otp_00001")
    assert loaded["generation"]["turns"][1]["leaked"] == ["OTP", "ID_CARD"]
    assert loaded["generation"]["turns"][0]["leaked"] == []
    assert loaded["generation"]["receiver_leaked"] == ["ID_CARD", "OTP"]
    assert loaded["generation"]["first_leak_turn"] == 1


def test_an_old_database_gets_the_new_columns(tmp_path):
    """A calls.db made before the leak fields still opens, and its calls load with defaults."""
    import sqlite3

    path = tmp_path / "calls.db"
    Store(path).save(make_record())
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE calls DROP COLUMN receiver_leaked")
    db.execute("ALTER TABLE calls DROP COLUMN first_leak_turn")
    db.execute("ALTER TABLE turns DROP COLUMN leaked")
    db.execute("ALTER TABLE turns DROP COLUMN spoken_text")
    db.execute("ALTER TABLE calls DROP COLUMN speech_style")
    db.commit()
    db.close()

    store = Store(path)  # adds the columns again
    loaded = store.load("phishing_otp_00001")
    assert loaded["generation"]["receiver_leaked"] == []
    assert loaded["generation"]["first_leak_turn"] is None
    assert loaded["generation"]["turns"][0]["leaked"] == []
    Store(path)  # opening twice is fine


def test_spoken_text_and_style_round_trip(tmp_path):
    store = Store(tmp_path / "calls.db")
    record = make_record(turns=2)
    record["generation"]["speech_style"] = "dash"
    record["generation"]["turns"][0]["spoken_text"] = "ขอรหัส หนึ่งสองสาม"
    store.save(record)
    loaded = store.load("phishing_otp_00001")
    assert loaded["generation"]["speech_style"] == "dash"
    assert loaded["generation"]["turns"][0]["spoken_text"] == "ขอรหัส หนึ่งสองสาม"
