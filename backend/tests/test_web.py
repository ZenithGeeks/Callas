import io
import json
import re
import wave
from pathlib import Path

import pytest

from backend import call, demo, live_ws, main

ROOT = Path(__file__).resolve().parents[1]
PHISHING = (
    "แจ้งเตือน: บัญชีของท่านถูกระงับชั่วคราว กรุณายืนยันตัวตนที่ https://bank-verify.co/th "
    "ภายใน 24 ชม. หรือส่งรหัส OTP ที่ได้รับกลับมาที่เบอร์ 062-555-0147"
)
CONTRACT = re.compile(
    r"<body[^>]*>\s*<!--\nTHESIS: Every check is issued as a numbered public warning"
)


def wav_bytes() -> bytes:
    return demo.silent_wav(0.2)


@pytest.mark.parametrize("path", ["/message", "/call", "/live"])
def test_pages_render_with_contract_comment(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert CONTRACT.search(r.text)
    assert "FINISH: unreviewed and undocumented is unfinished" in r.text
    assert "/static/css/app.css" in r.text


def test_call_page_names_the_default_speaker_split(client):
    html = client.get("/call").text
    assert "Nemotron-3 diarization" in html
    assert "pyannote" not in html


@pytest.mark.parametrize("path", ["/message", "/call", "/live"])
def test_assets_use_paths_not_absolute_urls(client, path):
    # Behind an https proxy (tailscale serve) the app sees plain http. An absolute
    # http:// asset URL on an https page is blocked as mixed content.
    html = client.get(path, headers={"Host": "callas.example.ts.net"}).text
    assets = re.findall(r'(?:href|src)="([^"]*/static/[^"]*)"', html)
    assert assets and all(a.startswith("/static/") for a in assets)


def test_band_is_a_size_container(client):
    css = (ROOT / "static" / "css" / "app.css").read_text(encoding="utf-8")
    assert "container-type:inline-size" in css.replace(" ", "")
    html = client.post("/message/check", data={"text": PHISHING}).text
    assert re.search(r'<section class="band [^"]*"', html)
    assert 'class="verdict-row"' in html and 'class="band-facts"' in html


def test_root_redirects(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/message"


def test_healthz_has_no_secrets(client):
    body = client.get("/healthz").json()
    assert body["mode"] == "demo"
    text = json.dumps(body).lower()
    for word in ("key", "token", "secret", "password", "http"):
        assert word not in text


def test_message_check_highlights_entities(client):
    r = client.post("/message/check", data={"text": PHISHING})
    assert r.status_code == 200
    assert 'data-kind="scam"' in r.text and "SCAM" in r.text
    assert "Phishing / OTP" in r.text
    for label in ("URL", "OTP", "PHONE"):
        assert f'data-label="{label}"' in r.text
    assert "https://bank-verify.co/th" in r.text
    assert "Why" in r.text and "Summary" in r.text


def test_message_check_safe(client):
    r = client.post("/message/check", data={"text": "พรุ่งนี้เจอกันที่ร้านกาแฟนะ"})
    assert r.status_code == 200 and 'data-kind="safe"' in r.text and "NORMAL" in r.text


def test_message_explanation_unavailable_does_not_fail(client, monkeypatch):
    from orchestrator import ToolCallingUnavailable

    def boom(self, text):
        raise ToolCallingUnavailable("x")

    monkeypatch.setattr("backend.runtime.AppRuntime.explain", boom)
    r = client.post("/message/check", data={"text": PHISHING})
    assert r.status_code == 200
    assert "Explanation unavailable: the LLM server does not allow tool calls." in r.text


def test_message_rejects_empty_and_too_long(client):
    r = client.post("/message/check", data={"text": "   "})
    assert r.status_code == 422 and "Paste a message first." in r.text
    assert r.headers["HX-Retarget"] == "#form-error"
    r = client.post("/message/check", data={"text": "ก" * 2001})
    assert r.status_code == 422 and "2000" in r.text


def stream_events(client):
    r = client.post("/call/simulate")
    assert r.status_code == 200
    run_id = re.search(r"/call/(\w+)/events", r.text).group(1)
    s = client.get(f"/call/{run_id}/events")
    assert s.status_code == 200 and s.headers["content-type"].startswith("text/event-stream")
    return run_id, s.text


def test_simulate_streams_seven_turns_and_escalates_at_four(client):
    _, body = stream_events(client)
    names = re.findall(r"^event: (\w+)", body, re.M)
    assert names[-1] == "done"
    assert {"stages", "verdict", "strip", "transcript", "entities", "summary", "log"} <= set(names)
    # the last transcript event holds all 7 turns in order
    transcripts = re.findall(r"event: transcript\n((?:data: .*\n)+)", body)
    last = "".join(transcripts[-1])
    assert [int(n) for n in re.findall(r'data-turn="(\d+)"', last)] == list(range(1, 8))
    assert last.count('data-state="scam"') == 1
    assert re.search(r'data-state="scam"[^>]*data-turn="4"', last)
    assert last.count('data-state="held"') == 3
    assert last.count("Reply sent") == 0
    verdicts = re.findall(r"event: verdict\n((?:data: .*\n)+)", body)
    scam = [v for v in verdicts if "SCAM" in v]
    assert len(scam) == 1 and "Turn 4" in scam[0]  # sent once, so it wipes in once
    assert "is-fresh" in scam[0]
    # no turn after 4 was replied to
    assert "T5–T7" in body and "no reply" in body


def test_events_replay_after_finish(client):
    run_id, _ = stream_events(client)
    again = client.get(f"/call/{run_id}/events")
    assert again.status_code == 200 and "event: done" in again.text
    assert client.get("/call/nope/events").status_code == 404


def test_simulate_rejects_non_audio(client):
    r = client.post("/call/simulate", files={"audio": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 422 and "not audio" in r.text
    r = client.post(
        "/call/simulate", files={"audio": ("fake.wav", b"hello world, not riff", "audio/wav")}
    )
    assert r.status_code == 422 and "does not look like" in r.text


def test_simulate_accepts_wav_upload(client):
    r = client.post("/call/simulate", files={"audio": ("mine.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200 and "mine.wav" not in r.text or r.status_code == 200


def test_ws_live_demo_ready_to_ended(client):
    with client.websocket_connect("/ws/live") as ws:
        events = []
        ws.send_bytes(b"\x00\x00" * 1600)
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] == "reply":
                ws.send_text(json.dumps({"type": "pause"}))
                audio = ev["audio_wav_b64"]
                assert audio
                ws.send_text(json.dumps({"type": "resume"}))
            if ev["type"] == "verdict" and ev["turn"] == 4:
                ws.send_text(json.dumps({"type": "end"}))
            if ev["type"] == "ended":
                break
    kinds = [e["type"] for e in events]
    assert kinds[0] == "ready" and kinds[-1] == "ended"
    assert "partial" in kinds
    turns = [e["turn"] for e in events if e["type"] == "turn"]
    assert turns[:4] == [1, 2, 3, 4]
    verdicts = {e["turn"]: e for e in events if e["type"] == "verdict"}
    assert not verdicts[3]["escalated"] and verdicts[4]["escalated"]
    assert verdicts[4]["label_type"] == "fake_authority"
    replies = [e for e in events if e["type"] == "reply"]
    assert [r["turn"] for r in replies] == [3]
    buf = io.BytesIO(__import__("base64").b64decode(replies[0]["audio_wav_b64"]))
    assert wave.open(buf).getframerate() == 16000


def test_finished_run_text_and_stale_uploads_expire(tmp_path, monkeypatch):
    """A finished run's snapshot holds call text: it must not outlive the reconnect window."""
    from backend import call_run

    clock = [1000.0]
    monkeypatch.setattr(call_run.time, "monotonic", lambda: clock[0])
    reg = call_run.RunRegistry()
    done_file, idle_file = tmp_path / "done.wav", tmp_path / "idle.wav"
    done_file.write_bytes(b"x")
    idle_file.write_bytes(b"x")
    done = reg.create(str(done_file), "done.wav")
    done.started, done.finished, done.finished_at = True, True, clock[0]
    done.snapshot = {"transcript": "call text"}
    idle = reg.create(str(idle_file), "idle.wav")

    clock[0] += call_run.RunRegistry.FINISHED_TTL + 1
    assert reg.get(done.id) is None and done.snapshot == {}
    assert not done_file.exists()
    assert reg.get(idle.id) is idle  # still inside its pending window

    clock[0] += call_run.RunRegistry.PENDING_TTL
    assert reg.get(idle.id) is None
    assert not idle_file.exists()


def test_uploads_are_removed_when_the_app_stops(monkeypatch, tmp_path):
    """An upload whose stream never opens must not outlive the server process."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(call, "UPLOAD_DIR", tmp_path / "uploads")
    stale = tmp_path / "uploads"
    stale.mkdir()
    (stale / "callas-old.wav").write_bytes(b"x")  # left by a crashed run
    with TestClient(main.app) as client:
        assert not (stale / "callas-old.wav").exists()  # cleared at startup
        r = client.post("/call/simulate")  # demo mode: creates a temp file, stream never opened
        assert r.status_code == 200
        assert list(stale.iterdir())
    assert list(stale.iterdir()) == []  # cleared at shutdown


def test_ws_live_reports_a_call_that_cannot_start(client, monkeypatch):
    def broken(app_runtime):
        raise RuntimeError("model would not load")

    monkeypatch.setattr(live_ws, "make_live_call", broken)
    with client.websocket_connect("/ws/live") as ws:
        ev = ws.receive_json()
    assert ev == {"type": "error", "turn": None, "message": "The live call could not start."}


def test_ws_live_builds_the_call_off_the_event_loop(client, monkeypatch):
    import asyncio

    seen = []
    real = live_ws.make_live_call

    def recording(app_runtime):
        try:  # inside the event loop's thread a loop is running; in a worker thread none is
            asyncio.get_running_loop()
            seen.append(True)
        except RuntimeError:
            seen.append(False)
        return real(app_runtime)

    monkeypatch.setattr(live_ws, "make_live_call", recording)
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "end"}))
    assert seen == [False]
