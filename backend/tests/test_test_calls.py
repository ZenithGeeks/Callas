import io
import json

import numpy as np
import pytest
import soundfile as sf

from backend import live, test_calls


def call(call_id: str, label_type: str | None) -> dict:
    return {
        "id": call_id,
        "label_scam": label_type is not None,
        "label_type": label_type,
        "speaker_turns": [
            {"speaker": "A", "text": f"caller line of {call_id}"},
            {"speaker": "B", "text": "receiver line"},
        ],
    }


@pytest.fixture
def split(tmp_path, monkeypatch):
    """A dataset with 3 calls, of which only romance_1 and normal_1 are in the test split."""
    (tmp_path / "romance.jsonl").write_text(
        "\n".join(json.dumps(call(i, "romance")) for i in ("romance_1", "romance_2")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "normal.jsonl").write_text(json.dumps(call("normal_1", None)), encoding="utf-8")
    test_csv = tmp_path / "test.csv"
    test_csv.write_text(
        "id,call_id\nromance_1_t00,romance_1\nromance_1_t02,romance_1\nnormal_1_t00,normal_1\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(test_calls, "TEST_SPLIT", test_csv)
    monkeypatch.setattr(test_calls, "DATASET_DIR", tmp_path)
    test_calls._test_calls.cache_clear()
    yield
    test_calls._test_calls.cache_clear()


def test_only_test_split_calls_are_offered(split):
    seen = {test_calls.random_test_call() for _ in range(50)}
    assert {c.call_id for c in seen} == {"romance_1", "normal_1"}


def test_a_test_call_holds_the_caller_lines_and_its_label(split, monkeypatch):
    monkeypatch.setattr(test_calls.random, "choice", lambda calls: calls[0])
    assert test_calls.random_test_call() == test_calls.TestCall(
        "normal_1", "normal", ("caller line of normal_1",)
    )


def test_no_test_split_gives_none(tmp_path, monkeypatch):
    monkeypatch.setattr(test_calls, "TEST_SPLIT", tmp_path / "missing.csv")
    assert test_calls.random_test_call() is None


def test_script_route_shows_a_test_call(split, client):
    r = client.get("/live/script")
    assert r.status_code == 200
    assert "Answer key" in r.text
    assert "romance_2" not in r.text


def test_find_test_call_only_finds_test_split_calls(split):
    assert test_calls.find_test_call("romance_1").caller_lines == ("caller line of romance_1",)
    assert test_calls.find_test_call("romance_2") is None


def test_script_line_route_speaks_a_test_call_line(split, client):
    r = client.get("/live/script/romance_1/0.wav")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content.startswith(b"RIFF")


@pytest.mark.parametrize("path", ["/live/script/romance_1/1.wav", "/live/script/romance_2/0.wav"])
def test_script_line_route_refuses_other_lines_and_calls(split, client, path):
    assert client.get(path).status_code == 404


def test_long_lines_are_split_at_spaces_under_the_limit():
    assert live.tts_chunks("aaa bb cc dddd", limit=6) == ["aaa bb", "cc", "dddd"]
    assert live.tts_chunks("toolongword ok", limit=4) == ["toolongword", "ok"]


def test_a_long_line_is_spoken_in_chunks_and_joined(monkeypatch):
    class Speaker:
        def synthesize_bytes(self, text, check_leaks=True):
            assert not check_leaks
            out = io.BytesIO()
            sf.write(out, np.ones(len(text), dtype="int16"), 16000, format="WAV")
            return out.getvalue()

    monkeypatch.setattr(live, "TTS_CHUNK_CHARS", 5)
    audio, rate = sf.read(io.BytesIO(live.speak_line(Speaker(), "abcd efgh ij")), dtype="int16")
    assert rate == 16000 and len(audio) == len("abcd") + len("efgh") + len("ij")
