import json

import pytest

from backend import test_calls


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
