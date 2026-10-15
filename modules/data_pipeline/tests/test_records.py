import json

import pytest

from data_pipeline import live, simulate
from data_pipeline.config import load_config
from data_pipeline.records import Briefing, CallRecord, Judge, Turn
from data_pipeline.store import Store

CONFIG = load_config()


def as_json(data):
    """Key order matters: the store and the exports write the dicts in this order."""
    return json.dumps(data, ensure_ascii=False)


def round_trip(data):
    return CallRecord.from_dict(data).to_dict()


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """Records written by fake-LLM runs: a scam call, a normal call and a spoken call."""
    store = Store(tmp_path_factory.mktemp("records") / "calls.db")
    for call_type, seed in (("phishing_otp", 1), ("benign", 2)):
        simulate.run_call(call_type, seed, store, live.fake_agents(CONFIG, call_type), CONFIG)
    voices = live.SilentVoices()
    agents = live.fake_agents(CONFIG, "romance")
    simulate.run_call("romance", 3, store, agents, CONFIG, voices=voices)
    return list(store.call_dicts())


def test_every_stored_record_round_trips_unchanged(stored):
    assert len(stored) == 3
    for data in stored:
        assert as_json(round_trip(data)) == as_json(data)


def test_a_record_reads_as_typed_fields(stored):
    record = CallRecord.from_dict(stored[0])
    assert record.id == "benign_00002"
    assert all(isinstance(turn, Turn) for turn in record.generation.turns)
    assert isinstance(record.generation.judge, Judge)
    scam = CallRecord.from_dict(stored[1])
    assert isinstance(scam.generation.briefing, Briefing) and scam.generation.briefing.plan


def test_to_dict_shares_nothing_with_the_record(stored):
    record = CallRecord.from_dict(stored[1])
    data = record.to_dict()
    data["generation"]["turns"][0]["entities"].append({"label": "X"})
    data["generation"]["call"]["seed"] = -1
    assert as_json(record.to_dict()) == as_json(stored[1])


def test_old_judge_and_briefing_shapes_round_trip(stored):
    data = json.loads(as_json(stored[1]))
    generation = data["generation"]
    # an old store: no prompt sizes, and an early planner that kept note ids
    del generation["judge"]["prompt_tokens"]
    generation["briefing"] = {"plan": ["a", "b"], "note_ids": [3, 1], "reasoning": "r"}
    assert as_json(round_trip(data)) == as_json(data)
    generation["briefing"] = {"plan": ["a"], "notes_used": ["note"], "reasoning": "r"}
    assert as_json(round_trip(data)) == as_json(data)


def test_a_skipped_judge_keeps_its_key_order():
    skipped = {
        "label_holds": False,
        "scam_start_turn": -1,
        "natural_thai": 0,
        "giveaway": False,
        "summary": "",
        "issues": "not judged",
        "reasoning": "",
        "skipped": "too_long_for_judge",
        "prompt_tokens": 9000,
    }
    assert as_json(Judge.from_dict(skipped).to_dict()) == as_json(skipped)


def test_a_turn_from_a_newer_store_with_unknown_keys_is_refused():
    with pytest.raises(TypeError):
        Turn.from_dict({"speaker": "A", "volume": 3})
