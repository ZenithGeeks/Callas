"""The ending note: a speaker is told the call is ending only on its last two turns (#150)."""

from dataclasses import replace

import pytest

from data_pipeline import turns as turn_module
from data_pipeline.config import load_config
from data_pipeline.scenarios import sample_config
from data_pipeline.simulate import run_call
from data_pipeline.store import Store
from data_pipeline.tests.test_simulate import judge, make_agents

CONFIG = load_config()
SOON = CONFIG.agents.speakers.ending.soon
LAST = CONFIG.agents.speakers.ending.last


@pytest.mark.parametrize("max_turns", [6, 7, 14])
def test_the_note_comes_only_on_the_last_two_turns(max_turns):
    call = replace(sample_config("phishing_otp", 1, CONFIG), max_turns=max_turns)
    notes = [turn_module.ending(call, CONFIG, done) for done in range(max_turns)]
    assert notes[-1] == LAST
    assert notes[-2] == SOON
    assert all(note == "" for note in notes[:-2])


def test_the_receiver_may_close_even_while_it_would_stay_on_the_line():
    receiver = CONFIG.agents.receiver
    stay = receiver.stage.stay.format(turn=1)
    free = receiver.stage.free.format(turn=1)
    assert turn_module.stage(CONFIG, 0) == stay
    assert turn_module.stage(CONFIG, 0, closing=True) == free


def _run(call_type, tmp_path, seed=1):
    caller = [dict(line="ครับ", asks_now=False, end_call=False)] * 7
    receiver = [dict(line="ค่ะ", end_call=False)] * 7
    agents = make_agents(caller, receiver, judge(start=0))
    run_call(call_type, seed, Store(tmp_path / "calls.db"), agents, CONFIG)
    return agents, sample_config(call_type, seed, CONFIG).max_turns


@pytest.mark.parametrize("seed", [0, 1, 2])  # phishing_otp: 0 scammed, 1 safe, 2 a fixed profile
def test_only_a_receiver_meant_to_be_scammed_confirms_the_ask_at_the_end(seed, tmp_path):
    scammed = sample_config("phishing_otp", seed, CONFIG).receiver.scam_outcome == "scammed"
    agents, max_turns = _run("phishing_otp", tmp_path, seed)
    note = CONFIG.agents.speakers.ending.scammed
    stages = [inputs["stage"] for inputs in agents.receiver.calls]
    closing = [stage for own, stage in enumerate(stages) if _note_at(2 * own + 1, max_turns)]
    assert len(closing) == 1  # the receiver speaks once in the last two turns
    assert closing[0].endswith(note) == scammed
    assert all(note not in stage for stage in stages if stage not in closing)
    assert all(note not in inputs["pacing"] for inputs in agents.scam_caller.calls)


def _note_at(done, max_turns):
    return {max_turns - 1: LAST, max_turns - 2: SOON}.get(done)


@pytest.mark.parametrize("call_type", ["phishing_otp", "benign"])
def test_each_agent_gets_the_note_in_its_prompt_near_the_end(call_type, tmp_path):
    agents, max_turns = _run(call_type, tmp_path)
    caller_calls, receiver_calls = agents.scam_caller.calls, agents.receiver.calls
    assert len(caller_calls) + len(receiver_calls) == max_turns
    for own, inputs in enumerate(caller_calls):
        note = _note_at(2 * own, max_turns)
        if note:
            assert inputs["pacing"].endswith(note)
        else:
            assert SOON not in inputs["pacing"] and LAST not in inputs["pacing"]
    for own, inputs in enumerate(receiver_calls):
        note = _note_at(2 * own + 1, max_turns)
        if note:
            assert inputs["stage"].endswith(note)
        else:
            assert SOON not in inputs["stage"] and LAST not in inputs["stage"]
    if call_type == "benign":  # the legit caller's pacing is empty until the end
        assert caller_calls[0]["pacing"] == ""
