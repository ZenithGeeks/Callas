import argparse
import threading
from types import SimpleNamespace

import pytest
import yaml
from dspy.utils import DummyLM

from data_pipeline import simulate, synthesize
from data_pipeline.agents import SIGNATURES, Agents
from data_pipeline.config import ConfigError, load_config
from data_pipeline.scenarios import sample_config
from data_pipeline.store import Store

CONFIG = load_config()


class RateLimitError(Exception):
    """Named like the errors DSPy raises for a rate limit."""


def make_record(call_type="romance", seed=0, status="in_progress", config_hash=None, receiver=None):
    agents = Agents(CONFIG, lms={name: DummyLM([]) for name in SIGNATURES})
    call = sample_config(call_type, seed, CONFIG, receiver)
    record = simulate.new_record(call, CONFIG, agents, receiver)
    record["generation"]["status"] = status
    if config_hash:
        record["generation"]["config_hash"] = config_hash
    return record


@pytest.fixture
def no_llm(monkeypatch):
    """The real Agents would build LLM clients; the jobs here never use them."""
    monkeypatch.setattr(synthesize, "Agents", lambda config: SimpleNamespace())


def job_args(**extra):
    return argparse.Namespace(workers=1, no_tts=True, **extra)


# ---- Ctrl-C ----------------------------------------------------------------------------------


def test_ctrl_c_cancels_the_queued_jobs(tmp_path, monkeypatch, no_llm, capsys):
    started = []
    second_running, release = threading.Event(), threading.Event()

    def fake_run_call(call_type, seed, store, agents, config, voices=None, receiver=None):
        started.append(seed)
        if seed == 1:
            second_running.set()
            release.wait(timeout=10)
        return make_record(call_type, seed, status="accepted")

    real_as_completed = synthesize.as_completed

    def interrupt_while_job_one_runs(futures):
        yield next(real_as_completed(futures))  # job 0 is done
        second_running.wait(timeout=10)  # job 1 is running, jobs 2.. are queued
        raise KeyboardInterrupt

    monkeypatch.setattr(synthesize, "run_call", fake_run_call)
    monkeypatch.setattr(synthesize, "as_completed", interrupt_while_job_one_runs)
    jobs = [("romance", seed) for seed in range(6)]

    result = synthesize._run_jobs(jobs, job_args(), CONFIG, Store(tmp_path / "calls.db"))
    release.set()  # let the running job end; the worker must find no queued job
    for thread in threading.enumerate():
        if thread.name.startswith("ThreadPoolExecutor"):
            thread.join(timeout=10)

    assert result is None
    assert started == [0, 1]
    assert "stopped by user; rerun to resume" in capsys.readouterr().err


def test_main_exits_non_zero_after_ctrl_c(tmp_path, monkeypatch, no_llm):
    monkeypatch.setattr(synthesize, "_run_jobs", lambda *a, **kw: None)
    argv = ["--type", "romance", "--n", "1", "--db", str(tmp_path / "c.db")]
    argv += ["--wiki", str(tmp_path / "no_wiki")]
    assert synthesize.main(argv) != 0


# ---- errors ----------------------------------------------------------------------------------


def test_only_a_model_error_rejects_the_call(tmp_path, monkeypatch, capsys):
    store = Store(tmp_path / "calls.db")
    outcomes = {
        1: RateLimitError("429"),  # a model error that survived the retries
        2: KeyError("bug"),  # not the model's fault
    }

    def fake_run_call(call_type, seed, store, agents, config, voices=None, receiver=None):
        store.save(make_record(call_type, seed))  # saved turns of an unfinished call
        raise outcomes[seed]

    monkeypatch.setattr(synthesize, "run_call", fake_run_call)

    assert synthesize._run_one("romance", 1, store, None, CONFIG, None) == (
        "romance_00001",
        "rejected",
    )
    rejected = store.load("romance_00001")["generation"]
    assert rejected["reasons"] == ["llm_error:RateLimitError"]

    assert synthesize._run_one("romance", 2, store, None, CONFIG, None) == (
        "romance_00002",
        "error",
    )
    assert store.load("romance_00002")["generation"]["status"] == "in_progress"
    assert "romance_00002" in capsys.readouterr().err


def test_error_count_is_printed_and_the_exit_code_is_not_zero(
    tmp_path, monkeypatch, no_llm, capsys
):
    def fake_run_call(call_type, seed, store, agents, config, voices=None, receiver=None):
        if seed == 1:
            raise KeyError("bug")
        record = make_record(call_type, seed, status="accepted")
        store.save(record)
        return record

    monkeypatch.setattr(synthesize, "run_call", fake_run_call)
    argv = ["--type", "romance", "--n", "3", "--db", str(tmp_path / "c.db")]
    argv += ["--wiki", str(tmp_path / "no_wiki"), "--no-tts"]
    assert synthesize.main(argv) == 1
    out = capsys.readouterr().out
    assert "romance: 2 accepted, 0 rejected, 1 error" in out


def test_the_printed_id_is_the_one_the_call_used(tmp_path, monkeypatch, no_llm, capsys):
    key = CONFIG.scenarios.receivers.profiles[0].key
    argv = ["--type", "romance", "--n", "1", "--receiver", key]
    argv += ["--db", str(tmp_path / "c.db"), "--wiki", str(tmp_path / "no_wiki"), "--no-tts"]

    def fake_run_call(call_type, seed, store, agents, config, voices=None, receiver=None):
        record = make_record(call_type, seed, status="accepted", receiver=receiver)
        store.save(record)
        return record

    monkeypatch.setattr(synthesize, "run_call", fake_run_call)
    assert synthesize.main(argv) == 0
    assert f"[1/1] romance_{key}_00000 accepted" in capsys.readouterr().out


# ---- skipped calls ---------------------------------------------------------------------------


def test_finished_calls_are_counted_and_other_configs_warned_about(
    tmp_path, monkeypatch, no_llm, capsys
):
    store = Store(tmp_path / "calls.db")
    store.save(make_record(seed=0, status="accepted", config_hash="old"))
    store.save(make_record(seed=1, status="rejected"))  # made with the current config
    ran = []

    def fake_run_call(call_type, seed, store, agents, config, voices=None, receiver=None):
        ran.append(seed)
        record = make_record(call_type, seed, status="accepted")
        store.save(record)
        return record

    monkeypatch.setattr(synthesize, "run_call", fake_run_call)
    argv = ["--type", "romance", "--n", "3", "--db", str(tmp_path / "calls.db")]
    argv += ["--wiki", str(tmp_path / "no_wiki"), "--no-tts"]
    assert synthesize.main(argv) == 0
    captured = capsys.readouterr()
    assert ran == [2]
    assert "skipped 2 calls that were already finished" in captured.out
    assert "1 of the skipped calls were made with a different config" in captured.err
    assert "romance: 2 accepted, 1 rejected, 0 error" in captured.out


def test_no_skip_message_when_nothing_was_finished(tmp_path, monkeypatch, no_llm, capsys):
    monkeypatch.setattr(synthesize, "run_call", lambda *a, **kw: make_record(status="accepted"))
    argv = ["--type", "romance", "--n", "1", "--db", str(tmp_path / "calls.db")]
    argv += ["--wiki", str(tmp_path / "no_wiki"), "--no-tts"]
    synthesize.main(argv)
    assert "skipped" not in capsys.readouterr().out


# ---- --receiver-file -------------------------------------------------------------------------


def receiver_file(tmp_path, **change):
    profile = CONFIG.scenarios.receivers.profiles[0].model_dump()
    profile["key"] = "aunt_noi"
    profile.update(change)
    path = tmp_path / "person.yaml"
    path.write_text(yaml.safe_dump(profile, allow_unicode=True), encoding="utf-8")
    return argparse.Namespace(receiver=None, receiver_file=path)


def test_receiver_file_with_a_new_key_is_accepted(tmp_path):
    assert synthesize._chosen_receiver(receiver_file(tmp_path), CONFIG).key == "aunt_noi"


def test_receiver_file_cannot_reuse_a_scenarios_key(tmp_path):
    taken = CONFIG.scenarios.receivers.profiles[0].key
    with pytest.raises(ConfigError, match=taken):
        synthesize._chosen_receiver(receiver_file(tmp_path, key=taken), CONFIG)


def test_export_has_the_leak_fields(tmp_path):
    import json

    store = Store(tmp_path / "calls.db")
    record = make_record(status="accepted")
    record["generation"]["receiver_leaked"] = ["ID_CARD", "OTP"]
    record["generation"]["first_leak_turn"] = 3
    store.save(record)
    quiet = make_record(seed=1, status="accepted")
    store.save(quiet)
    out = tmp_path / "calls.jsonl"
    assert synthesize.main(["--db", str(tmp_path / "calls.db"), "--export", str(out)]) == 0
    rows = {r["id"]: r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert rows[record["id"]]["receiver_leaked"] == ["ID_CARD", "OTP"]
    assert rows[record["id"]]["first_leak_turn"] == 3
    assert (
        rows[quiet["id"]]["receiver_leaked"] == [] and rows[quiet["id"]]["first_leak_turn"] is None
    )
    assert "generation" not in rows[record["id"]]
