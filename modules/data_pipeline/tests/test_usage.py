import dspy
import pytest
from dspy.utils import DummyLM

from data_pipeline import usage
from data_pipeline.records import Briefing, Judge
from data_pipeline.turns import ask


class FakeAgents:
    """One agent whose every attempt reports 100 tokens in, 20 out, like a real LM request."""

    def __init__(self, fail_first: int = 0) -> None:
        self.lms = {"receiver": DummyLM([])}
        self.fail_first = fail_first

    def receiver(self, **inputs):
        dspy.settings.usage_tracker.add_usage("m", {"prompt_tokens": 100, "completion_tokens": 20})
        if self.fail_first:
            self.fail_first -= 1
            raise ValueError("unparseable answer")
        return dspy.Prediction(reasoning="abc", line="d")


def test_ask_attaches_the_real_usage():
    pred = ask(FakeAgents(), "receiver", "k1")
    assert pred.get_lm_usage() == {
        "input": 100,
        "output": 20,
        "requests": 1,
        "failed": 0,
        "reasoning_share": 0.75,  # "abc" of "abc" + "d"
    }


def test_a_failed_attempt_is_added_to_the_retry_that_succeeds():
    agents = FakeAgents(fail_first=1)
    with pytest.raises(ValueError):
        ask(agents, "receiver", "k2")
    got = ask(agents, "receiver", "k2").get_lm_usage()
    assert (got["input"], got["output"], got["requests"], got["failed"]) == (200, 40, 2, 1)


def test_retries_of_two_calls_do_not_mix():
    with pytest.raises(ValueError):
        ask(FakeAgents(fail_first=1), "receiver", "call-a")
    assert ask(FakeAgents(), "receiver", "call-b").get_lm_usage()["requests"] == 1
    assert ask(FakeAgents(), "receiver", "call-a").get_lm_usage()["requests"] == 2


def _step(input_, output, requests=1, failed=0, share=None):
    step = {"input": input_, "output": output, "requests": requests, "failed": failed}
    return step if share is None else {**step, "reasoning_share": share}


def _record(call_type="romance", with_usage=True):
    u = (lambda *a, **k: _step(*a, **k)) if with_usage else (lambda *a, **k: {})
    return {
        "call_type": call_type,
        "generation": {
            "briefing": {"plan": ["x"], "usage": u(1000, 300)},
            "turns": [
                {"agent": "scam_caller", "usage": u(2000, 400, share=0.5)},
                {"agent": "receiver", "usage": u(1500, 200, requests=2, failed=1)},
                {"agent": "scam_caller", "usage": u(2200, 400, share=0.5)},
            ],
            "judge": {"summary": "", "usage": u(3000, 1000)},
        },
    }


def test_call_usage_sums_each_agent_in_call_order():
    got = usage.call_usage(_record())
    assert [r["agent"] for r in got["rows"]] == ["planner", "scam_caller", "receiver", "judge"]
    caller = got["rows"][1]
    assert (caller["steps"], caller["input"], caller["output"], caller["reasoning"]) == (
        2,
        4200,
        800,
        400,
    )
    assert got["total"]["input"] == 1000 + 2000 + 1500 + 2200 + 3000
    assert got["total"]["output"] == 300 + 400 + 200 + 400 + 1000
    assert (got["total"]["requests"], got["total"]["failed"]) == (6, 1)


def test_the_footing_step_has_its_own_row_after_the_planner():
    record = _record()
    record["generation"]["footing_usage"] = _step(500, 80)
    got = usage.call_usage(record)
    assert [r["agent"] for r in got["rows"]] == [
        "planner",
        "footing",
        "scam_caller",
        "receiver",
        "judge",
    ]
    assert (got["rows"][1]["input"], got["rows"][1]["output"]) == (500, 80)
    assert got["total"]["input"] == 1000 + 500 + 2000 + 1500 + 2200 + 3000


def test_a_call_without_footing_usage_has_no_footing_row():
    for footing_usage in (None, {}):  # older calls have no key; stranger calls keep {}
        record = _record()
        if footing_usage is not None:
            record["generation"]["footing_usage"] = footing_usage
        assert "footing" not in [r["agent"] for r in usage.call_usage(record)["rows"]]


def test_a_call_without_usage_is_not_recorded():
    assert usage.call_usage(_record(with_usage=False)) is None


def test_summary_per_agent_and_per_call_type():
    s = usage.summary([_record(), _record(), _record("investment"), _record(with_usage=False)])
    assert (s["calls"], s["not_recorded"]) == (3, 1)
    caller = next(a for a in s["agents"] if a["agent"] == "scam_caller")
    assert (caller["calls"], caller["steps"], caller["input"], caller["input_mean"]) == (
        3,
        6,
        3 * 4200,
        4200,
    )
    one_call = 1000 + 300 + 4200 + 800 + 1500 + 200 + 3000 + 1000
    assert s["per_call"] == {"mean": one_call, "max": one_call, "total": 3 * one_call}
    by_type = [(t["call_type"], t["calls"]) for t in s["by_type"]]
    assert by_type == [("investment", 1), ("romance", 2)]
    assert usage.summary([])["calls"] == 0


def test_report_by_call_type():
    text = usage.report([_record(), _record("investment"), _record(with_usage=False)])
    assert "romance: 1 calls" in text and "investment: 1 calls" in text
    assert "1 calls have no usage recorded" in text


def test_briefing_and_judge_keep_their_usage():
    step = _step(10, 5)
    assert Briefing.from_dict(Briefing(plan=["a"], usage=step).to_dict()).usage == step
    judge = Judge(
        label_holds=True,
        scam_start_turn=0,
        giveaway=False,
        summary="",
        issues="",
        reasoning="",
        prompt_tokens=1,
        usage=step,
    )
    assert Judge.from_dict(judge.to_dict()).usage == step
    assert "usage" not in Briefing(plan=["a"]).to_dict()  # old records stay as they were
