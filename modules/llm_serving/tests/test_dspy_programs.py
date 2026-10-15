import logging
from pathlib import Path
from types import SimpleNamespace

import dspy
import pytest
from dspy.utils import DummyLM

from llm_serving import dspy_optimize, dspy_programs
from llm_serving.client import LLMUnavailable
from llm_serving.dspy_programs import DSPyLeakJudge, DSPyReplyWriter, WriteReply, build_lm
from nlp_core.types import NLPResult

NOFILE = Path("does-not-exist.dspy.json")
REPLY = "ขณะนี้ไม่สะดวกรับสายค่ะ"


def writer(answers):
    return DSPyReplyWriter(lm=DummyLM(answers), compiled_path=NOFILE)


def test_write_returns_only_stripped_reply():
    w = writer([{"reasoning": "SECRET-THOUGHT", "reply": f"  {REPLY}\n"}])
    assert w.write("A: สวัสดี", NLPResult()) == REPLY


def test_write_passes_context_and_benign_verdict():
    seen = {}

    class Spy(dspy.Module):
        def forward(self, call_context, verdict):
            seen.update(call_context=call_context, verdict=verdict)
            return dspy.Prediction(reply="ok", reasoning="r")

    assert DSPyReplyWriter(program=Spy(), lm=DummyLM([])).write("ctx", NLPResult()) == "ok"
    assert seen == {"call_context": "ctx", "verdict": "benign"}


def test_signature_fields():
    assert list(WriteReply.input_fields) == ["call_context", "verdict"]
    assert list(WriteReply.output_fields) == ["reply"]
    assert list(dspy_programs.JudgeLeak.input_fields) == ["reply", "call_context"]
    assert list(dspy_programs.JudgeLeak.output_fields) == ["leaks"]


def test_connection_error_maps_to_unavailable():
    class Boom(dspy.Module):
        def forward(self, **kw):
            raise ConnectionError("down")

    with pytest.raises(LLMUnavailable):
        DSPyReplyWriter(program=Boom(), lm=DummyLM([])).write("x", NLPResult())


def test_compiled_file_is_loaded(tmp_path):
    prog = dspy.ChainOfThought(WriteReply)
    prog.predict.demos = [dspy.Example(call_context="ก", verdict="benign", reply="ข")]
    path = tmp_path / "r.dspy.json"
    prog.save(str(path))
    w = DSPyReplyWriter(lm=DummyLM([]), compiled_path=path)
    assert len(w.program.predict.demos) == 1


def test_broken_compiled_file_falls_back(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    w = DSPyReplyWriter(lm=DummyLM([]), compiled_path=bad)
    assert w.program.predict.demos == []


def test_judge_uses_chain_of_thought():
    assert isinstance(DSPyLeakJudge(lm=DummyLM([])).program, dspy.ChainOfThought)


@pytest.mark.parametrize(
    "answer,blocked",
    [(False, False), (True, True), ("false", False), ("true", True), ("maybe", True), ("", True)],
)
def test_judge_dummy_lm(answer, blocked):
    j = DSPyLeakJudge(lm=DummyLM([{"reasoning": "JUDGE-THOUGHT", "leaks": answer}]))
    assert j.blocks(REPLY, "A: สวัสดี") is blocked


def test_judge_server_down_raises_unavailable():
    class Boom(dspy.Module):
        def forward(self, **kw):
            raise TimeoutError("slow")

    j = DSPyLeakJudge(lm=DummyLM([]))
    j.program = Boom()
    with pytest.raises(LLMUnavailable):
        j.blocks(REPLY, "x")


def test_judge_unexpected_error_blocks():
    class Boom(dspy.Module):
        def forward(self, **kw):
            raise RuntimeError("weird")

    j = DSPyLeakJudge(lm=DummyLM([]))
    j.program = Boom()
    assert j.blocks(REPLY, "x") is True


def test_logs_have_no_text(caplog):
    caplog.set_level(logging.DEBUG)
    w = writer([{"reasoning": "SECRET-THOUGHT", "reply": REPLY}])
    w.write("A: ความลับของบทสนทนา", NLPResult())
    judge = DSPyLeakJudge(lm=DummyLM([{"reasoning": "JUDGE-THOUGHT", "leaks": False}]))
    assert judge.blocks(REPLY, "A: ความลับของบทสนทนา") is False
    assert "SECRET-THOUGHT" not in caplog.text and REPLY not in caplog.text
    assert "JUDGE-THOUGHT" not in caplog.text
    assert "ความลับ" not in caplog.text


def test_build_lm_uses_settings(monkeypatch):
    monkeypatch.setattr(dspy_programs, "LLMClient", lambda: SimpleNamespace(model="served-model"))
    lm = build_lm()
    assert lm.model == "openai/served-model"
    assert lm.kwargs["api_base"] == dspy_programs.get_settings().vllm_base_url
    assert lm.kwargs["max_tokens"] == dspy_programs.DSPY_MAX_TOKENS > 80  # room for reasoning


# ---- optimizer helpers (no server) ---------------------------------------------------------


def test_metric():
    assert dspy_optimize.reply_metric(None, SimpleNamespace(reply=REPLY)) == 1.0
    for bad in ["", "Please call back", "โทร 0812345678 ค่ะ", "ก" * 201, "ขอสาย John Smith ค่ะ"]:
        assert dspy_optimize.reply_metric(None, SimpleNamespace(reply=bad)) == 0.0


def test_default_trainset_is_redacted_and_sized():
    from tts_voice_agent.redact import redact_context

    ex = dspy_optimize.load_trainset(None)
    assert 12 <= len(ex) <= 16
    for e in ex:
        assert e.call_context == redact_context(e.call_context, [])
        assert set(e.inputs().keys()) == {"call_context", "verdict"}


def test_optimize_with_dummy_lm(tmp_path):
    ex = dspy_optimize.load_trainset(None)
    lm = DummyLM([{"reasoning": "r", "reply": REPLY}] * 200)
    out = tmp_path / "c" / "reply.dspy.json"
    before, after = dspy_optimize.optimize(ex, out, lm=lm)
    assert (before, after) == (100.0, 100.0)
    assert out.exists()


def test_optimizer_tunes_at_the_writers_temperature(monkeypatch, tmp_path):
    seen = {}

    def fake_build_lm(temperature=None, **kwargs):
        seen.setdefault("writer", temperature)
        return object()

    monkeypatch.setattr(dspy_programs, "build_lm", fake_build_lm)
    _ = DSPyReplyWriter(compiled_path=tmp_path / "missing.json").lm  # builds the LM

    def fake_optimize(trainset, out, lm=None, temperature=None):
        seen["optimizer"] = temperature
        return 0.0, 0.0

    monkeypatch.setattr(dspy_optimize, "optimize", fake_optimize)
    monkeypatch.setattr(dspy_optimize, "load_trainset", lambda path: [])
    dspy_optimize.main(["--out", str(tmp_path / "out.json")])
    assert seen["optimizer"] == seen["writer"] == dspy_programs.DSPY_REPLY_TEMPERATURE

    dspy_optimize.main(["--out", str(tmp_path / "out.json"), "--temperature", "0.3"])
    assert seen["optimizer"] == 0.3
