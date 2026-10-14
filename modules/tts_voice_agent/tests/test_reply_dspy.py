import logging
from pathlib import Path

import dspy
import pytest
from dspy.utils import DummyLM

from asr_diarization.types import Turn
from callas_settings import Settings, get_settings
from llm_serving.dspy_programs import DSPyReplyWriter
from llm_serving.fake import FakeLLM
from nlp_core.types import Entity, NLPResult
from tts_voice_agent.leak import leaks_sensitive
from tts_voice_agent.memory import CallState
from tts_voice_agent.reply import safe_reply
from tts_voice_agent.templates import pick_template

from .test_leak_and_reply import BAD, GOOD

_NOFILE = Path("does-not-exist.dspy.json")
GOOD_REPLY = "ขอบคุณที่โทรมาค่ะ ขณะนี้ไม่สะดวกคุยค่ะ"
REASONING = "ZEBRA-REASONING-SECRET คิดก่อนตอบ"
PHONE = "0812345678"
NAME = "สมชาย"
ENTS = [Entity(NAME, "PERSON", 0, 5), Entity(PHONE, "PHONE", 0, 10)]


class Writer:
    def __init__(self, text=GOOD_REPLY, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def write(self, context, result):
        self.calls.append(context)
        if self.exc:
            raise self.exc
        return self.text


class Judge:
    def __init__(self, answer=False, exc=None):
        self.answer, self.exc, self.calls = answer, exc, []

    def blocks(self, reply, context):
        self.calls.append((reply, context))
        if self.exc:
            raise self.exc
        return self.answer


def state_with_call():
    st = CallState()
    st.add(
        Turn("A", 0, 1, f"ผม{NAME} โทรจาก {PHONE} บัญชี 123-4-56789-0 เข้า https://x.example"), ENTS
    )
    return st


# ---- the writer only sees redacted context -------------------------------------------------


def test_writer_receives_redacted_context():
    w = Writer()
    safe_reply(NLPResult(), state_with_call(), FakeLLM(), writer=w)
    (ctx,) = w.calls
    ctx = ctx.split("\n", 1)[1]  # first line is the date and time facts
    assert NAME not in ctx and PHONE not in ctx and "x.example" not in ctx
    assert not any(c.isdigit() for c in ctx)
    assert "[PERSON]" in ctx and "[LINK]" in ctx


def test_prompt_path_also_gets_redacted_context():
    llm = FakeLLM()
    safe_reply(NLPResult(), state_with_call(), llm)
    (ctx,) = llm.calls_for("reply")
    ctx = ctx.split("\n", 1)[1]  # first line is the date and time facts
    assert NAME not in ctx and PHONE not in ctx and not any(c.isdigit() for c in ctx)


def test_judge_sees_redacted_context():
    j = Judge()
    safe_reply(NLPResult(), state_with_call(), FakeLLM(), writer=Writer(), judge=j)
    assert NAME not in j.calls[0][1] and PHONE not in j.calls[0][1]


# ---- OR rule: the judge can only add blocks ------------------------------------------------


@pytest.mark.parametrize("text", BAD)
def test_judge_no_leak_never_unblocks_rule_block(text):
    j = Judge(answer=False)
    out, used = safe_reply(NLPResult(), CallState(), FakeLLM(), writer=Writer(text), judge=j)
    assert used and out == pick_template(NLPResult())
    assert j.calls == []  # the judge does not even run when the rules block


@pytest.mark.parametrize("text", GOOD)
def test_clean_reply_and_judge_ok_passes(text):
    out, used = safe_reply(
        NLPResult(), CallState(), FakeLLM(), writer=Writer(text), judge=Judge(False)
    )
    assert (out, used) == (text, False)


def test_judge_true_gives_template(caplog):
    caplog.set_level(logging.DEBUG)
    out, used = safe_reply(NLPResult(), CallState(), FakeLLM(), writer=Writer(), judge=Judge(True))
    assert used and out != GOOD_REPLY
    assert "judge_block" in caplog.text


def test_judge_raising_gives_template(caplog):
    caplog.set_level(logging.DEBUG)
    j = Judge(exc=TimeoutError("slow"))
    out, used = safe_reply(NLPResult(), CallState(), FakeLLM(), writer=Writer(), judge=j)
    assert used and out != GOOD_REPLY
    assert "judge_error" in caplog.text


@pytest.mark.parametrize("garbage", [None, "false", "no", 0, "", "maybe", [False]])
def test_judge_garbage_gives_template(garbage):
    out, used = safe_reply(
        NLPResult(), CallState(), FakeLLM(), writer=Writer(), judge=Judge(answer=garbage)
    )
    assert used and out != GOOD_REPLY


def test_writer_failure_gives_template():
    out, used = safe_reply(
        NLPResult(), CallState(), FakeLLM(), writer=Writer(exc=RuntimeError("x"))
    )
    assert used


def test_defaults_reproduce_old_behaviour():
    llm = FakeLLM({"reply": GOOD_REPLY})
    assert safe_reply(NLPResult(), CallState(), llm) == (GOOD_REPLY, False)
    llm = FakeLLM({"reply": "โทร 0812345678"})
    out, used = safe_reply(NLPResult(), CallState(), llm)
    assert used and not leaks_sensitive(out, [])


# ---- reasoning is discarded ----------------------------------------------------------------


def dummy_writer():
    lm = DummyLM([{"reasoning": REASONING, "reply": f" {GOOD_REPLY} "}] * 5)
    return DSPyReplyWriter(lm=lm, compiled_path=_NOFILE)


def test_reasoning_never_in_output_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    w = dummy_writer()
    out, used = safe_reply(NLPResult(), state_with_call(), FakeLLM(), writer=w)
    assert (out, used) == (GOOD_REPLY, False)
    assert REASONING not in out and REASONING not in caplog.text
    assert GOOD_REPLY not in caplog.text and NAME not in caplog.text


def test_dummy_lm_really_produces_reasoning():
    # guards the test above: the program does return a reasoning field, the writer drops it
    w = dummy_writer()
    with dspy.context(lm=w.lm):
        pred = w.program(call_context="x", verdict="benign")
    assert pred.reasoning == REASONING
    assert dummy_writer().write("x", NLPResult()) == GOOD_REPLY


def test_default_settings():
    s = Settings(_env_file=None)
    assert (s.reply_engine, s.leak_judge) == ("prompt", "off")


@pytest.mark.parametrize("var,value", [("REPLY_ENGINE", "gpt"), ("LEAK_JUDGE", "maybe")])
def test_invalid_switch_errors(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        Settings(_env_file=None)
    get_settings.cache_clear()


# ---- the turn graph: reasoning never reaches the state or the events -------------------------


@pytest.fixture
def fake_nlp(monkeypatch):
    from nlp_core import registry

    class NER:
        def extract(self, text):
            return [Entity(PHONE, "PHONE", 0, 10)] if PHONE in text else []

    class Clf:
        def predict(self, text):
            return False, None, 1.0

    class Summ:
        def summarize(self, text):
            return "summary"

    saved = ({k: dict(v) for k, v in registry._FACTORIES.items()}, dict(registry._INSTANCES))
    for kind, inst in {"ner": NER(), "classifier": Clf(), "summarizer": Summ()}.items():
        registry.register(kind, "fake", lambda inst=inst: inst)
        monkeypatch.setenv(registry.ENV_VARS[kind], "fake")
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved[0])
    registry._INSTANCES.clear()
    registry._INSTANCES.update(saved[1])


def test_reasoning_not_in_graph_state_or_events(fake_nlp, caplog):
    from orchestrator import CallSessions, Runtime
    from orchestrator.sessions import to_jsonable

    caplog.set_level(logging.DEBUG)
    s = CallSessions(Runtime(llm=FakeLLM(), writer=dummy_writer(), judge=Judge(False)))
    cid = s.start()
    out = s.on_turn(cid, Turn("A", 0, 1, f"สวัสดี {PHONE}"))
    assert out["reply_text"] == GOOD_REPLY and out["template_used"] is False
    assert REASONING not in repr(to_jsonable(out)) and REASONING not in caplog.text
    events = [to_jsonable(e) for e in s.stream_turn(cid, Turn("A", 1, 2, "สวัสดีอีกที"))]
    assert REASONING not in repr(events)


def test_graph_judge_block_uses_template(fake_nlp):
    from orchestrator import CallSessions, Runtime

    s = CallSessions(Runtime(llm=FakeLLM(), writer=Writer(), judge=Judge(True)))
    out = s.on_turn(s.start(), Turn("A", 0, 1, "สวัสดี"))
    assert out["template_used"] is True and out["reply_text"] != GOOD_REPLY


def test_runtime_builds_writer_and_judge_from_settings(fake_nlp, monkeypatch):
    from orchestrator import Runtime

    get_settings.cache_clear()
    try:
        monkeypatch.setenv("REPLY_ENGINE", "prompt")
        monkeypatch.setenv("LEAK_JUDGE", "off")
        get_settings.cache_clear()
        rt = Runtime(llm=FakeLLM())
        assert rt.writer is None and rt.judge is None
        monkeypatch.setenv("REPLY_ENGINE", "dspy")
        monkeypatch.setenv("LEAK_JUDGE", "on")
        get_settings.cache_clear()
        rt = Runtime(llm=FakeLLM())  # lazy: no network until the first call
        assert type(rt.writer).__name__ == "DSPyReplyWriter"
        assert type(rt.judge).__name__ == "DSPyLeakJudge"
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
