from asr_diarization.types import Turn
from nlp_core.types import Entity, NLPResult
from tts_voice_agent.agent import on_turn
from tts_voice_agent.constants import CONTEXT_TOKEN_BUDGET
from tts_voice_agent.memory import CallState, default_count_tokens


def _turn(i, text=None):
    return Turn(
        "A" if i % 2 == 0 else "B", float(i), float(i + 1), text or f"ข้อความที่ยาวมากพอสมควร {i}"
    )


class FakeLLM:
    def complete(self, job, context):
        return "ขอบคุณค่ะ ขณะนี้ไม่สะดวกคุยค่ะ"


def test_context_format_and_summary():
    s = CallState(older_summary="สรุป")
    s.add(_turn(0, "สวัสดี"))
    assert s.context() == "สรุป\nA: สวัสดี"


def test_300_turn_call_stays_within_budget_and_keeps_entities():
    s = CallState()
    for i in range(300):
        ents = [Entity(f"ชื่อ{i}", "PERSON", 0, 3)]
        s.add(_turn(i), ents)
        ctx = s.context()
        assert default_count_tokens(ctx) <= CONTEXT_TOKEN_BUDGET
    assert len(s.entities_seen) == 300
    assert ctx.endswith("A: " + s.turns[-1].text.replace("A: ", "")) or "299" in ctx
    assert s.older_summary


def test_entities_dedup():
    s = CallState()
    e = Entity("สมชาย", "PERSON", 0, 5)
    s.add(_turn(0), [e])
    s.add(_turn(1), [Entity("สมชาย", "PERSON", 9, 14), Entity("สมชาย", "ORG", 0, 5)])
    assert [(x.text, x.label) for x in s.entities_seen] == [("สมชาย", "PERSON"), ("สมชาย", "ORG")]


def test_injected_summarizer_and_counter_used():
    s = CallState()
    for i in range(10):
        s.add(_turn(i, f"t{i}"))
    ctx = s.context(count_tokens=lambda t: len(t.split()), summarize=lambda t: "SUM", budget=14)
    assert ctx.startswith("SUM\n")
    assert len(s.turns) == 6 and ctx.count("\n") == 6


def test_summary_truncated_when_still_over():
    s = CallState()
    for i in range(20):
        s.add(_turn(i, "x"))
    big = lambda t: "y" * 500  # noqa: E731
    ctx = s.context(count_tokens=len, summarize=big, budget=60)
    assert len(ctx) <= 60 and ctx.startswith("y") and ctx.endswith("B: x")


def test_recent_turns_alone_over_budget_get_folded():
    s = CallState()
    for i in range(3):
        s.add(_turn(i, "z" * 40))
    ctx = s.context(count_tokens=len, budget=50)
    assert len(ctx) <= 50


def test_end_clears_everything():
    s = CallState(escalated=True)
    s.add(_turn(0), [Entity("สมชาย", "PERSON", 0, 5)])
    s.older_summary = "x"
    s.end()
    assert (s.turns, s.older_summary, s.entities_seen, s.escalated) == ([], "", [], False)


def test_escalation_is_sticky_across_later_benign_turns():
    results = iter(
        [
            NLPResult(),
            NLPResult(label_scam=True, label_type="lottery", confidence=0.9),
            NLPResult(),
            NLPResult(),
        ]
    )
    s = CallState()
    outcomes = [on_turn(s, _turn(i), lambda ctx: next(results), FakeLLM()) for i in range(4)]
    autos = [o.decision.auto_reply for o in outcomes]
    assert autos == [True, False, False, False]
    assert outcomes[0].decision.reply_text and outcomes[0].decision.template_used is False
    assert outcomes[2].decision.reply_text is None


def test_on_turn_collects_entities_and_passes_context():
    seen = []

    def analyze(ctx):
        seen.append(ctx)
        return NLPResult(entities=[Entity("สมชาย", "PERSON", 0, 5)])

    s = CallState()
    out = on_turn(s, _turn(0, "สวัสดี"), analyze, FakeLLM())
    assert seen == ["A: สวัสดี"] and len(s.entities_seen) == 1
    assert out.decision.auto_reply
