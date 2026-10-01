import pytest

from nlp_core.types import SCAM_TYPES, NLPResult
from tts_voice_agent.constants import ESCALATE_THRESHOLD
from tts_voice_agent.decision import decide
from tts_voice_agent.memory import CallState


def _scam(kind, conf):
    return NLPResult(label_scam=True, label_type=kind, confidence=conf)


@pytest.mark.parametrize("kind", SCAM_TYPES)
def test_each_scam_type_high_confidence_escalates(kind):
    state = CallState()
    d = decide(_scam(kind, 0.95), state)
    assert not d.auto_reply
    assert d.reason == "high-confidence scam, escalate to user"
    assert state.escalated


@pytest.mark.parametrize("kind", SCAM_TYPES)
def test_each_scam_type_low_confidence_escalates_for_review(kind):
    state = CallState()
    d = decide(_scam(kind, 0.3), state)
    assert not d.auto_reply
    assert d.reason == "low-confidence scam, escalate for review"
    assert state.escalated


def test_benign_gets_auto_reply():
    state = CallState()
    d = decide(NLPResult(), state)
    assert d.auto_reply and d.reason == "benign" and d.reply_text is None
    assert not state.escalated


def test_confidence_exactly_at_threshold_is_high():
    d = decide(_scam("lottery", ESCALATE_THRESHOLD), CallState())
    assert d.reason.startswith("high-confidence")


def test_confidence_just_below_threshold_is_low():
    d = decide(_scam("lottery", ESCALATE_THRESHOLD - 1e-9), CallState())
    assert d.reason.startswith("low-confidence")


def test_unknown_scam_type_zero_confidence_escalates():
    state = CallState()
    d = decide(NLPResult(label_scam=True, label_type=None, confidence=0.0), state)
    assert not d.auto_reply and state.escalated


def test_already_escalated_stays_no_reply_on_benign():
    state = CallState(escalated=True)
    d = decide(NLPResult(), state)
    assert not d.auto_reply and d.reason == "call already escalated"
    assert state.escalated


def test_deterministic():
    r = _scam("romance", 0.8)
    assert decide(r, CallState()) == decide(r, CallState())
