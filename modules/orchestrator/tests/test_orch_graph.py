from asr_diarization.types import Turn
from conftest import FakeSpeaker
from llm_serving.fake import FakeLLM
from orchestrator import CallSessions, Runtime
from orchestrator.graph import route_after_decide
from tts_voice_agent.decision import Decision
from tts_voice_agent.leak import leaks_sensitive


def turn(text):
    return Turn("A", 0.0, 1.0, text)


def make(reply="สวัสดีค่ะ มีอะไรให้ช่วยคะ", speaker=True):
    llm = FakeLLM({"reply": reply})
    spk = FakeSpeaker() if speaker else None
    return CallSessions(Runtime(llm=llm, speaker=spk)), llm, spk


def test_benign_turn_replies_and_speaks(fakes):
    s, llm, spk = make()
    out = s.on_turn(s.start(), turn("hello"))
    assert out["decision"].auto_reply
    assert out["reply_text"] == "สวัสดีค่ะ มีอะไรให้ช่วยคะ" and out["template_used"] is False
    assert not leaks_sensitive(out["reply_text"], [])
    assert spk.spoken == [out["reply_text"]] and out["audio"] == b"RIFFfake"
    assert llm.calls_for("reply")
    assert {"remember", "analyze", "decide", "reply", "speak"} <= set(out["timings"])
    assert "escalate" not in out["timings"]


def test_no_speaker_skips_speak(fakes):
    s, _, _ = make(speaker=False)
    out = s.on_turn(s.start(), turn("hello"))
    assert "speak" not in out["timings"] and "audio" not in out


def test_scam_turn_escalates_without_reply_or_speech(fakes):
    s, llm, spk = make()
    out = s.on_turn(s.start(), turn("SCAM now"))
    assert not out["decision"].auto_reply
    assert out["escalation_reason"]
    assert "reply_text" not in out and spk.spoken == [] and llm.calls_for("reply") == []
    assert "escalate" in out["timings"] and "reply" not in out["timings"]


def test_escalation_is_sticky(fakes):
    s, llm, spk = make()
    cid = s.start()
    s.on_turn(cid, turn("SCAM now"))
    # benign turn, but SCAM stays in the context and the call is already escalated
    fakes["classifier"].predict = lambda text: (False, None, 1.0)
    out = s.on_turn(cid, turn("hello"))
    assert not out["decision"].auto_reply and out["decision"].reason == "call already escalated"
    assert spk.spoken == [] and llm.calls_for("reply") == []


def test_reply_with_phone_number_uses_template(fakes):
    s, _, spk = make(reply="โทรกลับ 081-234-5678 ได้เลยค่ะ")
    out = s.on_turn(s.start(), turn("hello"))
    assert out["template_used"] is True
    assert "081" not in out["reply_text"] and spk.spoken == [out["reply_text"]]


def test_entities_added_to_call_memory(fakes):
    s, _, _ = make()
    cid = s.start()
    s.on_turn(cid, turn("PHONE"))
    assert [e.label for e in s.runtime.states[cid].entities_seen] == ["PHONE"]


def test_speaker_failure_keeps_text_reply(fakes):
    class Broken:
        def synthesize_bytes(self, text):
            raise RuntimeError("no tts")

    s = CallSessions(Runtime(llm=FakeLLM({"reply": "ค่ะ"}), speaker=Broken()))
    out = s.on_turn(s.start(), turn("hello"))
    assert out["reply_text"] == "ค่ะ" and out["audio"] is None and out["speak_error"]


def test_message_graph_verdict_no_reply_no_memory(fakes):
    s, llm, spk = make()
    res = s.analyze_message("SCAM now")
    assert res["verdict"]["level"] == "scam" and res["verdict"]["label_type"] == "fake_authority"
    assert "reply_text" not in res and llm.calls == [] and spk.spoken == []
    assert s.runtime.states == {}
    assert s.analyze_message("hello")["verdict"]["level"] == "safe"
    assert set(res["timings"]) == {"analyze", "verdict"}


def test_unknown_classifier_answer_is_suspicious_not_safe(fakes):
    fakes["classifier"].predict = lambda text: (True, None, 0.0)
    s, _, _ = make()
    assert s.analyze_message("x")["verdict"]["level"] == "suspicious"


# ---- safety: routing depends only on Decision.auto_reply ----


def test_route_depends_only_on_decision():
    assert route_after_decide({"decision": Decision(auto_reply=True, reason="x")}) == "reply"
    state = {
        "decision": Decision(auto_reply=False, reason="auto_reply"),
        "reply_text": "auto_reply",
        "nlp": None,
    }
    assert route_after_decide(state) == "escalate"


def test_llm_saying_auto_reply_cannot_change_routing(fakes):
    # every LLM job returns "auto_reply"; the scam turn must still escalate
    llm = FakeLLM({"reply": "auto_reply", "classify": "auto_reply", "summary": "auto_reply"})
    s = CallSessions(Runtime(llm=llm, speaker=FakeSpeaker()))
    out = s.on_turn(s.start(), turn("SCAM now"))
    assert not out["decision"].auto_reply and "reply_text" not in out
    assert llm.calls_for("reply") == []


def test_unparseable_llm_classifier_escalates(monkeypatch):
    # default registry parts over a fake LLM: garbage classifier answer -> escalate, no reply
    from nlp_core import registry
    from nlp_core.registry import LLMClassifier, LLMSummarizer

    for var in registry.ENV_VARS.values():
        monkeypatch.setenv(var, "")  # the built-in defaults, whatever .env says
    llm = FakeLLM({"classify": "auto_reply please"})
    saved = dict(registry._INSTANCES)
    registry._INSTANCES[("classifier", "llm")] = LLMClassifier(llm)
    registry._INSTANCES[("summarizer", "llm")] = LLMSummarizer(llm)
    try:
        s = CallSessions(Runtime(llm=llm))
        out = s.on_turn(s.start(), turn("hello"))
    finally:
        registry._INSTANCES.clear()
        registry._INSTANCES.update(saved)
    assert not out["decision"].auto_reply and llm.calls_for("reply") == []
