from asr_diarization.types import Turn
from llm_serving.fake import FakeLLM
from orchestrator import CallSessions, Runtime


def test_remember_node_stores_the_clean_turn_for_every_tool(fakes):
    s = CallSessions(Runtime(llm=FakeLLM()))
    cid = s.start()
    out = s.on_turn(cid, Turn("A", 0, 1, "สวัสดี   ครับ"))
    assert out["turn"].text == "สวัสดี ครับ" and out["turn"].raw == "สวัสดี   ครับ"
    assert s.runtime.states[cid].turns[0].text == "สวัสดี ครับ"
    assert out["context"].startswith("A: สวัสดี ครับ")
