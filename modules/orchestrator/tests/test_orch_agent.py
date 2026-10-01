import json

import httpx
import openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator import ToolCallingUnavailable, build_analyst_agent, explain
from orchestrator.tools import classify_scam, extract_entities, summarize_text


class ScriptedChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*msgs):
    return ScriptedChat(messages=iter(msgs))


def test_tools_return_json_dicts(fakes):
    for t, arg in ((extract_entities, "PHONE"), (classify_scam, "SCAM"), (summarize_text, "x")):
        json.dumps(t.invoke({"text": arg}))
    assert classify_scam.invoke({"text": "SCAM"}) == {
        "label_scam": True,
        "label_type": "fake_authority",
        "confidence": 0.95,
    }
    assert extract_entities.invoke({"text": "PHONE"})["entities"][0]["label"] == "PHONE"


def test_agent_calls_tool_then_answers(fakes):
    call = AIMessage(
        content="",
        tool_calls=[{"name": "classify_scam", "args": {"text": "SCAM now"}, "id": "t1"}],
    )
    final = AIMessage(content="บทสนทนานี้เสี่ยงสูง เพราะ classify_scam พบการแอบอ้างเจ้าหน้าที่")
    agent = build_analyst_agent(scripted(call, final))
    assert explain("SCAM now", agent=agent) == final.content
    assert fakes["classifier"].calls == ["SCAM now"]


def test_agent_without_tool_call(fakes):
    agent = build_analyst_agent(scripted(AIMessage(content="ปลอดภัย")))
    assert explain("hi", agent=agent) == "ปลอดภัย"
    assert fakes["classifier"].calls == []


def test_server_rejecting_tools_raises_clear_error():
    class Rejecting:
        def invoke(self, _):
            resp = httpx.Response(400, request=httpx.Request("POST", "http://x/v1/chat"))
            raise openai.BadRequestError(
                '"auto" tool choice requires --enable-auto-tool-choice', response=resp, body=None
            )

    with pytest.raises(ToolCallingUnavailable, match="enable-auto-tool-choice"):
        explain("x", agent=Rejecting())
