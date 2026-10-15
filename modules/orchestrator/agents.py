"""Tool-calling analyst agent on the vLLM server (OpenAI-compatible).

The only LLM endpoint used is `settings.vllm_base_url`. Call text never goes anywhere else.

The analyst agent writes advisory text for the UI only. Its output never feeds
`tts_voice_agent.decision.decide`, never becomes speech, and never changes routing.
"""

from __future__ import annotations

import openai

from callas_settings import get_settings

from .tools import TOOLS

ANALYST_PROMPT = (
    "You are a scam-call analyst for Thai users. You get a conversation or message. "
    "Call the tools extract_entities, classify_scam and summarize_text on it, then answer. "
    "Answer in 2-3 sentences of plain Thai: say why the conversation looks risky or safe, and "
    "name which tool results support that (for example the scam type and confidence, or the kind "
    "of entities found). Never repeat phone numbers, account numbers, OTPs or personal names; "
    "refer to them by kind only (for example 'a bank account number')."
)

TOOL_HINT = (
    "The vLLM server rejected tool calling. Start vLLM with "
    "--enable-auto-tool-choice --tool-call-parser <parser for your model>."
)


class ToolCallingUnavailable(RuntimeError):
    """The vLLM server does not accept tool calls (missing --enable-auto-tool-choice)."""


def build_chat_model():
    """ChatOpenAI pointed at the configured vLLM server."""
    from langchain_openai import ChatOpenAI

    s = get_settings()
    model = s.vllm_model
    if not model:
        from llm_serving.client import LLMClient

        models = LLMClient().list_models()
        if not models:
            raise RuntimeError(f"LLM server at {s.vllm_base_url} lists no models")
        model = models[0]
    return ChatOpenAI(
        base_url=s.vllm_base_url,
        model=model,
        api_key=s.vllm_api_key.get_secret_value() or "EMPTY",
        timeout=s.vllm_timeout_s,
        temperature=0,
        extra_body=s.llm_extra_body() or None,
    )


def build_analyst_agent(model=None):
    """ReAct-style agent with the three nlp-core tools. `model` defaults to build_chat_model()."""
    from langchain.agents import create_agent

    if model is None:
        model = build_chat_model()
    return create_agent(model, TOOLS, system_prompt=ANALYST_PROMPT)


def _content_text(content) -> str:
    """Text of a message whose content is a string or a list of text blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, str | dict)
    )


def explain(text: str, agent=None) -> str:
    """Plain-Thai advice on why `text` looks risky or safe. Advisory only: never feeds decide()."""
    agent = agent or build_analyst_agent()
    try:
        out = agent.invoke({"messages": [{"role": "user", "content": text}]})
    except openai.BadRequestError as exc:
        if "tool" in str(exc).lower():
            raise ToolCallingUnavailable(TOOL_HINT) from exc
        raise
    return _content_text(out["messages"][-1].content).strip()
