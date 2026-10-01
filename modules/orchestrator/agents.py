"""Tool-calling sub-agents on the vLLM server (OpenAI-compatible).

The only LLM endpoint used is `settings.vllm_base_url`. Call text never goes anywhere else.

The analyst agent writes advisory text for the UI only. Its output never feeds
`tts_voice_agent.decision.decide`, never becomes speech, and never changes routing.
"""

from __future__ import annotations

import warnings

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
    )


def build_analyst_agent(model=None):
    """ReAct-style agent with the three nlp-core tools. `model` defaults to build_chat_model()."""
    from langgraph.prebuilt import create_react_agent

    if model is None:
        model = build_chat_model()
    # create_react_agent is deprecated, but `langchain` (create_agent) is not installed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from langchain_core._api.deprecation import LangChainDeprecationWarning

            warnings.simplefilter("ignore", LangChainDeprecationWarning)
        except ImportError:
            pass
        return create_react_agent(model, TOOLS, prompt=ANALYST_PROMPT)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, str | dict)
    )


def explain(text: str, agent=None) -> str:
    """Plain-Thai advice on why `text` looks risky or safe. Advisory only: never feeds decide()."""
    agent = agent or build_analyst_agent()
    try:
        out = agent.invoke({"messages": [("user", text)]})
    except openai.BadRequestError as exc:
        if "tool" in str(exc).lower():
            raise ToolCallingUnavailable(TOOL_HINT) from exc
        raise
    return _text(out["messages"][-1].content).strip()
