"""Thinking models must not eat the answer: every chat call disables thinking by default."""

from unittest.mock import MagicMock

from callas_settings import get_settings
from llm_serving.client import LLMClient


def _client_with_fake_openai():
    c = LLMClient(base_url="http://x/v1", model="m", api_key="k")
    fake = MagicMock()
    fake.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="normal"))
    ]
    c._client = fake
    return c, fake


def test_thinking_disabled_by_default():
    c, fake = _client_with_fake_openai()
    c.complete("classify", "สวัสดี")
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_thinking_can_be_turned_on(monkeypatch):
    monkeypatch.setenv("VLLM_THINKING", "true")
    get_settings.cache_clear()
    try:
        c, fake = _client_with_fake_openai()
        c.complete("classify", "สวัสดี")
        assert fake.chat.completions.create.call_args.kwargs["extra_body"] is None
    finally:
        get_settings.cache_clear()


def test_dspy_lm_disables_thinking(monkeypatch):
    from llm_serving import dspy_programs

    monkeypatch.setattr(dspy_programs.LLMClient, "model", property(lambda self: "m"))
    lm = dspy_programs.build_lm()
    assert lm.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
