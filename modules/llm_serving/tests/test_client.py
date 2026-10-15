from types import SimpleNamespace

import httpx
import openai
import pytest

from llm_serving import client as client_mod
from llm_serving.base import JOB_SETTINGS, JOBS, JobSettings
from llm_serving.client import LLMClient, LLMUnavailable, parse_label


class FakeOpenAI:
    """Stands in for openai.OpenAI; records kwargs and returns canned output."""

    instances: list["FakeOpenAI"] = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.requests: list[dict] = []
        self.output = "normal"
        self.error: Exception | None = None
        self.model_ids = ["served-model"]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=self._list)
        FakeOpenAI.instances.append(self)

    def _create(self, **kwargs):
        if self.error:
            raise self.error
        self.requests.append(kwargs)
        msg = SimpleNamespace(content=self.output)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.model_ids])


@pytest.fixture
def fake(monkeypatch):
    FakeOpenAI.instances.clear()
    monkeypatch.setattr(client_mod, "OpenAI", FakeOpenAI)
    return FakeOpenAI


def make(model="m1"):
    c = LLMClient(base_url="http://x/v1", model=model, api_key="secret-key", timeout=5)
    return c, FakeOpenAI.instances[-1]


def conn_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x/v1"))


def test_job_settings_keep_their_values():
    assert JOB_SETTINGS == {
        "summary": JobSettings(max_tokens=200, temperature=0.0),
        "classify": JobSettings(max_tokens=20, temperature=0.0),
        "reply": JobSettings(max_tokens=80, temperature=0.3),
    }
    assert JOBS == {"summary": 200, "classify": 20, "reply": 80}
    assert client_mod.TEMPERATURE == {"classify": 0.0, "summary": 0.0, "reply": 0.3}


@pytest.mark.parametrize("job", sorted(JOBS))
def test_complete_sends_prompt_and_max_tokens(fake, job):
    c, api = make()
    api.output = "  hello \n"
    assert c.complete(job, "บริบท {ทดสอบ}") == "hello"
    req = api.requests[0]
    assert req["max_tokens"] == JOBS[job]
    assert req["model"] == "m1"
    assert req["temperature"] == (0.3 if job == "reply" else 0.0)
    (msg,) = req["messages"]
    assert msg["role"] == "user"
    assert "บริบท {ทดสอบ}" in msg["content"]
    assert "{context}" not in msg["content"]
    assert not any(ln.startswith("#") for ln in msg["content"].splitlines())


def test_constructor_passes_settings(fake):
    _, api = make()
    assert api.init_kwargs["base_url"] == "http://x/v1"
    assert api.init_kwargs["api_key"] == "secret-key"
    assert api.init_kwargs["timeout"] == 5


def test_unknown_job_raises(fake):
    c, api = make()
    with pytest.raises(ValueError):
        c.complete("translate", "x")
    assert api.requests == []


def test_connection_error_becomes_llm_unavailable(fake):
    c, api = make()
    api.error = conn_error()
    with pytest.raises(LLMUnavailable):
        c.complete("classify", "x")


def test_timeout_becomes_llm_unavailable(fake):
    c, api = make()
    api.error = openai.APITimeoutError(request=httpx.Request("POST", "http://x/v1"))
    with pytest.raises(LLMUnavailable):
        c.complete("summary", "x")


def test_empty_model_picks_first_listed(fake):
    c, api = make(model="")
    api.model_ids = ["first", "second"]
    c.complete("classify", "x")
    c.complete("classify", "y")
    assert api.requests[0]["model"] == "first"
    assert api.requests[1]["model"] == "first"


def test_logging_has_lengths_not_text_or_key(fake, caplog):
    c, api = make()
    with caplog.at_level("INFO", logger="llm_serving"):
        c.complete("classify", "เนื้อหาลับของสาย")
    log = caplog.text
    assert "model=m1" in log and "job=classify" in log and "prompt=v1" in log
    assert "latency_s=" in log and "context_chars=16" in log
    assert "เนื้อหาลับของสาย" not in log
    assert "secret-key" not in log


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("normal", "normal"),
        ("fake_authority", "fake_authority"),
        ("romance", "romance"),
        ("investment", "investment"),
        ("phishing_otp", "phishing_otp"),
        ("lottery", "lottery"),
        ("PHISHING_OTP", "phishing_otp"),
        ("  Lottery \n", "lottery"),
        ('"romance"', "romance"),
        ("'investment'.", "investment"),
        ("`fake_authority`", "fake_authority"),
        ("ป้ายกำกับ: lottery ครับ", "lottery"),
        ("คำตอบคือ normal", "normal"),
        ("normal or lottery", "unknown"),
        ("romance, investment", "unknown"),
        ("", "unknown"),
        ("   ", "unknown"),
        ("I cannot tell", "unknown"),
        ("abnormal", "unknown"),
        ("phishing", "unknown"),
        ("scam", "unknown"),
        ("ไม่แน่ใจ", "unknown"),
    ],
)
def test_parse_label(raw, expected):
    assert parse_label(raw) == expected
