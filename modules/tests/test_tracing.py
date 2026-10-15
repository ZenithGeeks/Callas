"""Tests for callas_tracing. No network and no Phoenix server: the OTel setup is replaced."""

import subprocess
import sys

import callas_tracing
import pytest

from callas_settings import get_settings

pytest.importorskip("phoenix.otel", reason="needs the trace extra")
from openinference.instrumentation.dspy import DSPyInstrumentor  # noqa: E402
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from openinference.instrumentation.litellm import LiteLLMInstrumentor  # noqa: E402


@pytest.fixture
def traced(monkeypatch):
    """Turn tracing on with fake OTel setup; returns the calls that were made."""
    calls = {"register": [], "config": []}

    def fake_register(**kwargs):
        calls["register"].append(kwargs)
        return "provider"

    def fake_instrument(self, **kwargs):
        calls["config"].append(kwargs["config"])

    monkeypatch.setattr("phoenix.otel.register", fake_register)
    monkeypatch.setattr(LangChainInstrumentor, "instrument", fake_instrument)
    monkeypatch.setattr(DSPyInstrumentor, "instrument", fake_instrument)
    monkeypatch.setattr(LiteLLMInstrumentor, "instrument", fake_instrument)
    monkeypatch.setattr(callas_tracing, "_started", False)
    monkeypatch.setenv("TRACING", "phoenix")
    get_settings.cache_clear()
    yield calls
    get_settings.cache_clear()


def test_off_imports_no_tracing_package():
    code = (
        "import sys; sys.path.insert(0, 'modules');"
        "from callas_tracing import setup_tracing;"
        "setup_tracing('callas-web', call_text=False);"
        "print([m for m in sys.modules if m.split('.')[0] in ('phoenix', 'openinference')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"TRACING": "off", "SYSTEMROOT": "C:\\Windows"},
    )
    assert out.stdout.strip() == "[]", out.stderr


def test_non_loopback_endpoint_is_refused(traced, monkeypatch):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "https://phoenix.example.com:6006")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="PHOENIX_COLLECTOR_ENDPOINT"):
        callas_tracing.setup_tracing("callas-web", call_text=False)
    assert traced["register"] == []


@pytest.mark.parametrize(
    "endpoint", ["http://localhost:6006", "http://127.0.0.1:6006", "http://[::1]:6006"]
)
def test_loopback_endpoint_is_accepted(traced, monkeypatch, endpoint):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)
    get_settings.cache_clear()
    callas_tracing.setup_tracing("callas-web", call_text=False)
    assert traced["register"][0]["endpoint"] == endpoint + "/v1/traces"
    assert traced["register"][0]["project_name"] == "callas-web"


def test_web_hides_call_text(traced):
    callas_tracing.setup_tracing("callas-web", call_text=False)
    assert len(traced["config"]) == 3  # LangChain, DSPy, LiteLLM
    assert all(c.hide_inputs and c.hide_outputs for c in traced["config"])


def test_simulator_keeps_call_text(traced):
    callas_tracing.setup_tracing("callas-simulator", call_text=True)
    assert all(not c.hide_inputs and not c.hide_outputs for c in traced["config"])


def test_second_call_does_nothing(traced):
    callas_tracing.setup_tracing("callas-web", call_text=False)
    callas_tracing.setup_tracing("callas-web", call_text=False)
    assert len(traced["register"]) == 1
    assert len(traced["config"]) == 3


def test_tracing_setting_is_validated(monkeypatch):
    monkeypatch.setenv("TRACING", "langsmith")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="TRACING"):
        get_settings()
    get_settings.cache_clear()
