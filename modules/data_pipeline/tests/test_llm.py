import pytest

import callas_settings
from data_pipeline.llm import TyphoonNotConfigured, build_typhoon_lm, is_typhoon


def use_settings(monkeypatch, **values):
    settings = callas_settings.Settings(_env_file=None, **values)
    monkeypatch.setattr(callas_settings, "get_settings", lambda: settings)


def test_typhoon_model_names():
    assert is_typhoon("typhoon-v2.5-30b-a3b-instruct")
    assert not is_typhoon("qwen3.8-27b-fp8") and not is_typhoon(None) and not is_typhoon("")


def test_typhoon_without_its_env_lines_raises(monkeypatch):
    # no silent fallback to VLLM_MODEL: the config hash would name a model that is not used
    use_settings(monkeypatch, typhoon_base_url="", typhoon_api_key="")
    with pytest.raises(TyphoonNotConfigured, match="TYPHOON_BASE_URL"):
        build_typhoon_lm("typhoon-x", max_tokens=100, temperature=0.5)


def test_typhoon_lm_uses_the_settings(monkeypatch):
    use_settings(
        monkeypatch,
        typhoon_base_url="https://typhoon.example/v1",
        typhoon_api_key="secret",
        vllm_timeout_s=12.0,
    )
    lm = build_typhoon_lm("typhoon-x", max_tokens=100, temperature=0.5)
    assert lm.model == "openai/typhoon-x"
    assert lm.kwargs["api_base"] == "https://typhoon.example/v1"
    assert lm.kwargs["timeout"] == 12.0  # no TYPHOON_TIMEOUT_S: the vLLM timeout
