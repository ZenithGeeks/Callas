"""Pin runtime switches for every test, so a developer's .env cannot change test behaviour."""

import pytest

from callas_settings import get_settings
from nlp_core import registry


@pytest.fixture(autouse=True)
def _default_switches(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLY_ENGINE", "prompt")
    monkeypatch.setenv("LEAK_JUDGE", "off")
    # TRACING=phoenix with no collector running blocks every request on the span export.
    monkeypatch.setenv("TRACING", "off")
    for name in ("NLP_NER", "NLP_CLASSIFIER", "NLP_SUMMARIZER"):
        monkeypatch.setenv(name, "")  # empty means the built-in default, and hides .env
    # The built-in classifier is "best" when models/ holds the fine-tuned models, which only
    # some machines have. Tests see none, so the default is "llm" everywhere.
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
