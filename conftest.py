"""Pin runtime switches for every test, so a developer's .env cannot change test behaviour."""

import pytest

from callas_settings import get_settings


@pytest.fixture(autouse=True)
def _default_switches(monkeypatch):
    monkeypatch.setenv("REPLY_ENGINE", "prompt")
    monkeypatch.setenv("LEAK_JUDGE", "off")
    for name in ("NLP_NER", "NLP_CLASSIFIER", "NLP_SUMMARIZER"):
        monkeypatch.setenv(name, "")  # empty means the built-in default, and hides .env
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
