"""Pin runtime switches for every test, so a developer's .env cannot change test behaviour."""

import pytest

from callas_settings import get_settings


@pytest.fixture(autouse=True)
def _default_switches(monkeypatch):
    monkeypatch.setenv("REPLY_ENGINE", "prompt")
    monkeypatch.setenv("LEAK_JUDGE", "off")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
