"""Shared fakes: registry entries that record calls, fake speaker."""

import pytest
from nlp_core import registry
from nlp_core.types import Entity


class FakeNER:
    def __init__(self):
        self.calls = []

    def extract(self, text):
        self.calls.append(text)
        return [Entity("0812345678", "PHONE", 0, 10)] if "PHONE" in text else []


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        if "SCAM" in text:
            return True, "fake_authority", 0.95
        return False, None, 1.0


class FakeSummarizer:
    def __init__(self):
        self.calls = []

    def summarize(self, text):
        self.calls.append(text)
        return "summary"


class FakeSpeaker:
    def __init__(self):
        self.spoken = []

    def synthesize_bytes(self, text):
        self.spoken.append(text)
        return b"RIFFfake"


@pytest.fixture
def fakes(monkeypatch):
    """Register fake NER/classifier/summarizer and select them through the env vars."""
    saved_f = {k: dict(v) for k, v in registry._FACTORIES.items()}
    saved_i = dict(registry._INSTANCES)
    made = {"ner": FakeNER(), "classifier": FakeClassifier(), "summarizer": FakeSummarizer()}
    for kind, inst in made.items():
        registry.register(kind, "fake", lambda inst=inst: inst)
        monkeypatch.setenv(registry.ENV_VARS[kind], "fake")
    yield made
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved_f)
    registry._INSTANCES.clear()
    registry._INSTANCES.update(saved_i)
