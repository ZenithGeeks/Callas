import pytest
from llm_serving.fake import FakeLLM
from nlp_core import registry
from nlp_core.registry import LLMClassifier, LLMSummarizer, RulesNER


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    saved_f = {k: dict(v) for k, v in registry._FACTORIES.items()}
    saved_i = dict(registry._INSTANCES)
    for var in registry.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved_f)
    registry._INSTANCES.clear()
    registry._INSTANCES.update(saved_i)


def test_default_selection():
    assert isinstance(registry.get("ner"), RulesNER)
    assert isinstance(registry.get("classifier"), LLMClassifier)
    assert isinstance(registry.get("summarizer"), LLMSummarizer)


def test_env_override(monkeypatch):
    class Mine:
        def predict(self, text):
            return False, None, 0.5

    registry.register("classifier", "mine", Mine)
    monkeypatch.setenv("NLP_CLASSIFIER", "mine")
    assert isinstance(registry.get("classifier"), Mine)
    assert isinstance(registry.get("classifier", "llm"), LLMClassifier)  # explicit name wins


def test_unknown_name_lists_available(monkeypatch):
    monkeypatch.setenv("NLP_NER", "wangchanberta")
    with pytest.raises(KeyError) as e:
        registry.get("ner")
    assert "wangchanberta" in str(e.value) and "rules" in str(e.value)


def test_unknown_kind():
    with pytest.raises(KeyError):
        registry.get("nope")


def test_instances_cached_and_replaced_on_register():
    a = registry.get("ner")
    assert registry.get("ner") is a
    registry.register("ner", "rules", RulesNER)
    assert registry.get("ner") is not a


def test_llm_classifier_mapping():
    def make(raw):
        return LLMClassifier(FakeLLM({"classify": raw}))

    assert make("normal").predict("x") == (False, None, 1.0)
    assert make("lottery").predict("x") == (True, "lottery", 1.0)
    assert make("???").predict("x") == (True, None, 0.0)


def test_llm_summarizer_and_rules_ner():
    assert LLMSummarizer(FakeLLM({"summary": "  s "})).summarize("x") == "s"
    assert [e.label for e in RulesNER().extract("โทร 081-234-5678")] == ["PHONE"]


def test_analyze_with_tools_composes():
    registry.register("ner", "r", RulesNER)
    registry.register("classifier", "c", lambda: LLMClassifier(FakeLLM({"classify": "romance"})))
    registry.register("summarizer", "s", lambda: LLMSummarizer(FakeLLM({"summary": "sum"})))
    r = registry.analyze_with_tools("โทร 081-234-5678", "r", "c", "s")
    assert (r.label_scam, r.label_type, r.confidence, r.summary) == (True, "romance", 1.0, "sum")
    assert [e.label for e in r.entities] == ["PHONE"]
