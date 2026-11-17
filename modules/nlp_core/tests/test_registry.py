import pytest

import callas_settings
from llm_serving.fake import FakeLLM
from nlp_core import parts, registry
from nlp_core.registry import LLMClassifier, LLMSummarizer, RulesNER


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    saved_f = {k: dict(v) for k, v in registry._FACTORIES.items()}
    saved_i = dict(registry._INSTANCES)
    for var in registry.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    # ignore the developer's own .env
    monkeypatch.setattr(
        callas_settings, "get_settings", lambda: callas_settings.Settings(_env_file=None)
    )
    yield
    registry._FACTORIES.clear()
    registry._FACTORIES.update(saved_f)
    registry._INSTANCES.clear()
    registry._INSTANCES.update(saved_i)


def test_parts_are_importable_from_the_registry_and_from_parts():
    assert registry.RulesNER is parts.RulesNER
    assert registry.LLMClassifier is parts.LLMClassifier
    assert registry.LLMSummarizer is parts.LLMSummarizer


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


def test_dotenv_is_read_and_the_process_environment_wins(monkeypatch, tmp_path):
    class Mine:
        def predict(self, text):
            return False, None, 0.5

    registry.register("classifier", "mine", Mine)
    env_file = tmp_path / ".env"
    env_file.write_text("NLP_CLASSIFIER=mine" + chr(10), encoding="utf-8")
    monkeypatch.setattr(
        callas_settings, "get_settings", lambda: callas_settings.Settings(_env_file=env_file)
    )
    assert registry.default_name("classifier") == "mine"
    assert isinstance(registry.get("classifier"), Mine)
    monkeypatch.setenv("NLP_CLASSIFIER", "llm")
    assert registry.default_name("classifier") == "llm"


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


def test_a_failed_summary_still_gives_the_scam_label():
    class DownSummarizer:
        def summarize(self, text):
            raise PermissionError("model turned off")

    registry.register("ner", "r", RulesNER)
    registry.register("classifier", "c", lambda: LLMClassifier(FakeLLM({"classify": "romance"})))
    registry.register("summarizer", "down", DownSummarizer)
    r = registry.analyze_with_tools("x", "r", "c", "down")
    assert (r.label_scam, r.label_type, r.summary) == (True, "romance", "")


def save_model(encoder, task):
    folder = registry.model_dir(encoder, task)
    folder.mkdir(parents=True)
    (folder / "preprocessing.json").write_text("{}", encoding="utf-8")


def test_the_built_in_classifier_is_best_once_both_models_are_saved():
    gate, type_ = registry.BEST_FINE_TUNED
    assert registry.default_name("classifier") == "llm"  # a fresh clone has no models/
    save_model(gate, "gate")
    assert registry.default_name("classifier") == "llm"  # the type model is still missing
    save_model(type_, "type")
    assert registry.default_name("classifier") == "best"


def test_a_set_classifier_wins_over_the_saved_models(monkeypatch):
    gate, type_ = registry.BEST_FINE_TUNED
    save_model(gate, "gate")
    save_model(type_, "type")
    monkeypatch.setenv("NLP_CLASSIFIER", "llm")
    assert registry.default_name("classifier") == "llm"
