import pytest

import nlp_core
from llm_serving.client import LLMUnavailable
from llm_serving.fake import FakeLLM
from nlp_core.analyzer import Analyzer

TEXT = "สวัสดีครับ โทร 081-234-5678"


def run(classify):
    llm = FakeLLM({"classify": classify, "summary": "  สรุป  "})
    return Analyzer(llm).analyze(TEXT), llm


def test_normal():
    r, llm = run("normal")
    assert (r.label_scam, r.label_type, r.confidence) == (False, None, 1.0)
    assert r.summary == "สรุป"
    assert [e.label for e in r.entities] == ["PHONE"]
    assert llm.calls_for("summary") == [TEXT]
    assert llm.calls_for("classify") == [TEXT]


@pytest.mark.parametrize(
    "label", ["fake_authority", "romance", "investment", "phishing_otp", "lottery"]
)
def test_scam_labels(label):
    r, _ = run(f'"{label.upper()}"')
    assert (r.label_scam, r.label_type, r.confidence) == (True, label, 1.0)


@pytest.mark.parametrize("raw", ["", "not sure", "normal lottery", "scam"])
def test_unknown_escalates(raw):
    r, _ = run(raw)
    assert (r.label_scam, r.label_type, r.confidence) == (True, None, 0.0)


def test_llm_unavailable_propagates():
    class Down:
        def complete(self, job, context):
            raise LLMUnavailable("down")

    with pytest.raises(LLMUnavailable):
        Analyzer(Down()).analyze(TEXT)


def test_module_level_analyze_uses_default(monkeypatch):
    monkeypatch.setattr(nlp_core, "_default", Analyzer(FakeLLM({"classify": "lottery"})))
    assert nlp_core.analyze(TEXT).label_type == "lottery"
