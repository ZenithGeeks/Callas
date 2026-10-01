import pytest
from llm_serving.fake import FakeLLM


def test_scripted_and_recorded():
    f = FakeLLM({"classify": ["lottery", "normal"], "summary": lambda c: c[:3]})
    assert f.complete("classify", "a") == "lottery"
    assert f.complete("classify", "b") == "normal"
    assert f.complete("classify", "c") == "normal"
    assert f.complete("summary", "abcdef") == "abc"
    assert f.calls_for("classify") == ["a", "b", "c"]
    assert f.calls[-1] == ("summary", "abcdef")


def test_unknown_job():
    with pytest.raises(ValueError):
        FakeLLM().complete("nope", "x")
