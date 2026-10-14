import pytest

from llm_serving.base import JOBS
from llm_serving.prompts import load_prompt, render, strip_comments


@pytest.mark.parametrize("job", sorted(JOBS))
def test_prompt_loads_with_placeholder_last(job):
    text, version = load_prompt(job)
    assert version == ("v2" if job == "reply" else "v1")
    assert "{context}" in text
    assert "#" not in "".join(ln[:1] for ln in text.splitlines())
    # fixed instructions come before the context (prefix caching)
    assert text.index("{context}") > len(text) // 2


def test_reply_comment_is_stripped():
    text, _ = load_prompt("reply")
    assert "SIGN-OFF" not in text
    assert not any(ln.lstrip().startswith("#") for ln in text.splitlines())


def test_strip_comments_keeps_hash_inside_line():
    assert strip_comments("# c\nabc # keep\n  # c2\nend") == "abc # keep\nend\n"


def test_render_fills_context_and_tolerates_braces():
    text, _ = load_prompt("classify")
    out = render(text, "สวัสดี {x} {0}")
    assert "สวัสดี {x} {0}" in out
    assert "{context}" not in out


def test_highest_version_wins(tmp_path):
    (tmp_path / "job.v1.txt").write_text("one {context}", encoding="utf-8")
    (tmp_path / "job.v2.txt").write_text("# note\ntwo {context}", encoding="utf-8")
    (tmp_path / "job.v10.txt").write_text("ten {context}", encoding="utf-8")
    assert load_prompt("job", tmp_path) == ("ten {context}\n", "v10")


def test_missing_prompt_and_missing_placeholder(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompt("nope", tmp_path)
    (tmp_path / "bad.v1.txt").write_text("no placeholder", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt("bad", tmp_path)
