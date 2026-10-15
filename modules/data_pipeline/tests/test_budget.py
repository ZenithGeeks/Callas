import shutil
from types import SimpleNamespace

import dspy
import pytest
import yaml

from data_pipeline import simulate
from data_pipeline.agents import configured
from data_pipeline.budget import (
    cut_article,
    estimate_tokens,
    hidden_marker,
    prompt_budget,
    prompt_tokens,
    trim_notes,
    trim_turns,
)
from data_pipeline.config import DEFAULT_DIR, ConfigError, load_config
from data_pipeline.patterns import MAX_ARTICLE_CHARS, extract
from data_pipeline.scenarios import sample_config
from data_pipeline.store import Store
from data_pipeline.tests.test_simulate import REPLIES, SCAM, judge, make_agents

CONFIG = load_config()
WINDOW, MARGIN = 100_000, 1_000


def config_with_budget(tmp_path, agent, budget):
    """A config in which `agent` may use exactly `budget` tokens for its prompt."""
    folder = tmp_path / f"cfg_{agent}_{budget}"
    shutil.copytree(DEFAULT_DIR, folder)
    path = folder / "agents.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["context"] = {"window_tokens": WINDOW, "safety_margin_tokens": MARGIN}
    data[agent]["lm"]["max_tokens"] = WINDOW - MARGIN - budget
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config = load_config(folder)
    assert prompt_budget(config, agent) == budget
    return config


def long_call(n=40):
    """n turns, A and B in turn, each about 40 tokens."""
    return [
        {
            "speaker": "AB"[i % 2],
            "raw": f"ข้อความที่ {i} " + "ก" * 40,
            "text": f"ข้อความที่ {i} " + "ก" * 40,
        }
        for i in range(n)
    ]


# ---- the estimate ------------------------------------------------------------------------


def test_estimate_tokens_thai_is_one_token_per_character():
    assert estimate_tokens("") == 0
    assert estimate_tokens("สวัสดี") == 6  # 6 Thai characters, 18 bytes
    assert estimate_tokens("ก" * 100) == 100


def test_estimate_tokens_over_counts_english():
    assert estimate_tokens("hello") == 2  # 5 bytes / 3, rounded up
    assert estimate_tokens("a" * 300) == 100  # a real tokenizer needs about 75


def test_prompt_tokens_measures_the_prompt_dspy_sends():
    agents = make_agents([], [], judge())
    signature = agents.signatures["receiver"]
    inputs = dict(persona={"age": 40}, voice="v", stage="s", conversation="")
    base = prompt_tokens(signature, inputs)
    assert base > estimate_tokens(signature.instructions)  # system message + fields + inputs
    more = prompt_tokens(signature, {**inputs, "conversation": "ก" * 300})
    assert more - base == pytest.approx(300, abs=2)


# ---- the pure helpers ----------------------------------------------------------------------


def test_trim_turns_keeps_the_first_two_and_the_newest():
    lines = [f"line {i:02d}" for i in range(10)]
    text, hidden = trim_turns(lines, lambda t: len(t) <= 70)
    kept = text.split("\n")
    assert kept[:2] == lines[:2]
    assert kept[2] == hidden_marker(hidden)
    assert kept[3:] == lines[len(lines) - len(kept[3:]) :] and kept[-1] == lines[-1]
    assert hidden == 10 - 2 - len(kept[3:]) > 0
    assert len(text) <= 70
    assert len(kept[3:]) == max(k for k in range(1, 9) if len("\n".join(_with(lines, k))) <= 70)


def _with(lines, newest):
    hidden = len(lines) - 2 - newest
    return [*lines[:2], hidden_marker(hidden), *lines[len(lines) - newest :]]


def test_trim_turns_leaves_a_call_that_fits_alone():
    lines = ["a", "b", "c", "d"]
    assert trim_turns(lines, lambda t: True) == ("a\nb\nc\nd", 0)


def test_trim_turns_always_keeps_the_newest_turn():
    lines = [f"line {i}" for i in range(6)]
    text, hidden = trim_turns(lines, lambda t: False)  # nothing fits: the smallest version
    assert text.split("\n") == [*lines[:2], hidden_marker(3), "line 5"]


def test_trim_notes_drops_the_oldest_first():
    notes = [{"summary": str(i)} for i in range(5)]  # newest first
    kept = trim_notes(notes, lambda n: len(n) <= 3)
    assert [n["summary"] for n in kept] == ["0", "1", "2"]
    assert trim_notes(notes, lambda n: True) == notes
    assert trim_notes(notes, lambda n: False) == []


def test_cut_article_takes_the_longest_start_that_fits():
    article = "ก" * 1000
    assert cut_article(article, lambda a: len(a) <= 123) == "ก" * 123
    assert cut_article(article, lambda a: True, upper=50) == "ก" * 50
    assert cut_article(article, lambda a: True) == article


# ---- speakers ----------------------------------------------------------------------------


def test_a_long_conversation_is_trimmed_with_a_marker(tmp_path):
    call = sample_config("phishing_otp", 1, CONFIG)
    turns = long_call(40)
    lines = simulate.conversation_lines(turns, CONFIG, raw=True)
    answer = dict(line="ครับ", asks_now=False, end_call=False)

    roomy = make_agents([answer], [], judge())
    _, _, whole_tokens = simulate._caller_says(roomy, call, CONFIG, turns, ["plan"])
    assert roomy.scam_caller.calls[0]["conversation"] == "\n".join(lines)  # fits: untouched

    budget = whole_tokens - 600
    tight = config_with_budget(tmp_path, "scam_caller", budget)
    agents = make_agents([answer], [], judge(), config=tight)
    _, _, tokens = simulate._caller_says(agents, call, tight, turns, ["plan"])

    sent = agents.scam_caller.calls[0]["conversation"].split("\n")
    shown_newest = sent[3:]
    assert sent[:2] == lines[:2]  # the opening turns stay
    assert shown_newest == lines[len(lines) - len(shown_newest) :]  # the newest turns stay
    hidden = len(lines) - 2 - len(shown_newest)
    assert 0 < len(shown_newest) < 38 and sent[2] == hidden_marker(hidden)
    assert tokens <= budget  # the estimate returned is what was measured


def test_the_receiver_is_trimmed_too(tmp_path):
    call = sample_config("phishing_otp", 1, CONFIG)
    turns = long_call(40)
    roomy = make_agents([], [dict(line="ค่ะ", end_call=False)], judge())
    _, whole_tokens = simulate._receiver_says(roomy, call, CONFIG, turns)
    tight = config_with_budget(tmp_path, "receiver", whole_tokens - 600)
    agents = make_agents([], [dict(line="ค่ะ", end_call=False)], judge(), config=tight)
    simulate._receiver_says(agents, call, tight, turns)
    sent = agents.receiver.calls[0]["conversation"]
    assert "earlier turns not shown" in sent and sent.endswith(turns[-1]["text"])


def test_every_turn_records_its_prompt_size(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    rec = simulate.run_call("phishing_otp", 1, Store(tmp_path / "calls.db"), agents, CONFIG)
    sizes = [t["prompt_tokens"] for t in rec["generation"]["turns"]]
    assert all(isinstance(n, int) and n > 0 for n in sizes)
    assert sizes[4] > sizes[0]  # the conversation grew
    assert rec["generation"]["briefing"]["prompt_tokens"] > 0


# ---- planner -----------------------------------------------------------------------------


def test_the_planner_drops_the_oldest_notes(tmp_path):
    store = Store(tmp_path / "calls.db")
    for i in range(6):  # added oldest first; the planner reads newest first
        store.add_note(
            "phishing_otp", f"note {i}", [f"ขั้นตอน {i} " + "ก" * 60] * 3, ["ประโยคตัวอย่าง " * 3]
        )
    call = sample_config("phishing_otp", 1, CONFIG)
    agents = make_agents([], [], judge())
    all_six = simulate._plan(agents, call, CONFIG, store)
    assert all_six["notes_used"] == [f"note {i}" for i in (5, 4, 3, 2, 1, 0)]

    budget = all_six["prompt_tokens"] - 1
    tight = config_with_budget(tmp_path, "planner", budget)
    agents = make_agents([], [], judge(), config=tight)
    briefing = simulate._plan(agents, call, tight, store)
    used = briefing["notes_used"]
    assert used == all_six["notes_used"][: len(used)] and 0 < len(used) < 6
    assert "note 0" not in used  # the oldest went first
    assert briefing["prompt_tokens"] <= budget
    assert "note 0" not in agents.planner.calls[0]["pattern_notes"]


# ---- judge -------------------------------------------------------------------------------


def test_a_call_too_long_for_the_judge_is_rejected_without_asking_it(tmp_path):
    agents = make_agents(SCAM, REPLIES, judge(start=2))
    fine = simulate.run_call("phishing_otp", 1, Store(tmp_path / "a.db"), agents, CONFIG)
    needed = fine["generation"]["judge"]["prompt_tokens"]
    assert fine["generation"]["status"] == "accepted" and len(agents.judge.calls) == 1

    tight = config_with_budget(tmp_path, "judge", needed - 1)
    agents = make_agents(SCAM, REPLIES, judge(start=2), config=tight)
    rec = simulate.run_call("phishing_otp", 1, Store(tmp_path / "b.db"), agents, tight)
    assert agents.judge.calls == []  # the judge program was never called
    assert rec["generation"]["status"] == "rejected"
    assert rec["generation"]["reasons"] == ["too_long_for_judge"]
    assert rec["generation"]["judge"]["skipped"] == "too_long_for_judge"


# ---- pattern extractor -------------------------------------------------------------------


class FakeExtractor:
    """Stands in for the extractor program: it has the real signature and records its inputs."""

    def __init__(self):
        signature = configured("pattern_extractor", CONFIG.agents.pattern_extractor)
        self.predict = SimpleNamespace(signature=dspy.ChainOfThought(signature).predict.signature)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return dspy.Prediction(relevant=False)


def extractor(config):
    return FakeExtractor()


def test_the_article_is_cut_to_the_prompt_budget():
    article = "ก" * 30_000
    program = extractor(CONFIG)
    extract(article, "romance", program, dspy.utils.DummyLM([]), budget=None)
    assert len(program.calls[0]["article"]) == MAX_ARTICLE_CHARS  # the old fixed cap

    signature = program.predict.signature
    base = prompt_tokens(signature, {"scam_type": "romance", "article": ""})
    program = extractor(CONFIG)
    extract(article, "romance", program, dspy.utils.DummyLM([]), budget=base + 500)
    cut = program.calls[0]["article"]
    assert 400 < len(cut) <= 500  # one token per Thai character
    assert prompt_tokens(signature, {"scam_type": "romance", "article": cut}) <= base + 500
    assert prompt_tokens(
        signature, {"scam_type": "romance", "article": article[: len(cut) + 1]}
    ) > (base + 500)


# ---- config ------------------------------------------------------------------------------


def edit_agents(tmp_path, change):
    folder = tmp_path / "cfg"
    shutil.copytree(DEFAULT_DIR, folder)
    path = folder / "agents.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return folder


def test_the_default_context_is_100k():
    assert CONFIG.agents.context.window_tokens == 100_000
    assert CONFIG.agents.context.safety_margin_tokens == 2_000


def test_max_tokens_that_leaves_no_room_is_refused_and_names_the_agent(tmp_path):
    folder = edit_agents(tmp_path, lambda d: d["judge"]["lm"].update(max_tokens=98_000))
    with pytest.raises(ConfigError, match="judge.lm.max_tokens"):
        load_config(folder)
    # just below the limit is fine
    folder = edit_agents(tmp_path / "ok", lambda d: d["judge"]["lm"].update(max_tokens=97_999))
    assert load_config(folder).agents.judge.lm.max_tokens == 97_999


@pytest.mark.parametrize("field", ["window_tokens", "safety_margin_tokens"])
@pytest.mark.parametrize("value", [0, -5])
def test_the_context_numbers_must_be_positive(tmp_path, field, value):
    folder = edit_agents(tmp_path, lambda d: d["context"].update({field: value}))
    with pytest.raises(ConfigError, match=field):
        load_config(folder)
