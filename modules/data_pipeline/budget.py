"""Keep every prompt the simulator sends inside the model's context window.

qwen3.8 on the gateway has a 100K-token window (`context.window_tokens` in agents.yaml). A
prompt plus the answer (the agent's `lm.max_tokens`) plus a safety margin has to fit in it, so
each agent's prompt budget is

    window_tokens - lm.max_tokens - safety_margin_tokens

The prompt is measured, not guessed from the inputs: `prompt_tokens` formats the exact messages
DSPy sends (ChatAdapter, with the reasoning field ChainOfThought adds) and estimates their size.
The helpers below are pure: they take a `fits` function and cut the input until it passes, so
they are tested without a model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import dspy

from .config import Config

# Rounded up. Thai text is about one model token per character and a Thai character is 3 UTF-8
# bytes, so bytes / 3 is the worst case. English is about 4 bytes per token, so this over-counts
# English text, which is the safe side.
BYTES_PER_TOKEN = 3
KEEP_FIRST_TURNS = 2  # a trimmed conversation always keeps its opening turns


def estimate_tokens(text: str) -> int:
    """The most tokens this text can be, from its UTF-8 size (see BYTES_PER_TOKEN)."""
    return math.ceil(len(text.encode("utf-8")) / BYTES_PER_TOKEN)


def prompt_tokens(signature: type[dspy.Signature], inputs: dict[str, Any]) -> int:
    """Estimated tokens of the prompt DSPy would send for these inputs.

    `signature` is the one the program really uses (`program.predict.signature`, which for a
    ChainOfThought already has the `reasoning` output field), so the system message holds the
    same instructions and field descriptions."""
    messages = dspy.ChatAdapter().format(signature, demos=[], inputs=inputs)
    return sum(estimate_tokens(str(message["content"])) for message in messages)


def prompt_budget(config: Config, agent: str) -> int:
    """Tokens a prompt for this agent may use."""
    context = config.agents.context
    max_tokens = getattr(config.agents, agent).lm.max_tokens
    return context.window_tokens - max_tokens - context.safety_margin_tokens


def hidden_marker(count: int) -> str:
    """The line that stands in for the turns that were cut from the middle."""
    noun = "turn" if count == 1 else "turns"
    return f"(... {count} earlier {noun} not shown ...)"


class ContextTooLong(Exception):
    """A prompt does not fit its agent's budget even after everything that can be cut was cut."""

    def __init__(self, agent: str, tokens: int, budget: int) -> None:
        super().__init__(f"{agent}: prompt of ~{tokens} tokens exceeds its budget of {budget}")
        self.agent, self.tokens, self.budget = agent, tokens, budget


def _most_that_fit(low: int, high: int, fits_at: Callable[[int], bool], default: int) -> int:
    """The largest n from low to high for which fits_at(n) is true, by binary search.

    fits_at must be true up to some n and false after it (a longer prompt only gets bigger).
    `default` is returned when no n fits."""
    best = default
    while low <= high:
        mid = (low + high) // 2
        if fits_at(mid):
            best, low = mid, mid + 1
        else:
            high = mid - 1
    return best


def trim_turns(lines: Sequence[str], fits: Callable[[str], bool]) -> tuple[str, int]:
    """The conversation text, cut to fit. Returns (text, number of turns left out).

    If the whole conversation fits it is returned as it is. Otherwise the first two turns and
    as many of the newest turns as fit are kept, with one marker line for the ones between.
    `fits` takes the conversation text. If even the first turns and the newest one do not fit,
    the smallest version is returned anyway; the caller still sends it (nothing else can be
    cut)."""
    lines = list(lines)
    whole = "\n".join(lines)
    if fits(whole) or len(lines) <= KEEP_FIRST_TURNS + 1:
        return whole, 0
    head = lines[:KEEP_FIRST_TURNS]

    def build(newest: int) -> tuple[str, int]:
        tail = lines[len(lines) - newest :] if newest else []
        hidden = len(lines) - KEEP_FIRST_TURNS - newest
        return "\n".join([*head, hidden_marker(hidden), *tail]), hidden

    # always keep at least the newest turn; more newest turns means a longer prompt
    most_newest = len(lines) - KEEP_FIRST_TURNS - 1
    return build(_most_that_fit(1, most_newest, lambda n: fits(build(n)[0]), default=1))


def trim_notes(notes: Sequence[dict], fits: Callable[[Sequence[dict]], bool]) -> list[dict]:
    """Pattern notes, newest first, with the oldest dropped until `fits` accepts them."""
    kept = list(notes)
    while kept and not fits(kept):
        kept.pop()
    return kept


def cut_article(article: str, fits: Callable[[str], bool], upper: int | None = None) -> str:
    """The longest start of the article that fits (and is at most `upper` characters)."""
    if upper is not None:
        article = article[:upper]
    if fits(article):
        return article
    return article[: _most_that_fit(0, len(article) - 1, lambda n: fits(article[:n]), default=0)]
