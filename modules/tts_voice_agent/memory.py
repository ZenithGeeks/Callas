"""Call memory: one CallState per call, in memory only, cleared when the call ends."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import cache

from asr_diarization.types import Turn
from nlp_core.types import Entity

from .constants import CONTEXT_TOKEN_BUDGET, RECENT_TURNS_VERBATIM, SUMMARY_FALLBACK_CHARS


def default_count_tokens(text: str) -> int:
    """Number of PyThaiNLP newmm word tokens."""
    from pythainlp.tokenize import word_tokenize

    return len(word_tokenize(text, engine="newmm", keep_whitespace=False))


def default_summarize(text: str) -> str:
    """No-model fallback: keep the most recent part of the text."""
    return text[-SUMMARY_FALLBACK_CHARS:]


def clean_turn(turn: Turn) -> Turn:
    """The turn with clean text: normalized, spoken numbers as digits, maybe corrected.

    raw keeps the original ASR text.
    """
    from nlp_core.numbers import spoken_numbers_to_digits
    from nlp_core.text import autocorrect_text, normalize_text

    # numbers spoken as words become digits so the entity rules can read them
    text = autocorrect_text(spoken_numbers_to_digits(normalize_text(turn.text)))
    if text == turn.text:
        return turn
    return replace(turn, text=text, raw=turn.raw if turn.raw is not None else turn.text)


def _line(turn: Turn) -> str:
    """One turn as it appears in the context text."""
    return f"{turn.speaker}: {turn.text}"


@dataclass
class CallState:
    """Everything remembered about one call. Lives in memory only; end() deletes it."""

    turns: list[Turn] = field(default_factory=list)
    older_summary: str = ""  # summary of turns that no longer fit the budget
    entities_seen: list[Entity] = field(default_factory=list)  # kept for the whole call
    escalated: bool = False  # once True, stays True until the call ends

    def add(self, turn: Turn, entities: Iterable[Entity] = ()) -> None:
        """Remember a turn and the entities found in it."""
        self.turns.append(turn)
        self.add_entities(entities)

    def add_entities(self, entities: Iterable[Entity]) -> None:
        """Remember entities, deduplicated by (text, label). Never dropped."""
        known = {(e.text, e.label) for e in self.entities_seen}
        for e in entities:
            if (e.text, e.label) not in known:
                known.add((e.text, e.label))
                self.entities_seen.append(e)

    def _render(self, turns: list[Turn], summary: str | None = None) -> str:
        """The summary line (if any) followed by one line per turn."""
        summary = self.older_summary if summary is None else summary
        parts = [summary] if summary else []
        parts.extend(_line(t) for t in turns)
        return "\n".join(parts)

    def _fold(self, count: int, summarize: Callable[[str], str]) -> None:
        """Move the `count` oldest turns into older_summary."""
        old, self.turns = self.turns[:count], self.turns[count:]
        lines = [_line(t) for t in old]
        self.older_summary = summarize("\n".join([self.older_summary, *lines]).strip())

    def _truncate_summary(self, count_tokens: Callable[[str], int], budget: int) -> None:
        """Cut older_summary (keeping its newest end) until the whole context fits."""
        summary = self.older_summary
        # turn tokens are constant across probes, so only the summary suffix is counted
        room = budget - count_tokens(self._render(self.turns, ""))
        # binary search for the longest suffix of the summary that still fits
        kept, most = 0, len(summary) if room > 0 else 0
        while kept < most:
            mid = (kept + most + 1) // 2
            if count_tokens(summary[len(summary) - mid :]) <= room:
                kept = mid
            else:
                most = mid - 1
        # token counts of the pieces may not add up to the whole: verify, then shrink
        while (
            kept and count_tokens(self._render(self.turns, summary[len(summary) - kept :])) > budget
        ):
            kept -= max(1, kept // 8)
        self.older_summary = summary[len(summary) - kept :] if kept else ""

    def context(
        self,
        count_tokens: Callable[[str], int] = default_count_tokens,
        summarize: Callable[[str], str] = default_summarize,
        budget: int = CONTEXT_TOKEN_BUDGET,
    ) -> str:
        """Older summary plus recent turns word for word, never over `budget` tokens.

        Over budget: fold turns older than the last RECENT_TURNS_VERBATIM into the summary,
        then truncate the summary; only if the recent turns alone are too long, fold those too.
        """
        count = cache(count_tokens)  # identical strings are tokenized once per call

        text = self._render(self.turns)
        if count(text) <= budget:
            return text
        extra = len(self.turns) - RECENT_TURNS_VERBATIM
        if extra > 0:
            self._fold(extra, summarize)
        while True:
            text = self._render(self.turns)
            if count(text) <= budget:
                return text
            self._truncate_summary(count, budget)
            text = self._render(self.turns)
            if count(text) <= budget:
                return text
            self._fold(1, summarize)  # recent turns alone exceed the budget

    def end(self) -> None:
        """The call is over: delete everything."""
        self.turns.clear()
        self.older_summary = ""
        self.entities_seen.clear()
        self.escalated = False
