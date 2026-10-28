"""The call record and its parts, as the simulator works with them.

A `CallRecord` has the spec's fields at the top, a `Generation` block (how the call was made: the
turns, the scam caller's plan, the judge's answer) and a `Review` block (a person's decision).
Dicts appear only at the edge, through `from_dict` and `to_dict`: Store.load_call and
Store.save_call, the JSON export and the studio's events. `to_dict` writes the keys in the order
the store has always used, so the stored JSON and the exports stay the same byte for byte.

Two names mean two things, and both are stored, so they stay:
- "rejected" in `Generation.status` means the simulator's checks failed the call; in
  `Review.status` it means a person rejected it on the /dataset page.
- `Generation.call` is the sampled scenarios.CallConfig as a dict; elsewhere `call` is that
  CallConfig, and the record itself is `record`.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

GenerationStatus = Literal["in_progress", "accepted", "rejected"]  # the simulator's decision
ReviewStatus = Literal["pending", "approved", "rejected"]  # a person's decision


@dataclass(kw_only=True)
class Turn:
    """One line of the call, as kept in generation.turns."""

    speaker: str  # turns.CALLER or turns.RECEIVER
    agent: str = ""  # scam_caller, legit_caller or receiver
    raw: str = ""  # the line as the agent wrote it, with {SLOTS}
    text: str = ""  # the line with this call's values in the slots
    spoken_text: str = ""  # what the voice is given: numbers written out (spoken.py)
    entities: list[dict] = field(default_factory=list)  # the filled values: label, span in text
    asks_now: bool = False  # the scam caller says this line makes its ask
    end_call: bool = False  # the speaker hangs up after this line
    reasoning: str = ""
    problems: list[str] = field(default_factory=list)  # reasons to reject the call
    leaked: list[str] = field(default_factory=list)  # receiver only: own details given away
    spelling: list[str] = field(default_factory=list)  # words that look misspelled
    latency_s: float | None = None
    prompt_tokens: int | None = None  # estimated size of the prompt that was sent
    audio: str | None = None  # the spoken line, relative to the store's folder
    audio_s: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Turn:
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(kw_only=True)
class Briefing:
    """The scam caller's plan, kept in generation.briefing."""

    plan: list[str]  # the steps the caller follows
    notes_used: list[str] = field(default_factory=list)  # summaries of the notes it read
    prompt_tokens: int | None = None  # None in briefings stored before prompt sizes were kept
    reasoning: str = ""
    # Briefings from an early planner kept note ids instead of notes_used. They are only read
    # back from old stores; nothing writes them now.
    note_ids: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Briefing:
        return cls(**data)

    def to_dict(self) -> dict:
        briefing: dict[str, Any] = {"plan": self.plan}
        if self.note_ids is not None:
            briefing["note_ids"] = self.note_ids
        else:
            briefing["notes_used"] = self.notes_used
        if self.prompt_tokens is not None:
            briefing["prompt_tokens"] = self.prompt_tokens
        briefing["reasoning"] = self.reasoning
        return briefing


@dataclass(kw_only=True)
class Judge:
    """The judge's answer, kept in generation.judge."""

    label_holds: bool  # the call shows what its label says
    scam_start_turn: int  # -1: no scam move
    giveaway: bool  # the call gives itself away as made up
    summary: str
    issues: str
    reasoning: str
    # The four scored checks. None means a record made before that check existed.
    scam_clear: bool | None = None
    scam_clarity_reason: str = ""
    label_reason: str = ""
    language_realistic: bool | None = None
    language_realism_reason: str = ""
    setup_holds: bool | None = None  # the call reflects the persona and scenario selected for it
    setup_reason: str = ""
    confidence: float | None = None
    # 1 to 5, how natural the Thai sounds. Only in answers stored before #98: the judge scored
    # calls down for spelling mistakes that were not there, so it is no longer asked.
    natural_thai: int | None = None
    prompt_tokens: int | None = None  # None in answers stored before prompt sizes were kept
    skipped: str = ""  # finish.TOO_LONG_FOR_JUDGE when the call was not judged at all

    @classmethod
    def from_dict(cls, data: dict) -> Judge:
        return cls(**data)

    def to_dict(self) -> dict:
        # The stored order: prompt_tokens first for a judged call, last for a skipped one.
        answer = {
            "label_holds": self.label_holds,
            **({} if self.scam_clear is None else {"scam_clear": self.scam_clear}),
            **(
                {}
                if not self.scam_clarity_reason
                else {"scam_clarity_reason": self.scam_clarity_reason}
            ),
            **({} if not self.label_reason else {"label_reason": self.label_reason}),
            **(
                {}
                if self.language_realistic is None
                else {"language_realistic": self.language_realistic}
            ),
            **(
                {}
                if not self.language_realism_reason
                else {"language_realism_reason": self.language_realism_reason}
            ),
            **({} if self.setup_holds is None else {"setup_holds": self.setup_holds}),
            **({} if not self.setup_reason else {"setup_reason": self.setup_reason}),
            "scam_start_turn": self.scam_start_turn,
            **({} if self.natural_thai is None else {"natural_thai": self.natural_thai}),
            "giveaway": self.giveaway,
            **({} if self.confidence is None else {"confidence": self.confidence}),
            "summary": self.summary,
            "issues": self.issues,
            "reasoning": self.reasoning,
        }
        if self.skipped:
            return {**answer, "skipped": self.skipped, "prompt_tokens": self.prompt_tokens}
        if self.prompt_tokens is None:
            return answer
        return {"prompt_tokens": self.prompt_tokens, **answer}


# Fields are declared in the order of the stored keys: to_dict relies on it.
@dataclass(kw_only=True)
class Generation:
    """How the call was made. The dataset build drops this block."""

    schema: int  # store.RECORD_SCHEMA
    status: GenerationStatus = "in_progress"
    reasons: list[str] = field(default_factory=list)  # reasons to reject; empty when accepted
    config_name: str
    config_hash: str
    models: dict[str, str]  # agent -> model name
    created_at: str
    # The sampled CallConfig as JSON, with receiver_override (a hash of a hand-chosen receiver)
    # and, for a studio call, receiver_name. Kept as a dict: it is written and read back whole.
    call: dict[str, Any]
    voices: dict[str, str]  # speaker -> TTS voice
    briefing: Briefing | None = None  # None for a normal call, and before the plan is written
    turns: list[Turn] = field(default_factory=list)
    receiver_leaked: list[str] = field(default_factory=list)  # every kind given away, sorted
    first_leak_turn: int | None = None
    speech_style: str  # dash | pause: how numbers are said
    scam_start_from: str | None = None  # caller | judge: who pointed at scam_start_turn
    judge: Judge | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Generation:
        briefing, judge = data["briefing"], data["judge"]
        return cls(
            **{
                **data,
                "briefing": None if briefing is None else Briefing.from_dict(briefing),
                "turns": [Turn.from_dict(turn) for turn in data["turns"]],
                "judge": None if judge is None else Judge.from_dict(judge),
            }
        )

    def to_dict(self) -> dict:
        generation = {f.name: getattr(self, f.name) for f in fields(self)}
        generation["briefing"] = None if self.briefing is None else self.briefing.to_dict()
        generation["turns"] = [turn.to_dict() for turn in self.turns]
        generation["judge"] = None if self.judge is None else self.judge.to_dict()
        return generation


@dataclass(kw_only=True)
class Review:
    """A person's decision on the /dataset page."""

    status: ReviewStatus = "pending"
    note: str = ""
    reviewed_at: str | None = None


# Fields are declared in the order of the stored keys: to_dict relies on it.
@dataclass(kw_only=True)
class CallRecord:
    """One simulated call: the spec's fields, then the generation and review blocks."""

    id: str
    channel: str = "call"
    text: str = ""  # one "A: ..." or "B: ..." line per turn, once the call is finished
    audio_path: str | None = None  # the joined call audio, relative to the store's folder
    speaker_turns: list[dict] = field(default_factory=list)  # speaker, text, start, end
    label_scam: bool
    label_type: str | None  # the scam type; None for a normal call
    entities: list[dict] = field(default_factory=list)  # spans within `text`
    reference_summary: str = ""  # the judge's summary
    scam_start_turn: int | None = None
    source: str = "synthetic"
    generation: Generation
    review: Review = field(default_factory=Review)

    @classmethod
    def from_dict(cls, data: dict) -> CallRecord:
        return cls(
            **{
                **data,
                "generation": Generation.from_dict(data["generation"]),
                "review": Review(**data["review"]),
            }
        )

    def to_dict(self) -> dict:
        """The record as the store and the exports keep it. It shares nothing with the record."""
        record = {f.name: getattr(self, f.name) for f in fields(self)}
        record["generation"] = self.generation.to_dict()
        record["review"] = asdict(self.review)
        return copy.deepcopy(record)
