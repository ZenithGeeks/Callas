"""Load and check the simulator config: three YAML files in one folder.

    agents.yaml     what each agent is told, its model settings, pacing text, TTS voices
    scenarios.yaml  what calls are about, who answers, how long calls run
    details.yaml    names, banks, links and amounts that fill the slots

The romance scam caller also reads docs/scam-wiki/romance_voice.md (see romance_voice.py); it
is part of the config hash.

The default folder is data_pipeline/config/. To try a variant, copy the folder, edit the copy
and pass --config <folder>. Every record stores the config name and hash, and a stopped run
only resumes with the same config.

Only text and numbers are configurable. Field names are the contract with the simulator code
(agents.py), so a config that adds, drops or renames a field fails to load.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nlp_core.types import SCAM_TYPES

from . import red_flags as red_flags_page
from . import romance_voice as romance_voice_page

DEFAULT_DIR = Path(__file__).resolve().parent / "config"
FILES = ("agents.yaml", "scenarios.yaml", "details.yaml")
BENIGN = "benign"
CALL_TYPES: tuple[str, ...] = (*SCAM_TYPES, BENIGN)
# Which way a scam call is meant to end for the receiver (ReceiverProfile.scam_outcome). Drawn
# 50/50 per call so the dataset is balanced between receivers who get scammed and those who do not.
SCAM_OUTCOMES = ("scammed", "safe")
SLOT_NAMES = (
    "CALLER_NAME",
    "RECEIVER_NAME",
    "ORG",
    "PHONE",
    "BANK_ACCOUNT",
    "OTP",
    "URL",
    "MONEY",
    "CARD_NUMBER",
    "RECEIVER_PHONE",
    "RECEIVER_ACCOUNT",
    "ID_CARD",
    "BIRTH_DATE",
    "ADDRESS",
)
_SLOT_IN_TEXT = re.compile(r"\{([A-Z_]+)\}")


class ConfigError(ValueError):
    """The config cannot be used. The message names the file and what to fix."""


class _Strict(BaseModel):
    """Base for every config section: an unknown key is an error, and values never change."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_slots(text: str, where: str) -> None:
    bad = sorted(set(_SLOT_IN_TEXT.findall(text)) - set(SLOT_NAMES))
    if bad:
        raise ValueError(f"{where}: unknown slot(s) {bad}; known slots are {list(SLOT_NAMES)}")


# agents.yaml


class LMSettings(_Strict):
    """The model settings of one agent."""

    model: str | None = None  # empty: VLLM_MODEL from .env (or the first served model)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=64)


class AgentSettings(_Strict):
    """One DSPy agent: its instructions, the description of each field, and its model."""

    instructions: str
    fields: dict[str, str]  # field name -> description shown to the model
    lm: LMSettings = LMSettings()  # the agents.yaml key; agents.Agents.lms holds the dspy.LMs


def _check_format(text: str, where: str, **values: int) -> None:
    """The text must survive str.format with the values the simulator passes."""
    try:
        text.format(**values)
    except (KeyError, IndexError, ValueError) as exc:
        names = ", ".join("{" + name + "}" for name in values)
        raise ValueError(
            f"{where}: cannot be formatted ({type(exc).__name__}: {exc}); "
            f"the only fields are {names}, and a literal brace is written {{{{ or }}}}"
        ) from exc


class Pacing(_Strict):
    """Scam caller pacing. {turn} and {turns} are filled in."""

    before: str  # before its reveal turn
    start: str  # on its reveal turn
    push: str  # after it
    last: str  # on its last two turns

    @model_validator(mode="after")
    def _formattable(self) -> Pacing:
        for name in ("before", "start", "push", "last"):
            _check_format(getattr(self, name), f"scam_caller.pacing.{name}", turn=1, turns=1)
        return self


class ScamCallerSettings(AgentSettings):
    """The scam caller, with the pacing text it gets on each turn.

    `instructions` are the rules every scam type shares; `by_type` adds each type's own rules.
    """

    pacing: Pacing
    by_type: dict[str, list[str]]  # scam type -> its rules, added after `instructions`

    @field_validator("by_type")
    @classmethod
    def _every_scam_type(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        if set(v) != set(SCAM_TYPES):
            raise ValueError(f"by_type needs exactly the keys {list(SCAM_TYPES)}")
        return v

    def instructions_for(self, scam_type: str) -> str:
        """The full instructions of one scam type: the shared rules, then the type's own."""
        parts = [self.instructions, *self.by_type[scam_type]]
        return "\n\n".join(part.strip() for part in parts)


class ReceiverStage(_Strict):
    """What the receiver is told about staying on the line. {turn} is filled in."""

    stay: str  # while it must stay on the line; {turn} is filled in
    free: str

    @model_validator(mode="after")
    def _formattable(self) -> ReceiverStage:
        # turns.stage passes only `turn`, so {turns} would fail there
        for name in ("stay", "free"):
            _check_format(getattr(self, name), f"receiver.stage.{name}", turn=1)
        return self


class ReceiverSettings(AgentSettings):
    """The person who answers: how many turns it must stay on the line, and its stage text."""

    stay_on_line: int = Field(ge=0)
    stage: ReceiverStage


class PlannerSettings(AgentSettings):
    """The agent that plans a scam call from recent pattern notes."""

    notes_limit: int = Field(6, ge=0)  # newest pattern notes the planner reads per call


class JudgeSettings(AgentSettings):
    """The agent that checks a finished call, with what each call type is meant to show."""

    intended: dict[str, str]  # call type -> what the call is meant to show
    realism_criteria: list[str] = Field(min_length=1)

    @field_validator("intended")
    @classmethod
    def _all_types(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v) != set(CALL_TYPES):
            raise ValueError(f"intended needs exactly the keys {list(CALL_TYPES)}")
        return v

    @field_validator("realism_criteria")
    @classmethod
    def _nonempty_criteria(cls, v: list[str]) -> list[str]:
        if any(not criterion.strip() for criterion in v):
            raise ValueError("realism_criteria cannot contain empty items")
        return v


class Ending(_Strict):
    """The note a speaker gets near the call's turn limit (turns.ending)."""

    soon: str  # on the second-to-last turn: the other person speaks once more
    last: str  # on the last turn of the call
    # added for the receiver when the call is meant to end with it scammed: it confirms the action
    scammed: str


class Speakers(_Strict):
    """How the two speakers are labelled and addressed in the conversation text."""

    caller_tag: str  # how caller lines are labelled in the conversation the agents read
    receiver_tag: str
    empty_call: str  # the conversation text before anyone has spoken
    voice: dict[str, str]  # gender -> pronoun and particle hint
    ending: Ending  # what the speaker is told on the last two turns of the call


class TTS(_Strict):
    """The voices for the call audio and the silence between two turns."""

    voices: dict[str, list[str]]  # gender -> VachanaTTS2 voice ids
    gap_s: float = Field(0.45, ge=0.0)

    @field_validator("voices")
    @classmethod
    def _two_each(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for gender, ids in v.items():
            if len(set(ids)) < 2:  # caller and receiver of the same gender need different voices
                raise ValueError(f"tts.voices.{gender} needs at least 2 voices")
        return v


class Speech(_Strict):
    """How numbers are said in the audio (see spoken.py)."""

    # share of calls whose speakers say "ขีด" between the groups of an account, phone or ID
    # card number; the other calls only pause. One style per call, chosen from its seed.
    say_dash_share: float = Field(0.5, ge=0.0, le=1.0)


class Context(_Strict):
    """The model's context window, shared by every agent (see budget.py)."""

    window_tokens: int = Field(100_000, gt=0)
    safety_margin_tokens: int = Field(2_000, gt=0)  # left free on top of the answer


AGENT_NAMES = (
    "scam_caller",
    "legit_caller",
    "receiver",
    "footing",
    "planner",
    "pattern_extractor",
    "judge",
)


class AgentsFile(_Strict):
    """agents.yaml as a whole."""

    slot_rule: str = ""  # shared text, reused through a YAML anchor; not read by the code
    speakers: Speakers
    scam_caller: ScamCallerSettings
    legit_caller: AgentSettings
    receiver: ReceiverSettings
    footing: AgentSettings
    planner: PlannerSettings
    pattern_extractor: AgentSettings
    judge: JudgeSettings
    tts: TTS
    speech: Speech = Speech()
    context: Context = Context()

    @model_validator(mode="after")
    def _answers_fit_the_window(self) -> AgentsFile:
        for name in AGENT_NAMES:
            max_tokens = getattr(self, name).lm.max_tokens
            if max_tokens + self.context.safety_margin_tokens >= self.context.window_tokens:
                raise ValueError(
                    f"{name}.lm.max_tokens ({max_tokens}) plus context.safety_margin_tokens "
                    f"({self.context.safety_margin_tokens}) must be less than "
                    f"context.window_tokens ({self.context.window_tokens}), or the prompt "
                    "has no room"
                )
        return self


# scenarios.yaml


class Scenario(_Strict):
    """What one kind of call is about. Text may use {SLOTS}."""

    key: str
    role: str  # who the caller says they are
    org: str = ""  # named organisation; "*" = a made-up company, "" = none
    goal: str
    tactics: str  # for legit calls: how the caller handles the call
    # Romance etc.: the prior relationship the receiver believes it is in, from the receiver's
    # side. Given to the receiver so it treats the caller as someone it already knows and trusts,
    # not a cold-call stranger. Empty for ordinary calls.
    receiver_context: str = ""
    # romance: the caller's relation to the receiver, a section of romance_voice.md (lover,
    # friend, ...). A lover is always the other gender from the receiver.
    relation: str = ""
    # romance: the one exception to pity-only pressure; the caller also gets romance_voice.md's
    # `threat` section
    threatens: bool = False
    # the caller's own turn of the first ask, one drawn per call; empty: calls.reveal_turn_slow
    # or reveal_turn_fast by scam type
    reveal_turn: list[Annotated[int, Field(ge=0)]] = []
    # the highest {MONEY} in baht; amounts above it are left out of the draw. None: no cap
    money_max: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _slots(self) -> Scenario:
        for name in ("role", "goal", "tactics"):
            _check_slots(getattr(self, name), f"scenario {self.key}.{name}")
        return self


PROFILE_KEY_RE = re.compile(r"^[a-z][a-z_]*[a-z]$")


class ReceiverProfile(_Strict):
    """Who answers the phone. The receiver agent sees every field as written, except `key` and
    `scam_outcome`, which are control metadata (see turns.receiver_says).

    `key` names a hand-written profile so a run can ask for it (--receiver); profiles built
    from the pools have no key. `scam_outcome` records which way a scam call is meant to end for
    this receiver ("scammed" or "safe"): it is drawn 50/50 per call so the dataset is balanced,
    and the receiver's scam_awareness and temperament are then drawn from the matching group
    (gullible or wary, Receivers.wary_*), so the person — not a forced instruction — leads to the
    outcome. It is kept in the stored record as the label; it is not shown to the model.
    """

    key: str = ""
    gender: str
    age: int = Field(ge=10, le=100)
    occupation: str
    education: str
    region: str  # where they live
    family: str  # who they live with
    digital_literacy: str  # how they use phones, apps and banking
    scam_awareness: str  # what they already know about phone scams
    temperament: str  # how they react to a stranger asking for things
    ending: str  # how they end a call they are unsure about
    speech: str  # how they talk: politeness, dialect words, pace
    # "" for a hand-written profile that does not lock an outcome; sampling sets "scammed"/"safe".
    scam_outcome: str = ""

    @field_validator("key")
    @classmethod
    def _key_shape(cls, v: str) -> str:
        if v and not PROFILE_KEY_RE.match(v):
            raise ValueError("a profile key uses lowercase letters and _ only, e.g. teacher_cm")
        return v

    @field_validator("scam_outcome")
    @classmethod
    def _outcome_value(cls, v: str) -> str:
        if v not in ("", *SCAM_OUTCOMES):
            raise ValueError(f"scam_outcome must be one of {['', *SCAM_OUTCOMES]}, got {v!r}")
        return v


class LifeStage(_Strict):
    """A coherent age bracket: the jobs, schooling, household and phone habits that fit an age.

    Sampling age, occupation, family, education and digital_literacy *together* from one stage
    stops nonsense receivers like a 20-year-old with grown children. Each list is a subset of the
    matching flat pool in Receivers, so the pools stay the one place that lists the vocabulary.
    """

    age: tuple[int, int]
    occupations: list[str] = Field(min_length=1)
    education: list[str] = Field(min_length=1)
    family: list[str] = Field(min_length=1)
    digital_literacy: list[str] = Field(min_length=1)


class Receivers(_Strict):
    """Receivers are either one of `profiles` as written, or built from the pools below.

    When `life_stages` is set, the age-linked fields (age, occupation, education, family,
    digital_literacy) are drawn together from one stage; the other fields are still drawn from
    their flat pools. With no life_stages every field is drawn independently (the old behaviour).
    """

    profile_share: float = Field(0.0, ge=0.0, le=1.0)  # share of calls that use `profiles`
    profiles: list[ReceiverProfile] = []
    life_stages: list[LifeStage] = []
    genders: list[str] = Field(min_length=1)
    age: tuple[int, int]
    occupations: list[str] = Field(min_length=1)
    education: list[str] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    family: list[str] = Field(min_length=1)
    digital_literacy: list[str] = Field(min_length=1)
    scam_awareness: list[str] = Field(min_length=1)
    temperaments: list[str] = Field(min_length=1)
    # The scam_awareness / temperaments entries that make a receiver resist a scam. A call's locked
    # scam_outcome (scenarios.sample_config, 50/50 by seed) draws from these for a "safe" call and
    # from the rest (the "gullible" group) for a "scammed" call, so the person fits the outcome.
    wary_scam_awareness: list[str] = []
    wary_temperaments: list[str] = []
    endings: list[str] = Field(min_length=1)
    speech: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _age_range(self) -> Receivers:
        low, high = self.age
        if not 10 <= low <= high <= 100:
            raise ValueError(
                f"receivers.age must be [low, high] with 10 <= low <= high <= 100, "
                f"got {list(self.age)}"
            )
        return self

    @model_validator(mode="after")
    def _life_stages_fit_the_pools(self) -> Receivers:
        fields = ("occupations", "education", "family", "digital_literacy")
        for i, stage in enumerate(self.life_stages):
            low, high = stage.age
            if not 10 <= low <= high <= 100:
                raise ValueError(
                    f"receivers.life_stages[{i}].age must be [low, high] with "
                    f"10 <= low <= high <= 100, got {list(stage.age)}"
                )
            for field in fields:
                pool = set(getattr(self, field))
                if extra := [v for v in getattr(stage, field) if v not in pool]:
                    raise ValueError(
                        f"receivers.life_stages[{i}].{field} has values not in "
                        f"receivers.{field}: {extra}"
                    )
        return self

    @model_validator(mode="after")
    def _wary_groups_are_subsets(self) -> Receivers:
        # the wary group must be drawn from the pool and leave at least one entry for the gullible
        # group, so both a "safe" and a "scammed" receiver can be built
        for field in ("scam_awareness", "temperaments"):
            pool = getattr(self, field)
            wary = getattr(self, f"wary_{field}")
            if extra := [v for v in wary if v not in pool]:
                raise ValueError(
                    f"receivers.wary_{field} has values not in receivers.{field}: {extra}"
                )
            if wary and len(set(wary)) >= len(set(pool)):
                raise ValueError(
                    f"receivers.wary_{field} must leave at least one entry for the gullible group"
                )
        return self

    @model_validator(mode="after")
    def _profiles_use_known_genders(self) -> Receivers:
        keys = [p.key for p in self.profiles]
        for i, profile in enumerate(self.profiles):
            if profile.gender not in self.genders:
                raise ValueError(f"receivers.profiles[{i}].gender must be one of {self.genders}")
            if not profile.key:
                raise ValueError(f"receivers.profiles[{i}] needs a key, e.g. key: teacher_cm")
            if keys.count(profile.key) > 1:
                raise ValueError(f"receivers.profiles: key {profile.key!r} is used twice")
        if self.profile_share > 0 and not self.profiles:
            raise ValueError("receivers.profile_share > 0 needs at least one profile")
        return self


class Calls(_Strict):
    """How long calls run and when a scam caller makes its first ask."""

    slow_types: list[str]  # scam types that build trust before asking
    reveal_turn_slow: list[int] = Field(min_length=1)  # caller's own turn of the first ask
    reveal_turn_fast: list[int] = Field(min_length=1)
    max_turns: tuple[int, int]  # both sides together
    caller_knows_name: float = Field(ge=0.0, le=1.0)
    min_turns: int = Field(4, ge=2)  # shorter calls are rejected


def shortest_call(calls: Calls, reveal: int) -> int:
    """Fewest speaker turns a call with this reveal turn can get (see scenarios.sample_config):
    room for the reveal plus a few exchanges, within the configured length."""
    low, high = calls.max_turns
    return min(high, max(low, 2 * reveal + 4))


class ScenariosFile(_Strict):
    """scenarios.yaml as a whole."""

    scams: dict[str, list[Scenario]]
    legit: list[Scenario] = Field(min_length=1)
    receivers: Receivers
    calls: Calls

    @field_validator("scams")
    @classmethod
    def _all_scam_types(cls, scams: dict[str, list[Scenario]]) -> dict[str, list[Scenario]]:
        if set(scams) != set(SCAM_TYPES):
            raise ValueError(f"scams needs exactly the keys {list(SCAM_TYPES)}")
        empty = [call_type for call_type, scenarios in scams.items() if not scenarios]
        if empty:
            raise ValueError(f"scams.{empty[0]} has no scenarios")
        return scams


# details.yaml


class Money(_Strict):
    """The amounts a caller can ask for."""

    big_types: list[str]  # call types that use the big amounts
    big: list[int] = Field(min_length=1)
    small: list[int] = Field(min_length=1)


class Urls(_Strict):
    """The pieces of a made-up link."""

    words: list[str] = Field(min_length=1)
    tlds: list[str] = Field(min_length=1)


DEFAULT_LINE_ID_PARTS = ["host", "vip", "land", "doc", "care", "help", "gold", "team"]


class AddressParts(_Strict):
    """Pieces of the receiver's made-up {ADDRESS}: <house> ซอย<soi> <district> <province>."""

    house_numbers: list[str] = Field(min_length=1)
    sois: list[str] = Field(min_length=1)
    districts: list[str] = Field(min_length=1)
    provinces: list[str] = Field(min_length=1)


DEFAULT_ADDRESS_PARTS = AddressParts(
    house_numbers=["12/3", "45", "88/21", "107", "9/14", "233", "56/7"],
    sois=["สุขสันต์ 4", "ร่มเย็น 12", "ประชาสุข 7", "เจริญใจ 2", "บ้านสวน 15", "ทรัพย์มั่งมี 9"],
    districts=["อำเภอบ้านใหม่", "อำเภอท่าช้าง", "อำเภอหนองแสง", "อำเภอเมืองเก่า", "อำเภอวังทอง"],
    provinces=[
        "จังหวัดขอนแก่น",
        "จังหวัดเชียงใหม่",
        "จังหวัดนครราชสีมา",
        "จังหวัดสงขลา",
        "จังหวัดชลบุรี",
        "จังหวัดพิษณุโลก",
    ],
)


class DetailsFile(_Strict):
    """details.yaml as a whole."""

    names: dict[str, list[str]]  # gender -> first names
    nicknames: dict[str, list[str]] = {}  # gender -> nicknames; older config folders have none
    banks: list[str] = Field(min_length=1)
    made_up_orgs: list[str] = Field(min_length=1)
    podcasts: list[str] = []  # older config folders have none; made_up_orgs is used instead
    schools: list[str] = []  # older config folders have none; made_up_orgs is used instead
    police_stations: list[str] = []  # older config folders have none; made_up_orgs is used
    urls: Urls
    # parts of a made-up {LINE_ID}; older config folders have none
    line_ids: list[str] = Field(default=DEFAULT_LINE_ID_PARTS, min_length=2)
    money: Money
    addresses: AddressParts = DEFAULT_ADDRESS_PARTS  # older config folders have none


# the whole config


@dataclass(eq=False)
class Config:
    """The three checked files, the folder they came from, and a hash of their parsed data."""

    folder: Path
    # agents.yaml: each agent's settings. The programs themselves are agents.Agents.
    agents: AgentsFile
    scenarios: ScenariosFile
    details: DetailsFile
    hash: str
    romance_voice: dict[str, str]  # romance_voice.md: section name -> text (romance_voice.py)
    red_flags: str  # scam warning signs every receiver reads, from the wiki (red_flags.py)

    @cached_property
    def name(self) -> str:
        return self.folder.name


def _read(folder: Path, name: str) -> dict:
    """One YAML file as a dict, or a ConfigError that names the file."""
    path = folder / name
    if not path.exists():
        raise ConfigError(f"missing {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _cross_check(agents: AgentsFile, scenarios: ScenariosFile, details: DetailsFile) -> None:
    """Checks that span files: every gender, call type and turn count has to agree."""
    genders = set(scenarios.receivers.genders)
    for where, keys in (
        ("agents.yaml speakers.voice", agents.speakers.voice),
        ("agents.yaml tts.voices", agents.tts.voices),
        ("details.yaml names", details.names),
    ):
        if missing := genders - set(keys):
            raise ConfigError(f"{where} has no entry for gender(s) {sorted(missing)}")
    for gender in genders:
        if len(set(details.names[gender])) < 2:  # the caller and receiver need different names
            raise ConfigError(f"details.yaml names.{gender} needs at least 2 different names")
        if details.nicknames and len(set(details.nicknames.get(gender, []))) < 2:
            raise ConfigError(f"details.yaml nicknames.{gender} needs at least 2 different names")
    unknown = set(scenarios.calls.slow_types) - set(SCAM_TYPES)
    unknown |= set(details.money.big_types) - set(CALL_TYPES)
    if unknown:
        raise ConfigError(f"unknown call type(s) {sorted(unknown)}")
    for call_type, group in scenarios.scams.items():
        amounts = details.money.big if call_type in details.money.big_types else details.money.small
        for scenario in group:
            if scenario.money_max is not None and min(amounts) > scenario.money_max:
                raise ConfigError(
                    f"scenarios.yaml scenario {scenario.key}.money_max is {scenario.money_max}, "
                    f"below every details.yaml money amount for {call_type} "
                    f"({min(amounts)} is the lowest)"
                )
    low, high = scenarios.calls.max_turns
    if not (scenarios.calls.min_turns <= low <= high):
        raise ConfigError(
            "scenarios.yaml calls.max_turns must be [low, high] with low >= min_turns"
        )
    # The caller's last own turn is (turns - 1) // 2 (0-based) and pacing.last makes it ask on
    # the turn before that, so the ask has to come at (turns - 1) // 2 - 1 at the latest, in the
    # shortest call that reveal turn can get. Longer reveals make longer calls (shortest_call),
    # so this only fails when max_turns[1] is too small for the reveal turn.
    reveal_lists = {
        f"calls.{name}": getattr(scenarios.calls, name)
        for name in ("reveal_turn_slow", "reveal_turn_fast")
    }
    for scenario in (s for group in scenarios.scams.values() for s in group):
        reveal_lists[f"scenario {scenario.key}.reveal_turn"] = scenario.reveal_turn
    for name, reveals in reveal_lists.items():
        for reveal in reveals:
            turns = shortest_call(scenarios.calls, reveal)
            latest = (turns - 1) // 2 - 1
            if reveal > latest:
                raise ConfigError(
                    f"scenarios.yaml {name} has {reveal}, but the shortest call it can get "
                    f"({turns} turns) leaves the caller no turn to ask after it "
                    f"(latest reveal turn there: {latest}); raise max_turns or lower it"
                )


def load_config(folder: Path | str | None = None) -> Config:
    """Read and check the three YAML files in `folder` (default: data_pipeline/config).

    Raises:
        ConfigError: a file is missing or not valid; the message says what to fix.
    """
    folder = Path(folder) if folder else DEFAULT_DIR
    raw = {name: _read(folder, name) for name in FILES}
    try:
        agents = AgentsFile.model_validate(raw["agents.yaml"])
        scenarios = ScenariosFile.model_validate(raw["scenarios.yaml"])
        details = DetailsFile.model_validate(raw["details.yaml"])
    except ValidationError as exc:
        raise ConfigError(f"config in {folder} is not valid:\n{exc}") from exc
    _cross_check(agents, scenarios, details)
    try:
        romance_voice = romance_voice_page.load()
        romance_voice_page.check(romance_voice, scenarios)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    # The receiver's scam warning signs, from the wiki. Advisory, so a wiki problem gives "".
    red_flags = red_flags_page.digest()
    # Hash the parsed content, not the bytes, so a comment or spacing change keeps the same hash
    # and a stopped run still resumes. red_flags is in the hash too, so editing the wiki keeps new
    # data apart from old.
    parsed = [raw[n] for n in FILES] + [romance_voice, red_flags]
    digest = hashlib.sha256(
        yaml.safe_dump(parsed, sort_keys=True, allow_unicode=True).encode()
    ).hexdigest()[:12]
    return Config(folder, agents, scenarios, details, digest, romance_voice, red_flags)
