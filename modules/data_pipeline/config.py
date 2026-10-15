"""Load and check the simulator config: three YAML files in one folder.

    agents.yaml     what each agent is told, its model settings, pacing text, TTS voices
    scenarios.yaml  what calls are about, who answers, how long calls run
    details.yaml    names, banks, links and amounts that fill the slots

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

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from nlp_core.types import SCAM_TYPES

DEFAULT_DIR = Path(__file__).resolve().parent / "config"
FILES = ("agents.yaml", "scenarios.yaml", "details.yaml")
BENIGN = "benign"
CALL_TYPES: tuple[str, ...] = (*SCAM_TYPES, BENIGN)
SLOT_NAMES = (
    "CALLER_NAME",
    "RECEIVER_NAME",
    "ORG",
    "PHONE",
    "BANK_ACCOUNT",
    "OTP",
    "URL",
    "MONEY",
)
_SLOT_IN_TEXT = re.compile(r"\{([A-Z_]+)\}")


class ConfigError(ValueError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_slots(text: str, where: str) -> None:
    bad = sorted(set(_SLOT_IN_TEXT.findall(text)) - set(SLOT_NAMES))
    if bad:
        raise ValueError(f"{where}: unknown slot(s) {bad}; known slots are {list(SLOT_NAMES)}")


# ---- agents.yaml ---------------------------------------------------------------------------


class LMSettings(_Strict):
    model: str | None = None  # empty: VLLM_MODEL from .env (or the first served model)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1000, ge=64)


class Agent(_Strict):
    instructions: str
    fields: dict[str, str]  # field name -> description shown to the model
    lm: LMSettings = LMSettings()


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


class ScamCaller(Agent):
    pacing: Pacing


class ReceiverStage(_Strict):
    stay: str  # while it must stay on the line; {turn} is filled in
    free: str

    @model_validator(mode="after")
    def _formattable(self) -> ReceiverStage:
        # simulate.stage passes only `turn`, so {turns} would fail there
        for name in ("stay", "free"):
            _check_format(getattr(self, name), f"receiver.stage.{name}", turn=1)
        return self


class Receiver(Agent):
    stay_on_line: int = Field(ge=0)
    stage: ReceiverStage


class Planner(Agent):
    notes_limit: int = Field(6, ge=0)  # newest pattern notes the planner reads per call


class Judge(Agent):
    intended: dict[str, str]  # call type -> what the call is meant to show

    @field_validator("intended")
    @classmethod
    def _all_types(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v) != set(CALL_TYPES):
            raise ValueError(f"intended needs exactly the keys {list(CALL_TYPES)}")
        return v


class Speakers(_Strict):
    caller_tag: str  # how caller lines are labelled in the conversation the agents read
    receiver_tag: str
    empty_call: str  # the conversation text before anyone has spoken
    voice: dict[str, str]  # gender -> pronoun and particle hint


class TTS(_Strict):
    voices: dict[str, list[str]]  # gender -> VachanaTTS2 voice ids
    gap_s: float = Field(0.45, ge=0.0)

    @field_validator("voices")
    @classmethod
    def _two_each(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        for gender, ids in v.items():
            if len(set(ids)) < 2:  # caller and receiver of the same gender need different voices
                raise ValueError(f"tts.voices.{gender} needs at least 2 voices")
        return v


class AgentsFile(_Strict):
    slot_rule: str = ""  # shared text, reused through a YAML anchor; not read by the code
    speakers: Speakers
    scam_caller: ScamCaller
    legit_caller: Agent
    receiver: Receiver
    planner: Planner
    pattern_extractor: Agent
    judge: Judge
    tts: TTS


# ---- scenarios.yaml ------------------------------------------------------------------------


class Scenario(_Strict):
    key: str
    role: str  # who the caller says they are
    org: str = ""  # named organisation; "*" = a made-up company, "" = none
    goal: str
    tactics: str  # for legit calls: how the caller handles the call

    @model_validator(mode="after")
    def _slots(self) -> Scenario:
        for name in ("role", "goal", "tactics"):
            _check_slots(getattr(self, name), f"scenario {self.key}.{name}")
        return self


PROFILE_KEY_RE = re.compile(r"^[a-z][a-z_]*[a-z]$")


class ReceiverProfile(_Strict):
    """Who answers the phone. The receiver agent sees every field, as written.

    `key` names a hand-written profile so a run can ask for it (--receiver); profiles built
    from the pools have no key.
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

    @field_validator("key")
    @classmethod
    def _key_shape(cls, v: str) -> str:
        if v and not PROFILE_KEY_RE.match(v):
            raise ValueError("a profile key uses lowercase letters and _ only, e.g. teacher_cm")
        return v


class Receivers(_Strict):
    """Receivers are either one of `profiles` as written, or built from the pools below."""

    profile_share: float = Field(0.0, ge=0.0, le=1.0)  # share of calls that use `profiles`
    profiles: list[ReceiverProfile] = []
    genders: list[str] = Field(min_length=1)
    age: tuple[int, int]
    occupations: list[str] = Field(min_length=1)
    education: list[str] = Field(min_length=1)
    regions: list[str] = Field(min_length=1)
    family: list[str] = Field(min_length=1)
    digital_literacy: list[str] = Field(min_length=1)
    scam_awareness: list[str] = Field(min_length=1)
    temperaments: list[str] = Field(min_length=1)
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


# ---- details.yaml --------------------------------------------------------------------------


class Money(_Strict):
    big_types: list[str]  # call types that use the big amounts
    big: list[int] = Field(min_length=1)
    small: list[int] = Field(min_length=1)


class Urls(_Strict):
    words: list[str] = Field(min_length=1)
    tlds: list[str] = Field(min_length=1)


class DetailsFile(_Strict):
    names: dict[str, list[str]]  # gender -> first names
    banks: list[str] = Field(min_length=1)
    made_up_orgs: list[str] = Field(min_length=1)
    urls: Urls
    money: Money


# ---- the whole config ----------------------------------------------------------------------


@dataclass(eq=False)
class Config:
    """The three checked files, the folder they came from, and a hash of their parsed data."""

    folder: Path
    agents: AgentsFile
    scenarios: ScenariosFile
    details: DetailsFile
    hash: str

    @cached_property
    def name(self) -> str:
        return self.folder.name


def _read(folder: Path, name: str) -> dict:
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
    unknown = set(scenarios.calls.slow_types) - set(SCAM_TYPES)
    unknown |= set(details.money.big_types) - set(CALL_TYPES)
    if unknown:
        raise ConfigError(f"unknown call type(s) {sorted(unknown)}")
    low, high = scenarios.calls.max_turns
    if not (scenarios.calls.min_turns <= low <= high):
        raise ConfigError(
            "scenarios.yaml calls.max_turns must be [low, high] with low >= min_turns"
        )
    # The caller's last own turn is (turns - 1) // 2 (0-based) and pacing.last makes it ask on
    # the turn before that, so the ask has to come at (turns - 1) // 2 - 1 at the latest, in the
    # shortest call that reveal turn can get. Longer reveals make longer calls (shortest_call),
    # so this only fails when max_turns[1] is too small for the reveal turn.
    for name in ("reveal_turn_slow", "reveal_turn_fast"):
        for reveal in getattr(scenarios.calls, name):
            turns = shortest_call(scenarios.calls, reveal)
            latest = (turns - 1) // 2 - 1
            if reveal > latest:
                raise ConfigError(
                    f"scenarios.yaml calls.{name} has {reveal}, but the shortest call it can get "
                    f"({turns} turns) leaves the caller no turn to ask after it "
                    f"(latest reveal turn there: {latest}); raise max_turns or lower it"
                )


def load_config(folder: Path | str | None = None) -> Config:
    folder = Path(folder) if folder else DEFAULT_DIR
    raw = {name: _read(folder, name) for name in FILES}
    try:
        agents = AgentsFile.model_validate(raw["agents.yaml"])
        scenarios = ScenariosFile.model_validate(raw["scenarios.yaml"])
        details = DetailsFile.model_validate(raw["details.yaml"])
    except ValidationError as exc:
        raise ConfigError(f"config in {folder} is not valid:\n{exc}") from exc
    _cross_check(agents, scenarios, details)
    # Hash the parsed content, not the bytes: comments and spacing do not change the data.
    digest = hashlib.sha256(
        yaml.safe_dump([raw[n] for n in FILES], sort_keys=True, allow_unicode=True).encode()
    ).hexdigest()[:12]
    return Config(folder, agents, scenarios, details, digest)
