"""Sample what one simulated call is about, from a seed, before any LLM call.

The scenarios, receivers and call lengths live in config/scenarios.yaml. The label of a call
comes from here and never from the model: a scam call is generated from a scam scenario of one
of the 5 spec types, a normal call from a legit scenario. The receiver is chosen the same way for
both, so how they react says nothing about the label.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .config import (
    BENIGN,
    CALL_TYPES,
    Config,
    ReceiverProfile,
    Receivers,
    Scenario,
    shortest_call,
)

__all__ = ["BENIGN", "CALL_TYPES", "CallConfig", "ReceiverProfile", "sample_config"]


@dataclass(frozen=True)
class CallConfig:
    """Everything that decides one call. Same (call_type, seed, config) gives the same call."""

    call_type: str  # one of CALL_TYPES
    seed: int
    scenario: Scenario
    receiver: ReceiverProfile
    caller_gender: str
    caller_knows_name: bool
    max_turns: int  # speaker turns in total, both sides
    reveal_turn: int  # scam calls: the caller's earliest own turn (0-based) to ask

    @property
    def label_scam(self) -> bool:
        """Whether this is a scam call. Decided by the call type, never by a model."""
        return self.call_type != BENIGN

    @property
    def label_type(self) -> str | None:
        """The scam type of a scam call, None for a normal call."""
        return self.call_type if self.label_scam else None

    def to_dict(self) -> dict:
        """The call as plain JSON-ready dicts, as stored in generation.call."""
        data = asdict(self)
        data["scenario"] = self.scenario.model_dump()
        data["receiver"] = self.receiver.model_dump()
        return data


def sample_config(
    call_type: str, seed: int, config: Config, receiver: ReceiverProfile | None = None
) -> CallConfig:
    """The call for this type and seed. `receiver` replaces the sampled receiver (a person
    chosen by hand); everything else is sampled exactly as without it."""
    if call_type not in CALL_TYPES:
        raise ValueError(f"call type must be one of {CALL_TYPES}, got {call_type!r}")
    scenarios = config.scenarios
    rng = random.Random(f"{call_type}:{seed}")
    scenario = rng.choice(scenarios.legit if call_type == BENIGN else scenarios.scams[call_type])
    pool = scenarios.receivers
    # Lock this call's scam outcome 50/50 by seed parity, so the dataset is balanced between
    # receivers who get scammed and those who stay safe. From the seed, not rng, so it does not
    # shift any other sampled value; sample_receiver then draws a persona that fits it.
    outcome = "scammed" if seed % 2 == 0 else "safe"
    sampled = sample_receiver(
        rng, pool, outcome
    )  # always drawn, so the rest of the call is the same
    receiver = receiver or sampled
    calls = scenarios.calls
    slow = call_type in calls.slow_types
    reveals = scenario.reveal_turn or (calls.reveal_turn_slow if slow else calls.reveal_turn_fast)
    reveal = rng.choice(reveals)
    high = calls.max_turns[1]
    # The keyword arguments below draw from rng in the order written. Do not reorder them: the
    # order fixes every call for a seed.
    return CallConfig(
        call_type=call_type,
        seed=seed,
        scenario=scenario,
        receiver=receiver,
        caller_gender=_caller_gender(rng, pool.genders, scenario, receiver),
        caller_knows_name=rng.random() < calls.caller_knows_name,
        # room for the reveal turn plus a few exchanges after it
        max_turns=rng.randint(shortest_call(calls, reveal), high),
        reveal_turn=reveal,
    )


def _caller_gender(
    rng: random.Random, genders: list[str], scenario: Scenario, receiver: ReceiverProfile
) -> str:
    """A random gender, except that a lover is the other gender from the receiver."""
    drawn = rng.choice(genders)  # always drawn, so the rest of the call is the same
    others = [g for g in genders if g != receiver.gender]
    return others[0] if scenario.relation == "lover" and others else drawn


def _outcome_pools(pool: Receivers, outcome: str) -> tuple[list[str], list[str]]:
    """The scam_awareness and temperament lists to draw from for a locked `outcome`: the wary group
    for "safe", the rest (gullible) for "scammed", so the person leads to the outcome. Falls back to
    the full pools when no outcome is locked or no wary group is configured."""

    def group(full: list[str], wary: list[str]) -> list[str]:
        if not wary or outcome not in ("scammed", "safe"):
            return full
        wary_set = set(wary)
        gullible = [v for v in full if v not in wary_set]
        return wary if outcome == "safe" else (gullible or full)

    return (
        group(pool.scam_awareness, pool.wary_scam_awareness),
        group(pool.temperaments, pool.wary_temperaments),
    )


def sample_receiver(rng: random.Random, pool: Receivers, outcome: str = "") -> ReceiverProfile:
    """One of the hand-written profiles (profile_share of the time), else one built from pools.

    When the config has life_stages, the age-linked fields (age, occupation, education, family,
    digital_literacy) are drawn together from one stage so the person is believable; without them
    every field is drawn independently (the old behaviour).

    `outcome` ("scammed"/"safe") is the call's locked result; scam_awareness and temperament are
    drawn from the group that fits it (Receivers.wary_*), and it is stamped on the receiver.

    A hand-written profile is a fixed persona with its own nature, so it is left unlocked (its own
    scam_outcome, usually ""): the 50/50 lock is not forced onto it, only onto pool-built receivers.
    """
    if pool.profiles and rng.random() < pool.profile_share:
        return rng.choice(pool.profiles)
    stage = rng.choice(pool.life_stages) if pool.life_stages else None
    age_range = stage.age if stage else pool.age
    occupations = stage.occupations if stage else pool.occupations
    education = stage.education if stage else pool.education
    family = stage.family if stage else pool.family
    digital_literacy = stage.digital_literacy if stage else pool.digital_literacy
    awareness_pool, temperament_pool = _outcome_pools(pool, outcome)
    return ReceiverProfile(
        gender=rng.choice(pool.genders),
        age=rng.randint(*age_range),
        occupation=rng.choice(occupations),
        education=rng.choice(education),
        region=rng.choice(pool.regions),
        family=rng.choice(family),
        digital_literacy=rng.choice(digital_literacy),
        scam_awareness=rng.choice(awareness_pool),
        temperament=rng.choice(temperament_pool),
        ending=rng.choice(pool.endings),
        speech=rng.choice(pool.speech),
        scam_outcome=outcome,
    )
