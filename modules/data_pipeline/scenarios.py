"""Sample what one simulated call is about, from a seed, before any LLM call.

The scenarios, receivers and pacing numbers live in config/scenarios.yaml. The label of a call
comes from here and never from the model: a scam call is generated from a scam scenario of one
of the 5 spec types, a normal call from a legit scenario. The receiver is chosen the same way for
both, so how they react says nothing about the label.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .config import BENIGN, CALL_TYPES, Config, ReceiverProfile, Receivers, Scenario

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
        return self.call_type != BENIGN

    @property
    def label_type(self) -> str | None:
        return self.call_type if self.label_scam else None

    def to_dict(self) -> dict:
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
    sampled = sample_receiver(rng, pool)  # always drawn, so the rest of the call is the same
    receiver = receiver or sampled
    calls = scenarios.calls
    slow = call_type in calls.slow_types
    reveal = rng.choice(calls.reveal_turn_slow if slow else calls.reveal_turn_fast)
    low, high = calls.max_turns
    return CallConfig(
        call_type=call_type,
        seed=seed,
        scenario=scenario,
        receiver=receiver,
        caller_gender=rng.choice(pool.genders),
        caller_knows_name=rng.random() < calls.caller_knows_name,
        # room for the reveal turn plus a few exchanges after it
        max_turns=rng.randint(min(high, max(low, 2 * reveal + 4)), high),
        reveal_turn=reveal,
    )


def sample_receiver(rng: random.Random, pool: Receivers) -> ReceiverProfile:
    """One of the hand-written profiles (profile_share of the time), else one built from pools."""
    if pool.profiles and rng.random() < pool.profile_share:
        return rng.choice(pool.profiles)
    return ReceiverProfile(
        gender=rng.choice(pool.genders),
        age=rng.randint(*pool.age),
        occupation=rng.choice(pool.occupations),
        education=rng.choice(pool.education),
        region=rng.choice(pool.regions),
        family=rng.choice(pool.family),
        digital_literacy=rng.choice(pool.digital_literacy),
        scam_awareness=rng.choice(pool.scam_awareness),
        temperament=rng.choice(pool.temperaments),
        ending=rng.choice(pool.endings),
        speech=rng.choice(pool.speech),
    )
