"""The DSPy agents that act out a call, and the judge that checks the finished call.

The classes below fix each agent's field names and types: that is the contract the simulator
reads (`line`, `asks_now`, `end_call`, the judge's verdict fields). Everything the model is
told, the instructions and each field's description, comes from config/agents.yaml, so the
team can tune an agent without touching code.

Each agent is a ChainOfThought program. Its `reasoning` is kept in the call log for review but
never goes into the call text the classifier trains on. The receiver is never told the label,
so its reactions cannot give the label away.
"""

from __future__ import annotations

from typing import Any

import dspy

from .config import Agent, Config, ConfigError

# Docstrings are placeholders: the real instructions are set from agents.yaml.


class ScamCallerTurn(dspy.Signature):
    """(set from agents.yaml: scam_caller.instructions)"""

    role: str = dspy.InputField()
    voice: str = dspy.InputField()
    org: str = dspy.InputField()
    goal: str = dspy.InputField()
    tactics: str = dspy.InputField()
    receiver_name: str = dspy.InputField()
    pacing: str = dspy.InputField()
    plan: list[str] = dspy.InputField()
    conversation: str = dspy.InputField()
    line: str = dspy.OutputField()
    asks_now: bool = dspy.OutputField()
    end_call: bool = dspy.OutputField()


class LegitCallerTurn(dspy.Signature):
    """(set from agents.yaml: legit_caller.instructions)"""

    role: str = dspy.InputField()
    voice: str = dspy.InputField()
    org: str = dspy.InputField()
    goal: str = dspy.InputField()
    manner: str = dspy.InputField()
    receiver_name: str = dspy.InputField()
    conversation: str = dspy.InputField()
    line: str = dspy.OutputField()
    end_call: bool = dspy.OutputField()


class ReceiverTurn(dspy.Signature):
    """(set from agents.yaml: receiver.instructions)"""

    persona: dict = dspy.InputField()  # a ReceiverProfile as a dict, without its key
    voice: str = dspy.InputField()
    stage: str = dspy.InputField()
    conversation: str = dspy.InputField()
    line: str = dspy.OutputField()
    end_call: bool = dspy.OutputField()


class PlanScamCall(dspy.Signature):
    """(set from agents.yaml: planner.instructions)"""

    scam_type: str = dspy.InputField()
    role: str = dspy.InputField()
    goal: str = dspy.InputField()
    tactics: str = dspy.InputField()
    pattern_notes: str = dspy.InputField()
    turns: str = dspy.InputField()
    plan: list[str] = dspy.OutputField()


class ExtractScamPattern(dspy.Signature):
    """(set from agents.yaml: pattern_extractor.instructions)"""

    scam_type: str = dspy.InputField()
    article: str = dspy.InputField()
    relevant: bool = dspy.OutputField()
    summary: str = dspy.OutputField()
    tactics: list[str] = dspy.OutputField()
    phrases: list[str] = dspy.OutputField()


class JudgeCall(dspy.Signature):
    """(set from agents.yaml: judge.instructions)"""

    intended: str = dspy.InputField()
    transcript: str = dspy.InputField()
    label_holds: bool = dspy.OutputField()
    scam_start_turn: int = dspy.OutputField()
    natural_thai: int = dspy.OutputField()
    giveaway: bool = dspy.OutputField()
    summary: str = dspy.OutputField()
    issues: str = dspy.OutputField()


SIGNATURES: dict[str, type[dspy.Signature]] = {
    "scam_caller": ScamCallerTurn,
    "legit_caller": LegitCallerTurn,
    "receiver": ReceiverTurn,
    "planner": PlanScamCall,
    "pattern_extractor": ExtractScamPattern,
    "judge": JudgeCall,
}


def configured(name: str, agent: Agent) -> type[dspy.Signature]:
    """The signature for `name` with its instructions and field descriptions from config."""
    sig = SIGNATURES[name]
    fields = set(sig.input_fields) | set(sig.output_fields)
    if missing := fields - set(agent.fields):
        raise ConfigError(f"agents.yaml {name}.fields is missing {sorted(missing)}")
    if extra := set(agent.fields) - fields:
        raise ConfigError(
            f"agents.yaml {name}.fields has unknown {sorted(extra)}; "
            f"the fields are fixed in code: {sorted(fields)}"
        )
    sig = sig.with_instructions(agent.instructions.strip())
    for field, desc in agent.fields.items():
        sig = sig.with_updated_fields(field, desc=desc.strip())
    return sig


def build_lms(config: Config) -> dict[str, Any]:
    """One dspy.LM per agent, from its lm settings in agents.yaml."""
    from llm_serving.dspy_programs import build_lm

    lms = {}
    for name in SIGNATURES:
        lm = getattr(config.agents, name).lm
        lms[name] = build_lm(model=lm.model, max_tokens=lm.max_tokens, temperature=lm.temperature)
    return lms


class Agents:
    """Every program and its model. Plain attributes so tests can swap any of them."""

    def __init__(self, config: Config, lms: dict[str, Any] | None = None) -> None:
        # the signature each program really sends (with ChainOfThought's reasoning field), kept
        # apart so the prompt can be measured even when a test swaps a program for a stand-in
        self.signatures: dict[str, type[dspy.Signature]] = {}
        for name in SIGNATURES:
            program = dspy.ChainOfThought(configured(name, getattr(config.agents, name)))
            setattr(self, name, program)
            self.signatures[name] = program.predict.signature
        self.lms = lms if lms is not None else build_lms(config)

    def models(self) -> dict[str, str]:
        """Model name per agent, for the record."""
        return {name: getattr(lm, "model", str(lm)) for name, lm in self.lms.items()}
