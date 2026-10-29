"""The DSPy agents that act out a call, and the judge that checks the finished call.

The classes below fix each agent's field names and types: that is the contract the simulator
reads (`line`, `asks_now`, `end_call`, the judge's verdict fields). Everything the model is
told, the instructions and each field's description, comes from config/agents.yaml, so the
team can tune an agent without touching code.

Each agent is a ChainOfThought program. Its `reasoning` is kept in the call log for review but
never goes into the call text the classifier trains on. The receiver is never told the label,
so its reactions cannot give the label away.

`Agents` holds the six programs and the dspy LM of each. `config.agents` is agents.yaml, with
one `AgentSettings` per agent (its `lm` is the model settings, not a dspy LM). The scam caller
is one program per scam type (`ScamCaller`), each with that type's instructions.
"""

from __future__ import annotations

from typing import Any

import dspy

from nlp_core.types import SCAM_TYPES

from .config import AgentSettings, Config, ConfigError, ScamCallerSettings

# Each docstring below is the agent's job in one line. `configured` swaps it for the full
# instructions in agents.yaml, found under the key named in the comment.


class ScamCallerTurn(dspy.Signature):
    """Play the scam caller and write its next line."""

    # Full instructions: agents.yaml, scam_caller.instructions + scam_caller.by_type.<scam type>

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
    """Play an ordinary, honest caller and write its next line."""

    # Full instructions: agents.yaml, legit_caller.instructions

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
    """Play the person who answered the phone and write their next line."""

    # Full instructions: agents.yaml, receiver.instructions

    persona: dict = dspy.InputField()  # a ReceiverProfile as a dict, without its key
    voice: str = dspy.InputField()
    stage: str = dspy.InputField()
    red_flags: str = dspy.InputField()  # general scam warning signs, same for every call
    conversation: str = dspy.InputField()
    line: str = dspy.OutputField()
    end_call: bool = dspy.OutputField()


class PlanScamCall(dspy.Signature):
    """Plan a scam call, one step for each caller turn."""

    # Full instructions: agents.yaml, planner.instructions

    scam_type: str = dspy.InputField()
    role: str = dspy.InputField()
    goal: str = dspy.InputField()
    tactics: str = dspy.InputField()
    pattern_notes: str = dspy.InputField()
    turns: str = dspy.InputField()
    plan: list[str] = dspy.OutputField()


class ExtractScamPattern(dspy.Signature):
    """Read a scam report and write down how the scam works."""

    # Full instructions: agents.yaml, pattern_extractor.instructions

    scam_type: str = dspy.InputField()
    article: str = dspy.InputField()
    relevant: bool = dspy.OutputField()
    summary: str = dspy.OutputField()
    tactics: list[str] = dspy.OutputField()
    phrases: list[str] = dspy.OutputField()


class JudgeCall(dspy.Signature):
    """Check a simulated call before it goes into the dataset."""

    # Full instructions: agents.yaml, judge.instructions

    intended: str = dspy.InputField()
    transcript: str = dspy.InputField()
    realism_criteria: list[str] = dspy.InputField()
    scenario: dict = dspy.InputField()  # the selected scenario's role, org, goal, tactics, context
    persona: dict = dspy.InputField()  # the selected ReceiverProfile as a dict, without its key
    scam_clear: bool = dspy.OutputField()
    scam_clarity_reason: str = dspy.OutputField()
    label_holds: bool = dspy.OutputField()
    label_reason: str = dspy.OutputField()
    language_realistic: bool = dspy.OutputField()
    language_realism_reason: str = dspy.OutputField()
    setup_holds: bool = dspy.OutputField()
    setup_reason: str = dspy.OutputField()
    scam_start_turn: int = dspy.OutputField()
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


def configured(
    name: str, settings: AgentSettings, instructions: str | None = None
) -> type[dspy.Signature]:
    """The signature for `name` with its instructions and field descriptions from config.

    `instructions` replaces settings.instructions (the scam caller's text for one scam type).
    """
    signature = SIGNATURES[name]
    fields = set(signature.input_fields) | set(signature.output_fields)
    if missing := fields - set(settings.fields):
        raise ConfigError(f"agents.yaml {name}.fields is missing {sorted(missing)}")
    if extra := set(settings.fields) - fields:
        raise ConfigError(
            f"agents.yaml {name}.fields has unknown {sorted(extra)}; "
            f"the fields are fixed in code: {sorted(fields)}"
        )
    signature = signature.with_instructions((instructions or settings.instructions).strip())
    for field, desc in settings.fields.items():
        signature = signature.with_updated_fields(field, desc=desc.strip())
    return signature


def build_lms(config: Config) -> dict[str, Any]:
    """One dspy.LM per agent, from its lm settings in agents.yaml."""
    from llm_serving.dspy_programs import build_lm

    lms = {}
    for name in SIGNATURES:
        lm = getattr(config.agents, name).lm  # the LMSettings in agents.yaml, not a dspy.LM
        lms[name] = build_lm(model=lm.model, max_tokens=lm.max_tokens, temperature=lm.temperature)
    return lms


class Agents:
    """The six DSPy programs and the LM each one runs on.

    Plain attributes so tests can swap any of them. Each program's settings are in
    `config.agents` (agents.yaml), one `AgentSettings` per agent.
    """

    scam_caller: dspy.Module
    legit_caller: dspy.Module
    receiver: dspy.Module
    planner: dspy.Module
    pattern_extractor: dspy.Module
    judge: dspy.Module

    def __init__(self, config: Config, lms: dict[str, Any] | None = None) -> None:
        settings = config.agents
        self.scam_caller = ScamCaller(settings.scam_caller)
        self.legit_caller = _program("legit_caller", settings.legit_caller)
        self.receiver = _program("receiver", settings.receiver)
        self.planner = _program("planner", settings.planner)
        self.pattern_extractor = _program("pattern_extractor", settings.pattern_extractor)
        self.judge = _program("judge", settings.judge)
        # the signature each program really sends (with ChainOfThought's reasoning field), kept
        # apart so the prompt can be measured even when a test swaps a program for a stand-in.
        # The scam caller has one per scam type, in scam_caller_signatures.
        self.signatures: dict[str, type[dspy.Signature]] = {
            name: getattr(self, name).predict.signature
            for name in SIGNATURES
            if name != "scam_caller"
        }
        self.scam_caller_signatures: dict[str, type[dspy.Signature]] = {
            scam_type: program.predict.signature
            for scam_type, program in self.scam_caller.programs.items()
        }
        self.lms = lms if lms is not None else build_lms(config)

    def signature(self, agent: str, scam_type: str | None = None) -> type[dspy.Signature]:
        """The signature `agent` sends; the scam caller's depends on the scam type."""
        if agent == "scam_caller":
            return self.scam_caller_signatures[scam_type]
        return self.signatures[agent]

    def models(self) -> dict[str, str]:
        """Model name per agent, for the record."""
        return {name: getattr(lm, "model", str(lm)) for name, lm in self.lms.items()}


class ScamCaller(dspy.Module):
    """The scam caller: one ChainOfThought program per scam type, each told the shared rules and
    that type's own (agents.yaml scam_caller.by_type). Called with the call's `scam_type`."""

    def __init__(self, settings: ScamCallerSettings) -> None:
        super().__init__()
        self.programs = {
            scam_type: dspy.ChainOfThought(
                configured("scam_caller", settings, settings.instructions_for(scam_type))
            )
            for scam_type in SCAM_TYPES
        }

    def forward(self, *, scam_type: str, **inputs: Any) -> dspy.Prediction:
        return self.programs[scam_type](**inputs)


def _program(name: str, settings: AgentSettings) -> dspy.ChainOfThought:
    """The agent's ChainOfThought program, with its instructions from config."""
    return dspy.ChainOfThought(configured(name, settings))
