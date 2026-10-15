"""Write one turn of a call: ask the agent whose turn it is, fill its slots, note problems.

Also the scam caller's plan, written once before the first turn, and what each agent is told
about where it is in the call (pacing, stage).
"""

from __future__ import annotations

import logging
import time

import dspy
import openai
from dspy.utils.exceptions import AdapterParseError

from llm_serving.client import LLMUnavailable
from llm_serving.dspy_programs import _UNAVAILABLE as UNAVAILABLE

from .agents import Agents
from .budget import ContextTooLong, prompt_budget, prompt_tokens, trim_notes, trim_turns
from .config import Config
from .fill import RECEIVER_ONLY_SLOTS, RECEIVER_OWN_SLOTS, SLOT_RE, SlotFiller, receiver_leaks
from .patterns import render_notes
from .scenarios import CallConfig
from .spoken import pick_speech_style, spoken_text
from .store import Store

log = logging.getLogger("data_pipeline")

CALLER, RECEIVER = "A", "B"  # speaker ids in the record; the caller always speaks first


def is_model_error(exc: BaseException) -> bool:
    """True for a failure that comes from the model's answer or from rate limiting: a reply
    DSPy could not parse, or a 429 / 5xx answer. Anything else (a bug, a bad config, a store
    error) is not the model's fault, so it is neither retried nor blamed on the model."""
    if isinstance(exc, (AdapterParseError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    ):
        return True
    # DSPy wraps rate limits in its own classes (LMRateLimitError, lm15.RateLimitError)
    return any("RateLimit" in cls.__name__ for cls in type(exc).__mro__)


def speech_style(call: CallConfig, config: Config) -> str:
    """ "dash" or "pause": how this call's numbers are said (see spoken.py)."""
    return pick_speech_style(call.call_type, call.seed, config.agents.speech.say_dash_share)


def _plan(agents: Agents, call: CallConfig, config: Config, store: Store) -> dict:
    """The scam caller's plan for this call, from its scenario and recent pattern notes."""
    notes = store.notes(call.call_type, limit=config.agents.planner.notes_limit)
    last = (call.max_turns - 1) // 2
    scenario = call.scenario
    turns = (
        f"The caller has {last + 1} turns. The first ask may come on turn "
        f"{call.reveal_turn + 1} at the earliest."
    )
    inputs = dict(
        scam_type=config.agents.judge.intended[call.call_type],
        role=scenario.role,
        goal=scenario.goal,
        tactics=scenario.tactics,
        turns=turns,
    )
    signature, budget = agents.signatures["planner"], prompt_budget(config, "planner")

    def tokens_with(kept) -> int:
        return prompt_tokens(signature, {**inputs, "pattern_notes": render_notes(list(kept))})

    # the notes are newest first: when the prompt is too long the oldest are dropped
    notes = trim_notes(notes, lambda kept: tokens_with(kept) <= budget)
    with dspy.context(lm=agents.lms["planner"]):
        pred = _model_call(lambda: agents.planner(**inputs, pattern_notes=render_notes(notes)))
    return {
        "plan": [step.strip() for step in pred.plan if step.strip()],
        # summaries, not ids: a wiki sync re-creates the notes and gives them new ids
        "notes_used": [note["summary"] for note in notes],
        "prompt_tokens": tokens_with(notes),
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
    }


def _next_turn(
    agents: Agents,
    call: CallConfig,
    config: Config,
    turns: list[dict],
    filler: SlotFiller,
    plan: list[str],
) -> dict:
    """Ask the agent whose turn it is for one line, fill its slots, and describe the turn."""
    started = time.perf_counter()
    speaker, agent_name, pred, tokens = _model_call(
        lambda: _speak(agents, call, config, turns, plan)
    )
    raw = (pred.line or "").strip()
    text, entities, problems = filler.fill(raw)
    leaked = receiver_leaks(raw) if speaker == RECEIVER else []
    if speaker == CALLER and any(name in RECEIVER_ONLY_SLOTS for name in SLOT_RE.findall(raw)):
        problems.append("caller_used_receiver_slot")  # a caller cannot know these details
    if not text:
        problems.append("empty_line")
    return {
        "speaker": speaker,
        "agent": agent_name,
        "raw": raw,
        "text": text,
        "spoken_text": spoken_text(text, entities, speech_style(call, config)),
        "entities": entities,
        "asks_now": bool(getattr(pred, "asks_now", False)),
        "end_call": bool(pred.end_call),
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
        "problems": problems,
        "leaked": leaked,
        "spelling": spelling(raw),
        "latency_s": round(time.perf_counter() - started, 3),
        "prompt_tokens": tokens,
        "audio": None,
        "audio_s": None,
    }


def _speak(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict], plan: list[str]
) -> tuple[str, str, dspy.Prediction, int]:
    """Run the agent whose turn it is. Returns (speaker, agent name, prediction, estimated
    prompt tokens)."""
    if len(turns) % 2 == 1:
        pred, tokens = _receiver_says(agents, call, config, turns)
        return RECEIVER, "receiver", pred, tokens
    agent_name, pred, tokens = _caller_says(agents, call, config, turns, plan)
    return CALLER, agent_name, pred, tokens


def _fit_conversation(
    agents: Agents, config: Config, agent: str, inputs: dict, lines: list[str]
) -> tuple[str, int]:
    """The conversation text for this agent's prompt and the prompt's estimated size. When the
    whole call does not fit the agent's budget, the first turns and the newest turns are kept
    and the middle is replaced by a marker (budget.trim_turns)."""
    signature, budget = agents.signatures[agent], prompt_budget(config, agent)

    def tokens(conversation: str) -> int:
        return prompt_tokens(signature, {**inputs, "conversation": conversation})

    if not lines:
        text = config.agents.speakers.empty_call
    else:
        text, hidden = trim_turns(lines, lambda conversation: tokens(conversation) <= budget)
        if hidden:
            log.info("%s: %d earlier turns left out of the prompt", agent, hidden)
    size = tokens(text)
    if size > budget:  # nothing left to cut: do not send a prompt the model cannot take
        raise ContextTooLong(agent, size, budget)
    return text, size


def _receiver_says(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict]
) -> tuple[dspy.Prediction, int]:
    own_turn = len(turns) // 2
    inputs = dict(
        # the profile key is for choosing a profile on the command line, not for the model
        persona=call.receiver.model_dump(exclude={"key"}),
        voice=config.agents.speakers.voice[call.receiver.gender],
        stage=stage(config, own_turn),
    )
    conversation_text, tokens = _fit_conversation(
        agents, config, "receiver", inputs, _receiver_lines(turns, call, config)
    )
    with dspy.context(lm=agents.lms["receiver"]):
        return agents.receiver(**inputs, conversation=conversation_text), tokens


def _receiver_lines(turns: list[dict], call: CallConfig, config: Config) -> list[str]:
    """The call as the receiver hears it, one line per turn: real values, except its own name
    and details, which stay the slots that its instructions call "your name", "your bank
    account" and so on."""
    values = SlotFiller(call, config.details).values
    own = sorted(RECEIVER_OWN_SLOTS, key=lambda slot: -len(values[slot]))
    lines = []
    for line in conversation_lines(turns, config, raw=False):
        for slot in own:
            line = line.replace(values[slot], "{" + slot + "}")
        lines.append(line)
    return lines


def _receiver_view(turns: list[dict], call: CallConfig, config: Config) -> str:
    """The call as the receiver hears it: real values, except its own name and details, which
    stay the slots that its instructions call "your name", "your bank account" and so on.
    Without this the receiver did not recognise its own name and answered "you have the wrong
    number"."""
    lines = _receiver_lines(turns, call, config)
    return "\n".join(lines) if lines else config.agents.speakers.empty_call


def _caller_says(
    agents: Agents, call: CallConfig, config: Config, turns: list[dict], plan: list[str]
) -> tuple[str, dspy.Prediction, int]:
    own_turn = len(turns) // 2
    scenario = call.scenario
    inputs = dict(
        role=scenario.role,
        voice=config.agents.speakers.voice[call.caller_gender],
        org="{ORG}" if scenario.org else "",
        goal=scenario.goal,
        receiver_name="{RECEIVER_NAME}" if call.caller_knows_name else "unknown",
    )
    lines = conversation_lines(turns, config, raw=True)
    if call.label_scam:
        inputs |= dict(tactics=scenario.tactics, pacing=pacing(call, config, own_turn), plan=plan)
        text, tokens = _fit_conversation(agents, config, "scam_caller", inputs, lines)
        with dspy.context(lm=agents.lms["scam_caller"]):
            return "scam_caller", agents.scam_caller(**inputs, conversation=text), tokens
    inputs["manner"] = scenario.tactics
    text, tokens = _fit_conversation(agents, config, "legit_caller", inputs, lines)
    with dspy.context(lm=agents.lms["legit_caller"]):
        return "legit_caller", agents.legit_caller(**inputs, conversation=text), tokens


def conversation_lines(turns: list[dict], config: Config, *, raw: bool) -> list[str]:
    """One line per turn: the speaker's tag and what was said."""
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    key = "raw" if raw else "text"
    return [f"{tag[turn['speaker']]}: {turn[key]}" for turn in turns]


def conversation(turns: list[dict], config: Config, *, raw: bool) -> str:
    """The call so far as Thai lines. Callers see their slots ({PHONE}); receivers see values."""
    if not turns:
        return config.agents.speakers.empty_call
    return "\n".join(conversation_lines(turns, config, raw=raw))


def pacing(call: CallConfig, config: Config, own_turn: int) -> str:
    """Tell the scam caller where it is in the call: before, at, or after its reveal turn."""
    texts = config.agents.scam_caller.pacing
    last = (call.max_turns - 1) // 2  # the caller's last possible own turn (0-based)
    if own_turn >= last - 1 and own_turn >= call.reveal_turn:
        text = texts.last
    elif own_turn < call.reveal_turn:
        text = texts.before
    elif own_turn == call.reveal_turn:
        text = texts.start
    else:
        text = texts.push
    return text.format(turn=own_turn + 1, turns=last + 1)


def stage(config: Config, own_turn: int) -> str:
    receiver = config.agents.receiver
    text = receiver.stage.stay if own_turn < receiver.stay_on_line else receiver.stage.free
    return text.format(turn=own_turn + 1)


def _model_call(fn):
    """Run one DSPy call. A connection failure becomes LLMUnavailable, which stops the run
    instead of being retried (see MODEL_RETRY)."""
    try:
        return fn()
    except UNAVAILABLE as exc:
        raise LLMUnavailable(f"cannot reach LLM server: {type(exc).__name__}") from exc


def spelling(raw: str) -> list[str]:
    """Words that look misspelled. Checked on the line before filling, so names are skipped."""
    from tts_voice_agent.spelling import spelling_problems

    return spelling_problems(SLOT_RE.sub(" ", raw))
