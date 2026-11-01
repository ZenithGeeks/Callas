"""Write one turn of a call: ask the speaker's agent for a line, fill its slots, note problems.

`write_turn` is what the caller and receiver nodes in graph.py run. `write_plan` is the planner,
which writes the scam caller's plan once before the first turn. Every agent's prompt must fit
its context window, so the conversation is cut to fit first (budget.py).
"""

from __future__ import annotations

import logging
import re
import time
from difflib import SequenceMatcher

import dspy
import openai
from dspy.utils.exceptions import AdapterParseError

from llm_serving.client import LLMUnavailable
from llm_serving.dspy_programs import _UNAVAILABLE as UNAVAILABLE
from nlp_core.text import autocorrect_text, normalize_text

from . import romance_voice
from .agents import Agents
from .budget import ContextTooLong, prompt_budget, prompt_tokens, trim_notes, trim_turns
from .config import Config
from .fill import (
    CALLER_NUMBER_SLOTS,
    ENDS_WITH,
    RECEIVER_ONLY_SLOTS,
    RECEIVER_OWN_SLOTS,
    SLOT_RE,
    SlotFiller,
    receiver_leaks,
    self_identity_question,
    strip_leading_interjection,
    wrong_particles,
)
from .patterns import render_notes
from .records import Briefing, Turn
from .scenarios import CallConfig
from .spoken import pick_speech_style, spoken_text
from .store import Store

log = logging.getLogger("data_pipeline")

CALLER, RECEIVER = "A", "B"  # speaker ids in the record; the caller always speaks first


def write_turn(
    agents: Agents,
    call: CallConfig,
    config: Config,
    turns: list[Turn],
    filler: SlotFiller,
    plan_steps: list[str],
    speaker: str,
    footing: str = "",
) -> Turn:
    """Ask the speaker's agent for its next line, fill the line's slots and note its problems."""
    started = time.perf_counter()
    if speaker == CALLER:
        agent, pred, tokens = caller_says(agents, call, config, turns, plan_steps)
    else:
        agent = "receiver"
        pred, tokens = receiver_says(agents, call, config, turns, filler, footing)
    raw = (pred.line or "").strip()
    if speaker == RECEIVER:  # drop a canned opening exclamation the model keeps reaching for
        raw = strip_leading_interjection(raw)
    # text (what the judge, the record and the voice get) is the cleaned line. Autocorrect is
    # always on here: typed lines have real misspellings, unlike ASR text
    # (docs/eval/autocorrect.md), so the AUTOCORRECT setting only covers live calls.
    # Turn.spelling checks the cleaned line (what is still wrong); Turn.fixes lists what changed.
    cleaned = autocorrect_text(normalize_text(raw), mode="pythainlp")
    text, entities, problems = filler.fill(cleaned)
    if speaker == CALLER and any(name in RECEIVER_ONLY_SLOTS for name in SLOT_RE.findall(raw)):
        problems.append("caller_used_receiver_slot")  # a caller cannot know these details
    if speaker == RECEIVER:
        problems += [f"wrong_particle:{p}" for p in wrong_particles(raw, call.receiver.gender)]
        if self_identity_question(raw):
            problems.append("self_identity_question")
    if not text:
        problems.append("empty_line")
    return Turn(
        speaker=speaker,
        agent=agent,
        raw=raw,
        text=text,
        spoken_text=spoken_text(text, entities, speech_style(call, config)),
        entities=entities,
        asks_now=bool(getattr(pred, "asks_now", False)),
        end_call=bool(pred.end_call),
        reasoning=(getattr(pred, "reasoning", "") or "").strip(),
        problems=problems,
        leaked=receiver_leaks(raw) if speaker == RECEIVER else [],
        spelling=spelling(cleaned),
        fixes=text_fixes(raw, cleaned),
        latency_s=round(time.perf_counter() - started, 3),
        prompt_tokens=tokens,
    )


def caller_says(
    agents: Agents, call: CallConfig, config: Config, turns: list[Turn], plan_steps: list[str]
) -> tuple[str, dspy.Prediction, int]:
    """Ask the scam or legit caller for its next line. Returns (agent, answer, prompt tokens)."""
    scenario = call.scenario
    if call.call_type == "romance":
        voice = romance_voice.voice(
            config.romance_voice,
            call.caller_gender,
            call.receiver.gender,
            call.scenario,
            f"{call.call_type}:{call.seed}",
        )
    else:
        voice = config.agents.speakers.voice[call.caller_gender]
    inputs = dict(
        role=scenario.role,
        voice=voice,
        # an office-style scam caller always has an org: the filler makes one up if the scenario
        # has none, and its opener says it; a romance caller is someone the receiver knows
        org="{ORG}" if scenario.org or (call.label_scam and call.call_type != "romance") else "",
        goal=scenario.goal,
        receiver_name="{RECEIVER_NAME}" if call.caller_knows_name else "unknown",
    )
    closing = ending(call, config, len(turns))
    if call.label_scam:
        agent = "scam_caller"
        own_turn = len(turns) // 2
        steer = " ".join(filter(None, (pacing(call, config, own_turn), closing)))
        inputs |= dict(tactics=scenario.tactics, pacing=steer, plan=plan_steps)
    else:
        agent = "legit_caller"
        inputs |= dict(manner=scenario.tactics, pacing=closing)
    # callers read their own slots ({PHONE}), not the values the receiver heard
    lines = conversation_lines(turns, config, raw=True)
    # the scam caller has one prompt per scam type (agents.ScamCaller)
    scam_type = call.call_type if call.label_scam else None
    text, tokens = _fit_conversation(agents, config, agent, inputs, lines, scam_type)
    extra = {"scam_type": scam_type} if scam_type else {}
    return agent, ask(agents, agent, **inputs, **extra, conversation=text), tokens


def receiver_says(
    agents: Agents,
    call: CallConfig,
    config: Config,
    turns: list[Turn],
    filler: SlotFiller,
    footing: str = "",
) -> tuple[dspy.Prediction, int]:
    """Ask the receiver for its next line. Returns (answer, prompt tokens)."""
    closing = ending(call, config, len(turns))
    if closing and call.label_scam and call.receiver.scam_outcome == "scammed":
        # a call locked to end with the receiver scammed closes on it agreeing to the ask
        closing = f"{closing} {config.agents.speakers.ending.scammed}"
    stage_text = stage(config, len(turns) // 2, closing=bool(closing))
    if closing:
        stage_text = f"{stage_text} {closing}"
    # The receiver always hears the scenario's backstory (romance/family/friend): who the caller is
    # and the prior relationship. `footing` adds to this, setting the register from the opening; the
    # two agree because footing is worked out from this same context plus the caller's first line.
    if call.scenario.receiver_context:
        stage_text = f"{call.scenario.receiver_context} {stage_text}"
    inputs = dict(
        # key picks a profile on the command line; scam_outcome is the call's locked label, and
        # the persona (drawn to fit it) carries the behaviour, so neither is shown to the model
        persona=call.receiver.model_dump(exclude={"key", "scam_outcome"}),
        voice=config.agents.speakers.voice[call.receiver.gender],
        stage=stage_text,
        footing=footing,
        red_flags=config.red_flags,
    )
    lines = receiver_lines(turns, filler, config)
    text, tokens = _fit_conversation(agents, config, "receiver", inputs, lines)
    return ask(agents, "receiver", **inputs, conversation=text), tokens


def read_footing(
    agents: Agents, call: CallConfig, config: Config, turns: list[Turn], filler: SlotFiller
) -> str:
    """Work out once, from the scenario's backstory and the caller's opening, who the caller is to
    the receiver and how it should talk back. Empty for a stranger call: the scenario sets no prior
    relationship, so the receiver stays politely formal and no extra model call is made.

    Only the caller's first line is given as the opening, never the whole call, so a resume (which
    runs this again) gets the same footing it did the first time."""
    if not call.scenario.receiver_context:
        return ""
    opening = receiver_view(turns[:1], filler, config)  # the caller's first line, as heard
    pred = ask(
        agents,
        "footing",
        persona=call.receiver.model_dump(exclude={"key", "scam_outcome"}),
        context=call.scenario.receiver_context,
        opening=opening,
    )
    return (getattr(pred, "footing", "") or "").strip()


def write_plan(agents: Agents, call: CallConfig, config: Config, store: Store) -> Briefing:
    """The scam caller's plan for this call, from its scenario and recent pattern notes."""
    notes = store.notes(call.call_type, limit=config.agents.planner.notes_limit)
    last = (call.max_turns - 1) // 2
    scenario = call.scenario
    turn_budget = (
        f"The caller has {last + 1} turns. The first ask may come on turn "
        f"{call.reveal_turn + 1} at the earliest."
    )
    inputs = dict(
        scam_type=config.agents.judge.intended[call.call_type],
        role=scenario.role,
        goal=scenario.goal,
        tactics=scenario.tactics,
        turns=turn_budget,
    )
    signature, budget = agents.signatures["planner"], prompt_budget(config, "planner")

    def tokens_with(kept) -> int:
        return prompt_tokens(signature, {**inputs, "pattern_notes": render_notes(list(kept))})

    # the notes are newest first: when the prompt is too long the oldest are dropped
    notes = trim_notes(notes, lambda kept: tokens_with(kept) <= budget)
    pred = ask(agents, "planner", **inputs, pattern_notes=render_notes(notes))
    return Briefing(
        plan=[step.strip() for step in pred.plan if step.strip()],
        # summaries, not ids: a wiki sync re-creates the notes and gives them new ids
        notes_used=[note["summary"] for note in notes],
        prompt_tokens=tokens_with(notes),
        reasoning=(getattr(pred, "reasoning", "") or "").strip(),
    )


def ask(agents: Agents, agent: str, **inputs) -> dspy.Prediction:
    """Run one agent on its own LM.

    A server that cannot be reached raises LLMUnavailable, which stops the run instead of
    being retried (graph.retry_policy retries only is_model_error).
    """
    with dspy.context(lm=agents.lms[agent]):
        try:
            return getattr(agents, agent)(**inputs)
        except UNAVAILABLE as exc:
            raise LLMUnavailable(f"cannot reach LLM server: {type(exc).__name__}") from exc


def is_model_error(exc: BaseException) -> bool:
    """True for a failure that comes from the model's answer or from rate limiting: a reply
    DSPy could not parse, or a 429 / 5xx answer. Anything else (a bug, a bad config, a store
    error) is not the model's fault, so it is neither retried nor blamed on the model."""
    # DSPy 3.4 turns a 5xx answer into dspy.LMServerError, not openai.APIStatusError
    if isinstance(exc, (AdapterParseError, openai.RateLimitError, dspy.LMServerError)):
        return True
    if isinstance(exc, openai.APIStatusError) and (
        exc.status_code == 429 or exc.status_code >= 500
    ):
        return True
    # DSPy wraps rate limits in its own classes (LMRateLimitError, lm15.RateLimitError)
    return any("RateLimit" in cls.__name__ for cls in type(exc).__mro__)


def _fit_conversation(
    agents: Agents,
    config: Config,
    agent: str,
    inputs: dict,
    lines: list[str],
    scam_type: str | None = None,
) -> tuple[str, int]:
    """The conversation text for this agent's prompt and the prompt's estimated size. When the
    whole call does not fit the agent's budget, the first turns and the newest turns are kept
    and the middle is replaced by a marker (budget.trim_turns). `scam_type` picks the scam
    caller's prompt."""
    signature, budget = agents.signature(agent, scam_type), prompt_budget(config, agent)

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


def conversation_lines(turns: list[Turn], config: Config, *, raw: bool) -> list[str]:
    """One line per turn: the speaker's tag and what was said (`raw`: with the slots)."""
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    return [f"{tag[turn.speaker]}: {turn.raw if raw else turn.text}" for turn in turns]


def receiver_lines(turns: list[Turn], filler: SlotFiller, config: Config) -> list[str]:
    """The call as the receiver hears it, one line per turn: real values, except its own name
    and details, which stay the slots that its instructions call "your name", "your bank
    account" and so on, and the caller's numbers and links (fill.CALLER_NUMBER_SLOTS). Without
    its own slots the receiver did not recognise its own name and answered "you have the wrong
    number"; without the caller's, it repeated the digits instead of the slot (#99).

    `filler` must be this call's: the studio can give the receiver another name than the seed.
    """
    values = filler.values
    # longest value first, so a value inside a longer one is not replaced on its own
    slots = sorted(RECEIVER_OWN_SLOTS + CALLER_NUMBER_SLOTS, key=lambda slot: -len(values[slot]))
    lines = []
    for line in conversation_lines(turns, config, raw=False):
        for slot in slots:
            line = line.replace(values[slot], "{" + slot + "}")
        # a card said by its last 4 digits only (fill.ENDS_WITH) is the slot too
        line = re.sub(f"({ENDS_WITH}){values['CARD_NUMBER'][-4:]}", r"\g<1>{CARD_NUMBER}", line)
        lines.append(line)
    return lines


def receiver_view(turns: list[Turn], filler: SlotFiller, config: Config) -> str:
    """The receiver's lines as one text, before any cut to fit its prompt."""
    lines = receiver_lines(turns, filler, config)
    return "\n".join(lines) if lines else config.agents.speakers.empty_call


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


def stage(config: Config, own_turn: int, closing: bool = False) -> str:
    """Tell the receiver whether it must still stay on the line or may react as it likes.

    `closing`: the call is at its end (see ending), so "do not hang up yet" no longer applies.
    """
    receiver = config.agents.receiver
    stay = own_turn < receiver.stay_on_line and not closing
    text = receiver.stage.stay if stay else receiver.stage.free
    return text.format(turn=own_turn + 1)


def ending(call: CallConfig, config: Config, done: int) -> str:
    """The note for the speaker of turn `done` (0-based) when the call is near max_turns:
    `last` on the call's final turn, `soon` on the one before it, else "". The graph stops the
    call at max_turns (graph.next_speaker), so without it the last line ends mid-scene (#150)."""
    texts = config.agents.speakers.ending
    left = call.max_turns - done  # turns left, this one included
    return {1: texts.last, 2: texts.soon}.get(left, "")


def speech_style(call: CallConfig, config: Config) -> str:
    """How this call's numbers are said, "dash" or "pause" (see spoken.py)."""
    return pick_speech_style(call.call_type, call.seed, config.agents.speech.say_dash_share)


_THAI_CONSONANT = re.compile("[ก-ฮ]")


def text_fixes(before: str, after: str) -> list[dict]:
    """What the cleanup changed in a line, as {"before", "after"} pairs, for the review pages.

    Changes in spacing alone are not listed.
    """
    if before == after:
        return []
    from pythainlp.tokenize import word_tokenize

    old, new = (word_tokenize(s, engine="newmm", keep_whitespace=False) for s in (before, after))
    fixes = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        was, now = "".join(old[i1:i2]), "".join(new[j1:j2])
        # a lone tone mark or vowel ("่่") cannot be read alone: show the words around it too
        if not (_THAI_CONSONANT.search(was) and _THAI_CONSONANT.search(now)):
            was = "".join(old[max(i1 - 1, 0) : i2 + 1])
            now = "".join(new[max(j1 - 1, 0) : j2 + 1])
        fixes.append({"before": was, "after": now})
    return fixes


def spelling(raw: str) -> list[str]:
    """Words that look misspelled. Checked on the line before filling, so names are skipped."""
    from tts_voice_agent.spelling import spelling_problems

    return spelling_problems(SLOT_RE.sub(" ", raw))
