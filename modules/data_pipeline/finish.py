"""Check a finished call: build its text, join its audio, ask the judge, find where the scam
starts, and list the reasons to reject it.

`finish_call` is what the finish node in graph.py runs. It returns an Ending, the fields the
stored call gets once it is finished.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .agents import Agents
from .audio import join_audio
from .budget import prompt_budget, prompt_tokens
from .config import Config
from .records import CallRecord, Judge, Turn
from .tts import Voices
from .turns import CALLER, RECEIVER, ask

log = logging.getLogger("data_pipeline")

TOO_LONG_FOR_JUDGE = "too_long_for_judge"
TOO_LONG_FOR_CONTEXT = "too_long_for_context"
SCAM_CLARITY_POINTS = 25
LABEL_MATCH_POINTS = 25
LANGUAGE_REALISM_POINTS = 25
SETUP_MATCH_POINTS = 25
PASS_MARK_POINTS = 50  # a judged call scoring this or less is rejected, more is passed
JUDGE_LOW_CONFIDENCE = "judge_low_confidence"
# What the judge is shown of the selected scenario; the rest only steers generation.
_JUDGED_SCENARIO_FIELDS = ("role", "org", "goal", "tactics", "receiver_context")

# The judge reads clean typed text but still reports typos that are not there (#58).
_SPELLING_TALK = re.compile(r"spell|typo|tone mark|สะกด", re.IGNORECASE)
_SPELLING_CLAIM = re.compile(r'\[(\d+)\]\s*"([^"]+)"')  # [n] "word", as agents.yaml asks
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(kw_only=True)
class Ending:
    """What a finished call adds to its record. A call that ended early because a prompt did
    not fit gets only the text, the leaks and the reasons; the rest keep their defaults."""

    text: str  # one "A: ..." or "B: ..." line per turn
    speaker_turns: list[dict]  # speaker, text, and start and end in the joined audio
    entities: list[dict]  # the filled values, with spans within `text`
    receiver_leaked: list[str]  # every kind of detail the receiver gave away, sorted
    first_leak_turn: int | None  # the first receiver turn that gave one away
    reasons: list[str]  # reasons to reject the call; empty means accepted
    audio_path: str | None = None  # the joined call audio; only when every turn was spoken
    reference_summary: str = ""  # the judge's summary
    scam_start_turn: int | None = None
    scam_start_from: str | None = None  # caller | judge: who pointed at scam_start_turn
    judge: Judge | None = None


def finish_call(
    record: CallRecord, agents: Agents, config: Config, voices: Voices | None, folder: Path
) -> Ending:
    """Build the text, join the audio, ask the judge, and list the reasons to reject the call.

    `record` is the call as stored so far.
    """
    turns = record.generation.turns
    text, speaker_turns, entities = call_text(turns)
    audio_path = None
    gap_s = config.agents.tts.gap_s
    joined = None if voices is None else join_audio(turns, record.id, folder, gap_s)
    if joined is not None:
        audio_path, times = joined
        speaker_turns = [
            {**speaker_turn, "start": start, "end": end}
            for speaker_turn, (start, end) in zip(speaker_turns, times, strict=True)
        ]
    judge = ask_judge(agents, config, turns, record.label_type, record.generation.call)
    start, start_from = None, None
    if record.label_scam:
        reveal_turn = record.generation.call["reveal_turn"]
        start, start_from = scam_start(turns, reveal_turn, judge, record.id)
    receiver_leaked, first_leak_turn = leak_summary(turns)
    return Ending(
        text=text,
        speaker_turns=speaker_turns,
        entities=entities,
        receiver_leaked=receiver_leaked,
        first_leak_turn=first_leak_turn,
        reasons=reasons_to_reject(turns, judge, config),
        audio_path=audio_path,
        reference_summary=judge.summary,
        scam_start_turn=start,
        scam_start_from=start_from,
        judge=judge,
    )


def ended_early(turns: list[Turn]) -> Ending:
    """The ending of a call cut short because the next prompt could not fit."""
    text, speaker_turns, entities = call_text(turns)
    receiver_leaked, first_leak_turn = leak_summary(turns)
    return Ending(
        text=text,
        speaker_turns=speaker_turns,
        entities=entities,
        receiver_leaked=receiver_leaked,
        first_leak_turn=first_leak_turn,
        reasons=[TOO_LONG_FOR_CONTEXT],
    )


def call_text(turns: list[Turn]) -> tuple[str, list[dict], list[dict]]:
    """The call's text, its speaker_turns (no audio times yet), and the entity spans in it."""
    lines, spans, offset = [], [], 0
    for turn in turns:
        prefix = f"{turn.speaker}: "
        for entity in turn.entities:
            start = offset + len(prefix) + entity["start"]
            spans.append({**entity, "start": start, "end": start + len(entity["text"])})
        line = prefix + turn.text
        lines.append(line)
        offset += len(line) + 1  # newline
    speaker_turns = [
        {"speaker": turn.speaker, "start": None, "end": None, "text": turn.text} for turn in turns
    ]
    return "\n".join(lines), speaker_turns, spans


def leak_summary(turns: list[Turn]) -> tuple[list[str], int | None]:
    """Every kind of detail the receiver gave away, sorted, and the first turn that gave one
    away."""
    leaked = sorted({label for turn in turns for label in turn.leaked})
    first = next((i for i, turn in enumerate(turns) if turn.leaked), None)
    return leaked, first


def ask_judge(
    agents: Agents, config: Config, turns: list[Turn], label_type: str | None, call: dict
) -> Judge:
    """The judge's answer: does the call show what its label says, does it reflect the persona
    and scenario selected before it started, and where does the scam start.

    `call` is the sampled CallConfig as stored in generation.call.
    """
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    transcript = "\n".join(
        f"[{i}] {tag[turn.speaker]}: {turn.text}" for i, turn in enumerate(turns)
    )
    intended = config.agents.judge.intended[label_type or "benign"]
    judge_inputs = dict(
        intended=intended,
        transcript=transcript,
        realism_criteria=config.agents.judge.realism_criteria,
        scenario={
            name: call["scenario"][name]
            for name in _JUDGED_SCENARIO_FIELDS
            if call["scenario"].get(name)
        },
        # the profile key is for choosing a profile on the command line, not for the model
        persona={name: value for name, value in call["receiver"].items() if name != "key"},
    )
    tokens = prompt_tokens(agents.signatures["judge"], judge_inputs)
    if tokens > prompt_budget(config, "judge"):
        # The judge must read the whole call, so a call that does not fit is not judged at all.
        return Judge(
            label_holds=False,
            scam_start_turn=-1,
            giveaway=False,
            confidence=0.0,
            summary="",
            issues="not judged: the call does not fit the judge's context window",
            reasoning="",
            skipped=TOO_LONG_FOR_JUDGE,
            prompt_tokens=tokens,
        )
    pred = ask(agents, "judge", **judge_inputs)
    scam_clear = bool(pred.scam_clear)
    label_holds = bool(pred.label_holds)
    language_realistic = bool(pred.language_realistic)
    setup_holds = bool(pred.setup_holds)
    return Judge(
        prompt_tokens=tokens,
        label_holds=label_holds,
        scam_clear=scam_clear,
        scam_clarity_reason=(pred.scam_clarity_reason or "").strip(),
        label_reason=(pred.label_reason or "").strip(),
        language_realistic=language_realistic,
        language_realism_reason=(pred.language_realism_reason or "").strip(),
        setup_holds=setup_holds,
        setup_reason=(pred.setup_reason or "").strip(),
        scam_start_turn=int(pred.scam_start_turn),
        giveaway=bool(pred.giveaway),
        confidence=_judge_score(scam_clear, label_holds, language_realistic, setup_holds),
        summary=(pred.summary or "").strip(),
        issues=drop_false_spelling((pred.issues or "").strip(), turns),
        reasoning=(getattr(pred, "reasoning", "") or "").strip(),
    )


def _judge_score(
    scam_clear: bool, label_holds: bool, language_realistic: bool, setup_holds: bool
) -> float:
    """The deterministic score behind the judge's displayed confidence rate."""
    points = (
        SCAM_CLARITY_POINTS * scam_clear
        + LABEL_MATCH_POINTS * label_holds
        + LANGUAGE_REALISM_POINTS * language_realistic
        + SETUP_MATCH_POINTS * setup_holds
    )
    return points / 100


def drop_false_spelling(issues: str, turns: list[Turn]) -> str:
    """The judge's issues without the sentences about spelling that code does not back.

    A sentence about spelling stays only if it quotes a word as [n] "word" and the spell check
    flagged that word in turn n. Code checks spelling itself (Turn.spelling, tone_mark_order),
    so a dropped claim loses nothing.
    """
    kept = [
        sentence
        for sentence in _SENTENCE_END.split(issues)
        if not _SPELLING_TALK.search(sentence) or _backed_by_spell_check(sentence, turns)
    ]
    return " ".join(kept) if kept else "none"


def _backed_by_spell_check(sentence: str, turns: list[Turn]) -> bool:
    """True if a [n] "word" claim in the sentence names a word the spell check flagged in turn n."""
    return any(
        int(n) < len(turns) and word in turns[int(n)].spelling
        for n, word in _SPELLING_CLAIM.findall(sentence)
    )


def scam_start(
    turns: list[Turn], reveal_turn: int, judge: Judge, call_id: str = ""
) -> tuple[int | None, str | None]:
    """First turn where the scam shows: the caller's own flag, else the judge's answer.

    Callers often under-report their own ask (a link "to check the parcel" is an ask), while
    the judge reads the whole call. The source is kept so the two can be compared later.
    `reveal_turn` counts caller turns; `call_id` only names the call in the log.
    """
    for i, turn in enumerate(turns):
        if turn.speaker != CALLER or not turn.asks_now:
            continue
        if i // 2 < reveal_turn:  # the plan forbids asking before the reveal turn
            log.warning(
                "%s turn %d: asks_now before reveal turn %d, ignored", call_id, i, reveal_turn
            )
            continue
        return i, "caller"
    at = judge.scam_start_turn
    # the judge must point at a caller turn: the receiver never makes a scam move
    if judge.label_holds and 0 <= at < len(turns) and turns[at].speaker == CALLER:
        return at, "judge"
    return None, None


def reasons_to_reject(turns: list[Turn], judge: Judge, config: Config) -> list[str]:
    """Reasons to reject the call; empty means accepted."""
    reasons = []
    if len(turns) < config.scenarios.calls.min_turns:
        reasons.append("too_short")
    # wrong_particle is a soft flag: it is kept on the turn for review but does not reject the
    # whole call for one gender-marker slip (turns.write_turn, fill.wrong_particles).
    reasons += sorted(
        {
            problem
            for turn in turns
            for problem in turn.problems
            if not problem.startswith("wrong_particle")
        }
    )
    if any("tone_mark_order" in turn.spelling for turn in turns):
        reasons.append("tone_mark_order")
    # several misspelled words in one receiver line means it is garbled (ตองจายเทาไร for
    # ต้องจ่ายเท่าไร); one odd word alone (a name or loanword the dictionary lacks) is tolerated.
    if any(
        turn.speaker == RECEIVER and len({s for s in turn.spelling if s != "tone_mark_order"}) >= 2
        for turn in turns
    ):
        reasons.append("garbled_spelling")
    if judge.skipped:  # not judged: there is nothing else to hold against the call
        return [*reasons, judge.skipped]
    # The confidence score is the judge's only say: above the pass mark it passes the call,
    # and no other judge answer (label_holds, giveaway, scam_start_turn) can reject it.
    if (judge.confidence or 0.0) * 100 <= PASS_MARK_POINTS:
        reasons.append(JUDGE_LOW_CONFIDENCE)
    return reasons
