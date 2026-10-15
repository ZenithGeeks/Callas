"""Check a finished call: build its text, join its audio, ask the judge, find where the scam
starts, and list the reasons to reject it.

`finish_call` is what the finish node in graph.py runs. It returns an Ending, the fields the
stored call gets once it is finished.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from .agents import Agents
from .audio import join_audio
from .budget import prompt_budget, prompt_tokens
from .config import Config
from .tts import Voices
from .turns import CALLER, RECEIVER, Turn, ask

log = logging.getLogger("data_pipeline")

TOO_LONG_FOR_JUDGE = "too_long_for_judge"
TOO_LONG_FOR_CONTEXT = "too_long_for_context"

# Ending fields that sit at the top of the stored record; the others go in its generation block.
RECORD_FIELDS = (
    "text",
    "speaker_turns",
    "entities",
    "audio_path",
    "reference_summary",
    "scam_start_turn",
)


class Ending(TypedDict, total=False):
    """What a finished call adds to its record. A call that ended early because a prompt did
    not fit gets only the text, the leaks and the reasons."""

    text: str  # one "A: ..." or "B: ..." line per turn
    speaker_turns: list[dict]  # speaker, text, and start and end in the joined audio
    entities: list[dict]  # the filled values, with spans within `text`
    audio_path: str  # the joined call audio; only when every turn was spoken
    reference_summary: str  # the judge's summary
    scam_start_turn: int | None
    scam_start_from: str | None  # caller | judge: who pointed at scam_start_turn
    receiver_leaked: list[str]  # every kind of detail the receiver gave away, sorted
    first_leak_turn: int | None  # the first receiver turn that gave one away
    judge: dict
    reasons: list[str]  # reasons to reject the call; empty means accepted


def finish_call(
    record: dict, agents: Agents, config: Config, voices: Voices | None, folder: Path
) -> Ending:
    """Build the text, join the audio, ask the judge, and list the reasons to reject the call.

    `record` is the call as stored so far.
    """
    turns = record["generation"]["turns"]
    ending: Ending = {**call_text(turns), **leak_summary(turns)}
    gap_s = config.agents.tts.gap_s
    joined = None if voices is None else join_audio(turns, record["id"], folder, gap_s)
    if joined is not None:
        ending["audio_path"], times = joined
        ending["speaker_turns"] = [
            {**speaker_turn, "start": start, "end": end}
            for speaker_turn, (start, end) in zip(ending["speaker_turns"], times, strict=True)
        ]
    judge = ask_judge(agents, config, turns, record["label_type"])
    ending["judge"], ending["reference_summary"] = judge, judge["summary"]
    if record["label_scam"]:
        judged = {**record, "generation": {**record["generation"], "judge": judge}}
        ending["scam_start_turn"], ending["scam_start_from"] = scam_start(judged)
    else:
        ending["scam_start_turn"], ending["scam_start_from"] = None, None
    ending["reasons"] = reasons_to_reject(
        turns, judge, record["label_scam"], ending["scam_start_turn"], config
    )
    return ending


def ended_early(turns: list[Turn]) -> Ending:
    """The ending of a call cut short because the next prompt could not fit."""
    return {**call_text(turns), **leak_summary(turns), "reasons": [TOO_LONG_FOR_CONTEXT]}


def call_text(turns: list[Turn]) -> Ending:
    """The call's text, its speaker_turns (no audio times yet), and the entity spans in it."""
    lines, spans, offset = [], [], 0
    for turn in turns:
        prefix = f"{turn['speaker']}: "
        for entity in turn["entities"]:
            start = offset + len(prefix) + entity["start"]
            spans.append({**entity, "start": start, "end": start + len(entity["text"])})
        line = prefix + turn["text"]
        lines.append(line)
        offset += len(line) + 1  # newline
    speaker_turns = [
        {"speaker": turn["speaker"], "start": None, "end": None, "text": turn["text"]}
        for turn in turns
    ]
    return {"text": "\n".join(lines), "speaker_turns": speaker_turns, "entities": spans}


def leak_summary(turns: list[Turn]) -> Ending:
    """Every kind of detail the receiver gave away, and the first turn that gave one away."""
    return {
        "receiver_leaked": sorted({label for t in turns for label in t.get("leaked", [])}),
        "first_leak_turn": next((i for i, t in enumerate(turns) if t.get("leaked")), None),
    }


def ask_judge(agents: Agents, config: Config, turns: list[Turn], label_type: str | None) -> dict:
    """The judge's answer: does the call show what its label says, and is the Thai natural."""
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    transcript = "\n".join(
        f"[{i}] {tag[turn['speaker']]}: {turn['text']}" for i, turn in enumerate(turns)
    )
    intended = config.agents.judge.intended[label_type or "benign"]
    tokens = prompt_tokens(
        agents.signatures["judge"], dict(intended=intended, transcript=transcript)
    )
    if tokens > prompt_budget(config, "judge"):
        # The judge must read the whole call, so a call that does not fit is not judged at all.
        return {
            "label_holds": False,
            "scam_start_turn": -1,
            "natural_thai": 0,
            "giveaway": False,
            "summary": "",
            "issues": "not judged: the call does not fit the judge's context window",
            "reasoning": "",
            "skipped": TOO_LONG_FOR_JUDGE,
            "prompt_tokens": tokens,
        }
    pred = ask(agents, "judge", intended=intended, transcript=transcript)
    return {
        "prompt_tokens": tokens,
        "label_holds": bool(pred.label_holds),
        "scam_start_turn": int(pred.scam_start_turn),
        "natural_thai": int(pred.natural_thai),
        "giveaway": bool(pred.giveaway),
        "summary": (pred.summary or "").strip(),
        "issues": (pred.issues or "").strip(),
        "reasoning": (getattr(pred, "reasoning", "") or "").strip(),
    }


def scam_start(record: dict) -> tuple[int | None, str | None]:
    """First turn where the scam shows: the caller's own flag, else the judge's answer.

    Callers often under-report their own ask (a link "to check the parcel" is an ask), while
    the judge reads the whole call. The source is kept so the two can be compared later.
    """
    turns = record["generation"]["turns"]
    reveal = record["generation"]["call"]["reveal_turn"]
    for i, turn in enumerate(turns):
        if turn["speaker"] != CALLER or not turn["asks_now"]:
            continue
        if i // 2 < reveal:  # the plan forbids asking before the reveal turn: not a scam move
            log.warning(
                "%s turn %d: asks_now before reveal turn %d, ignored", record["id"], i, reveal
            )
            continue
        return i, "caller"
    judge = record["generation"]["judge"]
    at = judge["scam_start_turn"]
    # the judge must point at a caller turn: the receiver never makes a scam move
    if judge["label_holds"] and 0 <= at < len(turns) and turns[at]["speaker"] == CALLER:
        return at, "judge"
    return None, None


def reasons_to_reject(
    turns: list[Turn],
    judge: dict,
    label_scam: bool,
    scam_start_turn: int | None,
    config: Config,
) -> list[str]:
    """Reasons to reject the call; empty means accepted."""
    reasons = []
    if len(turns) < config.scenarios.calls.min_turns:
        reasons.append("too_short")
    reasons += sorted({problem for turn in turns for problem in turn["problems"]})
    if any("tone_mark_order" in turn["spelling"] for turn in turns):
        reasons.append("tone_mark_order")
    if judge.get("skipped"):  # not judged: there is nothing else to hold against the call
        return [*reasons, judge["skipped"]]
    if not judge["label_holds"]:
        reasons.append("judge_label_does_not_hold")
    if judge["giveaway"]:
        reasons.append("judge_giveaway")
    if label_scam and scam_start_turn is None:
        reasons.append("no_scam_move")
    if not label_scam and judge["scam_start_turn"] >= 0:
        reasons.append("judge_saw_scam_in_normal_call")
    return reasons
