"""Check a finished call: its text and entity spans, the receiver's leaks, the judge's answer,
where the scam starts, and the reasons to reject it."""

from __future__ import annotations

import logging
from pathlib import Path

import dspy

from .agents import Agents
from .audio import _join_audio
from .budget import prompt_budget, prompt_tokens
from .config import Config
from .tts import Voices
from .turns import CALLER, RECEIVER, _model_call

log = logging.getLogger("data_pipeline")

TOO_LONG_FOR_JUDGE = "too_long_for_judge"
TOO_LONG_FOR_CONTEXT = "too_long_for_context"


def _finish_call(
    record: dict, agents: Agents, config: Config, voices: Voices | None, out_dir: Path
) -> None:
    generation = record["generation"]
    _finish_text(record)
    _finish_leaks(record)
    if voices is not None:
        _join_audio(record, out_dir, config.agents.tts.gap_s)
    generation["judge"] = _judge(agents, config, record)
    record["reference_summary"] = generation["judge"]["summary"]
    if record["label_scam"]:
        record["scam_start_turn"], generation["scam_start_from"] = _scam_start(record)
    generation["reasons"] = _verdict(record, config)
    generation["status"] = "rejected" if generation["reasons"] else "accepted"


def _finish_text(record: dict) -> None:
    """Build text, speaker_turns and entity spans from the turns."""
    lines, spans, offset = [], [], 0
    turns = record["generation"]["turns"]
    for turn in turns:
        prefix = f"{turn['speaker']}: "
        for entity in turn["entities"]:
            start = offset + len(prefix) + entity["start"]
            spans.append({**entity, "start": start, "end": start + len(entity["text"])})
        line = prefix + turn["text"]
        lines.append(line)
        offset += len(line) + 1  # newline
    record["text"] = "\n".join(lines)
    record["speaker_turns"] = [
        {"speaker": turn["speaker"], "start": None, "end": None, "text": turn["text"]}
        for turn in turns
    ]
    record["entities"] = spans


def _finish_leaks(record: dict) -> None:
    """generation.receiver_leaked: every kind of detail the receiver gave away (sorted);
    generation.first_leak_turn: index of the first receiver turn that gave one, else None."""
    generation = record["generation"]
    turns = generation["turns"]
    generation["receiver_leaked"] = sorted({label for t in turns for label in t.get("leaked", [])})
    generation["first_leak_turn"] = next((i for i, t in enumerate(turns) if t.get("leaked")), None)


def _judge(agents: Agents, config: Config, record: dict) -> dict:
    speakers = config.agents.speakers
    tag = {CALLER: speakers.caller_tag, RECEIVER: speakers.receiver_tag}
    transcript = "\n".join(
        f"[{i}] {tag[turn['speaker']]}: {turn['text']}"
        for i, turn in enumerate(record["generation"]["turns"])
    )
    intended = config.agents.judge.intended[record["label_type"] or "benign"]
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
    with dspy.context(lm=agents.lms["judge"]):
        pred = _model_call(lambda: agents.judge(intended=intended, transcript=transcript))
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


def _scam_start(record: dict) -> tuple[int | None, str | None]:
    """First turn where the scam shows: the caller's own flag, else the judge's answer.

    Callers often under-report their own ask (a link "to check the parcel" is an ask), while
    the judge reads the whole call. The source is kept so the two can be compared later.
    """
    turns = record["generation"]["turns"]
    reveal = record["generation"]["call"]["reveal_turn"]
    flagged = None
    for i, turn in enumerate(turns):
        if turn["speaker"] != CALLER or not turn["asks_now"]:
            continue
        if i // 2 < reveal:  # the plan forbids asking before the reveal turn: not a scam move
            log.warning(
                "%s turn %d: asks_now before reveal turn %d, ignored", record["id"], i, reveal
            )
            continue
        flagged = i
        break
    if flagged is not None:
        return flagged, "caller"
    judge = record["generation"]["judge"]
    at = judge["scam_start_turn"]
    # the judge must point at a caller turn: the receiver never makes a scam move
    if judge["label_holds"] and 0 <= at < len(turns) and turns[at]["speaker"] == CALLER:
        return at, "judge"
    return None, None


def _verdict(record: dict, config: Config) -> list[str]:
    """Reasons to reject the call; empty means accepted."""
    generation, reasons = record["generation"], []
    turns = generation["turns"]
    if len(turns) < config.scenarios.calls.min_turns:
        reasons.append("too_short")
    reasons += sorted({problem for turn in turns for problem in turn["problems"]})
    if any("tone_mark_order" in turn["spelling"] for turn in turns):
        reasons.append("tone_mark_order")
    judge = generation["judge"]
    if judge.get("skipped"):  # not judged: there is nothing else to hold against the call
        return [*reasons, judge["skipped"]]
    if not judge["label_holds"]:
        reasons.append("judge_label_does_not_hold")
    if judge["giveaway"]:
        reasons.append("judge_giveaway")
    if record["label_scam"] and record["scam_start_turn"] is None:
        reasons.append("no_scam_move")
    if not record["label_scam"] and judge["scam_start_turn"] >= 0:
        reasons.append("judge_saw_scam_in_normal_call")
    return reasons
