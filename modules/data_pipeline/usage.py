"""Real token usage of the model requests a call makes, from the server's answers.

`turns.ask` runs every agent under `dspy.track_usage()`; this module turns what DSPy collected
into one `usage` dict per step (a turn, the plan, the judge), sums them per call, and prints a
report over many calls:

    python -m data_pipeline.usage report [--type romance] [--since 2026-10-30] [--db PATH]

A step's usage: {"input": 2846, "output": 512, "requests": 2, "failed": 1,
"reasoning_share": 0.62}. input/output are the server's prompt and completion tokens over every
attempt, including attempts that raised and were retried (graph.retry_policy). `requests` counts
model requests (more than one: a retry, or DSPy's JSON fallback after an unparseable answer).
`reasoning_share` is an estimate: the share of the answer's characters in its `reasoning`
field, since the server counts the chain of thought as ordinary output.

`prompt_tokens` (budget.py) stays as it is: it is the estimate made before sending, used to
trim the prompt to fit.
"""

from __future__ import annotations

import argparse
import statistics
import threading
from pathlib import Path
from typing import Any

import dspy

# Steps in the order a call runs them, for the per-call table.
AGENTS = ("planner", "footing", "scam_caller", "legit_caller", "receiver", "judge")
_COUNTS = ("input", "output", "requests", "failed")

# Usage of attempts that raised, kept until the step succeeds on a retry:
# (call key, agent) -> usage. Locked: the studio runs calls in parallel threads.
_failed: dict[tuple[str, str], dict[str, int]] = {}
_failed_lock = threading.Lock()


def usage_key(call_type: str, seed: int, receiver_key: str) -> str:
    """Which call a step belongs to, so retries of parallel calls do not mix."""
    return f"{call_type}:{seed}:{receiver_key}"


def tally(tracker: Any) -> dict[str, int]:
    """input, output and requests of everything a dspy.track_usage() tracker saw."""
    entries = [entry for per_lm in tracker.usage_data.values() for entry in per_lm]
    return {
        "input": sum(entry.get("prompt_tokens") or 0 for entry in entries),
        "output": sum(entry.get("completion_tokens") or 0 for entry in entries),
        "requests": len(entries),
        "failed": 0,
    }


def keep_failed(key: str, agent: str, tracker: Any) -> None:
    """Remember a failed attempt's usage until the same step succeeds."""
    attempt = tally(tracker)
    attempt["failed"] = 1
    with _failed_lock:
        _failed[(key, agent)] = _add(_failed.get((key, agent), {}), attempt)


def step_usage(key: str, agent: str, tracker: Any, pred: dspy.Prediction) -> dict[str, Any]:
    """The successful attempt plus any failed ones before it, and the reasoning share."""
    with _failed_lock:
        earlier = _failed.pop((key, agent), {})
    usage: dict[str, Any] = _add(earlier, tally(tracker))
    share = reasoning_share(pred)
    if share is not None:
        usage["reasoning_share"] = share
    return usage


def reasoning_share(pred: dspy.Prediction) -> float | None:
    """The share of the answer's characters in its `reasoning` field, or None without one."""
    reasoning = str(pred.get("reasoning") or "")
    if not reasoning:
        return None
    total = sum(len(str(value)) for value in pred.values())
    return round(len(reasoning) / total, 2) if total else None


def _add(a: dict, b: dict) -> dict[str, int]:
    return {name: (a.get(name) or 0) + (b.get(name) or 0) for name in _COUNTS}


def call_usage(record: dict) -> dict[str, Any] | None:
    """Totals and a per-agent breakdown of one stored call, or None when it has no usage
    recorded (calls made before usage was kept).

    `rows` has one entry per agent that ran: agent, steps, requests, failed, input, output and
    reasoning (estimated output tokens spent on the chain of thought).
    """
    generation = record.get("generation", {})
    steps = [(turn.get("agent", ""), turn.get("usage")) for turn in generation.get("turns", [])]
    steps.append(("planner", (generation.get("briefing") or {}).get("usage")))
    steps.append(("footing", generation.get("footing_usage")))
    steps.append(("judge", (generation.get("judge") or {}).get("usage")))
    rows: dict[str, dict[str, Any]] = {}
    for agent, usage in steps:
        if not usage:
            continue
        row = rows.setdefault(agent, {"agent": agent, "steps": 0, "reasoning": 0})
        row["steps"] += 1
        row.update(_add(row, usage))
        row["reasoning"] += round(usage["output"] * usage.get("reasoning_share", 0))
    if not rows:
        return None
    ordered = sorted(
        rows.values(), key=lambda r: AGENTS.index(r["agent"]) if r["agent"] in AGENTS else 99
    )
    total = {"agent": "total", "steps": 0, "reasoning": 0}
    for row in ordered:
        total["steps"] += row["steps"]
        total["reasoning"] += row["reasoning"]
        total.update(_add(total, row))
    return {"rows": ordered, "total": total}


def summary(records: list[dict]) -> dict[str, Any]:
    """Token usage over many calls: per agent, for all of them together, and per call type.

    Only calls with usage count; `not_recorded` says how many were left out. Each agent row has
    the totals over the calls, the mean and max per call (over the calls the agent ran in),
    requests and failed attempts. `by_type` has the calls, mean, max and total tokens (in + out)
    per call of each call type.
    """
    usages: list[tuple[str, dict]] = []
    not_recorded = 0
    for record in records:
        usage = call_usage(record)
        if usage is None:
            not_recorded += 1
        else:
            usages.append((_call_type(record), usage))
    agents = [
        _agent_summary(agent, [r for _, u in usages for r in u["rows"] if r["agent"] == agent])
        for agent in AGENTS
    ]
    agents = [a for a in agents if a["calls"]]
    per_call = [u["total"]["input"] + u["total"]["output"] for _, u in usages]
    by_type = []
    for call_type in sorted({t for t, _ in usages}):
        tokens = [u["total"]["input"] + u["total"]["output"] for t, u in usages if t == call_type]
        by_type.append(
            {
                "call_type": call_type,
                "calls": len(tokens),
                "mean": round(statistics.mean(tokens)),
                "max": max(tokens),
                "total": sum(tokens),
            }
        )
    return {
        "calls": len(usages),
        "not_recorded": not_recorded,
        "agents": agents,
        "total": _agent_summary("total", [u["total"] for _, u in usages]),
        "per_call": {
            "mean": round(statistics.mean(per_call)) if per_call else 0,
            "max": max(per_call, default=0),
            "total": sum(per_call),
        },
        "by_type": by_type,
    }


def _agent_summary(agent: str, rows: list[dict]) -> dict[str, Any]:
    """One agent's rows (one per call) summed, with the mean and max per call."""
    ins, outs = [r["input"] for r in rows], [r["output"] for r in rows]
    return {
        "agent": agent,
        "calls": len(rows),
        "steps": sum(r["steps"] for r in rows),
        "input": sum(ins),
        "output": sum(outs),
        "input_mean": round(statistics.mean(ins)) if rows else 0,
        "output_mean": round(statistics.mean(outs)) if rows else 0,
        "input_max": max(ins, default=0),
        "output_max": max(outs, default=0),
        "reasoning": sum(r["reasoning"] for r in rows),
        "requests": sum(r["requests"] for r in rows),
        "failed": sum(r["failed"] for r in rows),
    }


def report(records: list[dict]) -> str:
    """Mean and max tokens per call and per agent, by call type, over calls with usage."""
    lines = []
    whole = summary(records)
    for row in whole["by_type"]:
        call_type = row["call_type"]
        s = summary([r for r in records if _call_type(r) == call_type])
        lines.append(
            f"{call_type}: {s['calls']} calls, tokens per call mean {s['per_call']['mean']:,}"
            f" max {s['per_call']['max']:,}"
        )
        lines.append(
            f"  {'agent':<13}{'in mean':>10}{'out mean':>10}{'in max':>9}{'out max':>9}"
            f"{'requests':>10}{'failed':>8}"
        )
        for a in s["agents"]:
            lines.append(
                f"  {a['agent']:<13}{a['input_mean']:>10,}{a['output_mean']:>10,}"
                f"{a['input_max']:>9,}{a['output_max']:>9,}{a['requests']:>10}{a['failed']:>8}"
            )
        lines.append(f"  failed attempts (retried): {s['total']['failed']}")
    if whole["not_recorded"]:
        lines.append(
            f"{whole['not_recorded']} calls have no usage recorded (made before it was kept)"
        )
    return "\n".join(lines) if lines else "no calls"


def _call_type(record: dict) -> str:
    """A stored record keeps its type in generation.call; the test records keep it on top."""
    call = record.get("generation", {}).get("call") or {}
    return call.get("call_type") or record.get("call_type") or "?"


def main(argv: list[str] | None = None) -> None:
    from .store import DEFAULT_DB, Store

    parser = argparse.ArgumentParser(prog="python -m data_pipeline.usage")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="tokens per call and per agent")
    rep.add_argument("--type", default="", help="only this call type")
    rep.add_argument("--since", default="", help="only calls created on or after this date")
    rep.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"default: {DEFAULT_DB}")
    args = parser.parse_args(argv)
    store = Store(args.db, read_only=True)
    records = [r for r in store.call_dicts(args.type) if r.get("created_at", "") >= args.since]
    print(report(records))


if __name__ == "__main__":
    main()
