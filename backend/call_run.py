"""State of one simulated call, built from the updates CallSessions.simulate_call yields.

RunView only collects what the graph produced (turn text, entities, decisions, timings) and
shapes it for the templates. The scam decision itself always comes from the orchestrator.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.graph import verdict_from

from . import runtime
from .views import (
    Band,
    confidence_word,
    fmt_seconds,
    fmt_time,
    type_name,
    who,
)

TERMINAL_NODES = frozenset({"escalate", "speak"})


@dataclass
class TurnView:
    n: int
    speaker: str
    start: float
    end: float
    text: str
    status: str = "pending"  # pending | reading | normal | scam | held
    level: str | None = None
    label_type: str | None = None
    confidence: float = 0.0
    reason: str = ""
    timings: dict = field(default_factory=dict)

    @property
    def who(self) -> str:
        return who(self.speaker)

    @property
    def height(self) -> int:
        """Bar height in percent of the well for a normal turn: longer turns read taller."""
        return 28 + min(len(self.text), 100) * 3 // 10


class RunView:
    def __init__(self, filename: str, demo: bool) -> None:
        self.filename = filename
        self.demo = demo
        self.model = ""
        self.asr_seconds: float | None = None
        self.turns: list[TurnView] = []
        self.entities: list[Any] = []
        self.summary = ""
        self.esc_turn: int | None = None
        self.esc_level: str | None = None
        self.esc_type: str | None = None
        self.esc_confidence = 0.0
        self.done = False
        self.error: str | None = None
        self.fresh_pending = False

    # ---- collecting -------------------------------------------------------------------

    def set_transcript(self, transcript: Any, seconds: float) -> None:
        self.model = transcript.model
        self.asr_seconds = seconds
        self.turns = [
            TurnView(i + 1, t.speaker, t.start, t.end, t.text)
            for i, t in enumerate(transcript.turns)
        ]

    def feed(self, item: dict) -> None:
        turn = self.turns[item["turn_index"]]
        node, update = item["node"], item["update"]
        turn.timings = update.get("timings", turn.timings)
        if node == "remember":
            turn.status = "reading"
            turn.text = update["turn"].text  # the cleaned text
        elif node == "analyze":
            nlp = update["nlp"]
            keys = {(e.text, e.label) for e in self.entities}
            self.entities += [e for e in nlp.entities if (e.text, e.label) not in keys]
            self.summary = nlp.summary
            verdict = verdict_from(nlp)
            turn.level, turn.label_type, turn.confidence = (
                verdict["level"],
                verdict["label_type"],
                verdict["confidence"],
            )
        elif node == "decide":
            decision = update["decision"]
            turn.reason = decision.reason
            if decision.auto_reply:
                turn.status = "normal"
            elif self.esc_turn is None:
                turn.status = "scam"
                self.esc_turn = turn.n
                self.esc_level = turn.level if turn.level != "safe" else "suspicious"
                self.esc_type, self.esc_confidence = turn.label_type, turn.confidence
                self.fresh_pending = True
            else:
                turn.status = "held"

    def finish(self) -> None:
        self.done = True

    def take_fresh(self) -> bool:
        """True once, in the first render after the escalation was found (the one animation)."""
        fresh, self.fresh_pending = self.fresh_pending, False
        return fresh

    # ---- shaping ----------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.turns)

    @property
    def decided(self) -> int:
        return sum(1 for t in self.turns if t.status in {"normal", "scam", "held"})

    @property
    def speakers(self) -> int:
        return len({t.speaker for t in self.turns})

    @property
    def meta(self) -> str:
        if not self.turns:
            return "Listening to the recording"
        parts = [fmt_time(self.turns[-1].end), f"{self.speakers} speakers"]
        if self.demo:
            parts.append("synthetic demo call")
        return " · ".join(parts)

    @property
    def model_label(self) -> str:
        if self.demo:
            return "Scripted demo transcript"
        return "Typhoon ASR, offline" if self.model in {"", "typhoon_asr"} else self.model

    def stage_time(self, node: str) -> float:
        return sum(t.timings.get(node, 0.0) for t in self.turns)

    def stages(self) -> list[dict]:
        have = bool(self.turns)
        running = not self.done and self.error is None
        decided = self.decided > 0
        if self.esc_turn is not None:
            decide_state = "scam"
            decide_desc = f"Fixed rules. Escalated at turn {self.esc_turn}."
        elif decided:
            decide_state = "safe"
            decide_desc = (
                "Fixed rules. Reply allowed so far." if running else "Fixed rules. No escalation."
            )
        else:
            decide_state, decide_desc = "idle", "Fixed rules. Reply or warn."
        return [
            {
                "n": 1,
                "name": "Listen",
                "desc": self.model_label,
                "state": "done" if have else "busy",
                "right": fmt_seconds(self.asr_seconds) if have else "",
            },
            {
                "n": 2,
                "name": "Split speakers",
                "desc": "pyannote segmentation 3.0",
                "state": "done" if have else "busy",
                "right": "with Listen" if have else "",
            },
            {
                "n": 3,
                "name": "Read",
                "desc": "Entities · summary · scam check, per turn",
                "state": ("busy" if running else "done") if have else "idle",
                "right": fmt_seconds(self.stage_time("analyze")) if decided else "",
            },
            {
                "n": 4,
                "name": "Decide",
                "desc": decide_desc,
                "state": decide_state,
                "right": fmt_seconds(self.stage_time("decide")) if decided else "",
            },
        ]

    def columns(self) -> list[dict]:
        cols = []
        for t in self.turns:
            state = {"pending": "idle", "reading": "live"}.get(t.status, t.status)
            cols.append({"n": t.n, "who": t.who, "state": state, "height": t.height})
        return cols

    def band(self, fresh: bool = False) -> Band | None:
        if self.esc_turn is not None:
            conf = confidence_word(self.esc_confidence)
            kind = "scam" if self.esc_level == "scam" else "unsure"
            return Band(
                kind,
                "SCAM" if kind == "scam" else "UNSURE",
                type_name(self.esc_type) if kind == "scam" else "Might be a scam. Type not known",
                "Warn the user. No reply was spoken.",
                "Escalated at",
                f"Turn {self.esc_turn}",
                f"of {self.total}. Stays escalated until the call ends.",
                fact_extra=f"Confidence: {conf.lower()}",
                inline_fact=(
                    f"Escalated at turn {self.esc_turn} of {self.total}. Callas stopped replying."
                ),
                fresh=fresh,
            )
        if self.decided == 0:
            return None
        n = self.decided
        plural = "turn" if n == 1 else "turns"
        if self.done:
            note = "The call is finished. No scam signs found."
        else:
            note = f"Turn {n + 1} is being read now. The verdict can still change."
        return Band(
            "safe",
            "NORMAL",
            f"No scam signs in {n} {plural}",
            "Callas may answer. Every reply passes the leak check first.",
            "Checked",
            f"{n} {plural}",
            note,
            inline_fact=note,
        )

    def log(self) -> list[dict]:
        """Decision log: runs of turns with the same outcome, e.g. T1-T3 normal."""
        entries: list[dict] = []
        for t in self.turns:
            if t.status == "normal":
                kind, text = "ok", "normal"
                outcome = "reply allowed"
            elif t.status == "scam":
                kind = "alarm"
                text = (
                    "scam, high confidence"
                    if "low-confidence" not in t.reason
                    else "scam, low confidence"
                )
                outcome = "escalate" if "low-confidence" not in t.reason else "escalate for review"
            elif t.status == "held":
                kind, text, outcome = "ok", "already escalated", "no reply"
            else:
                continue
            if entries and entries[-1]["kind"] == kind and entries[-1]["text"] == text:
                entries[-1]["last"] = t.n
            else:
                entries.append(
                    {"kind": kind, "text": text, "outcome": outcome, "first": t.n, "last": t.n}
                )
        for e in entries:
            e["range"] = (
                f"T{e['first']}" if e["first"] == e["last"] else f"T{e['first']}–T{e['last']}"
            )
        return entries


# ---- running the call ---------------------------------------------------------------------


@dataclass
class Run:
    id: str
    path: str | None
    filename: str
    created: float = field(default_factory=lambda: time.monotonic())
    started: bool = False
    finished: bool = False
    finished_at: float = 0.0
    snapshot: dict[str, str] = field(default_factory=dict)


class RunRegistry:
    """Pending and finished runs. A finished run's snapshot holds call text (the rendered
    transcript), so it lives only long enough for a browser reconnect, then it is dropped
    with its temp file. Call memory is deleted when the call ends."""

    PENDING_TTL = 300.0  # an upload whose event stream never opened
    FINISHED_TTL = 60.0  # reconnect window for a finished run's final picture

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    def sweep(self) -> None:
        now = time.monotonic()
        for rid, r in list(self._runs.items()):
            if r.finished and now - r.finished_at > self.FINISHED_TTL:
                self.drop(rid)
            elif not r.started and now - r.created > self.PENDING_TTL:
                self.drop(rid)

    def create(self, path: str | None, filename: str) -> Run:
        self.sweep()
        run = Run(uuid.uuid4().hex, path, filename)
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        self.sweep()
        return self._runs.get(run_id)

    def drop(self, run_id: str) -> None:
        run = self._runs.pop(run_id, None)
        if run:
            run.snapshot = {}
        if run and run.path:
            Path(run.path).unlink(missing_ok=True)


async def iter_updates(app_rt, path: str, stop: threading.Event) -> AsyncIterator[tuple[str, Any]]:
    """Run CallSessions.simulate_call in one worker thread and yield what it produces.

    Yields ("transcript", (Transcript, seconds)) once, then ("item", update) per node, then
    ("end", None). A failure arrives as ("error", exception). Setting `stop` ends the
    generator early, which ends the call session.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(kind: str, value: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

    def work() -> None:
        sent = False
        gen = None
        try:
            gen = app_rt.sessions.simulate_call(path)
            for item in gen:
                if not sent:
                    put("transcript", runtime.last_transcription())
                    sent = True
                if stop.is_set():
                    break
                put("item", item)
            if not sent:
                put("transcript", runtime.last_transcription())
            put("end", None)
        except Exception as exc:
            put("error", exc)
        finally:
            if gen is not None:
                gen.close()  # ends the call session even when stopped early

    thread = threading.Thread(target=work, name="simulate-call", daemon=True)
    thread.start()
    try:
        while True:
            kind, value = await queue.get()
            yield kind, value
            if kind in {"end", "error"}:
                return
    finally:
        stop.set()
