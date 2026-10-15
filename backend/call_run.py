"""State of one simulated call, built from the updates CallSessions.simulate_call yields.

RunView only collects what the graph produced (turn text, entities, decisions, timings) and
shapes it for the templates. The scam decision itself always comes from the orchestrator.
Read RunView first, then Run and RunRegistry, then iter_updates.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.graph import verdict_from

from . import runtime
from .views import Band, confidence_word, diarizer_label, fmt_seconds, type_name, who

TERMINAL_NODES = frozenset({"escalate", "speak"})


@dataclass
class TurnView:
    """One turn of the call and what the graph decided about it so far."""

    n: int  # 1-based
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
        """ "You" for the person called, "Caller" for everyone else."""
        return who(self.speaker)

    @property
    def height(self) -> int:
        """Bar height in percent of the well for a normal turn: longer turns read taller."""
        return 28 + min(len(self.text), 100) * 3 // 10


@dataclass
class Stage:
    """One row of the four-step pipeline list."""

    n: int
    name: str
    desc: str
    state: str  # idle | busy | done | safe | scam
    right: str  # time taken, or a short note


@dataclass
class Column:
    """One bar of the turn strip."""

    n: int
    who: str
    state: str  # idle | live | normal | scam | held
    height: int


@dataclass
class LogEntry:
    """A run of turns with the same outcome in the decision log."""

    kind: str  # ok | alarm
    text: str
    outcome: str
    first: int
    last: int

    @property
    def range(self) -> str:
        if self.first == self.last:
            return f"T{self.first}"
        return f"T{self.first}–T{self.last}"


class RunView:
    """Everything the result regions of the call page show, collected as updates arrive."""

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

    def set_transcript(self, transcript: Any, seconds: float) -> None:
        """Fill in the turns from the ASR result. `seconds` is the time ASR took."""
        self.model = transcript.model
        self.asr_seconds = seconds
        self.turns = [
            TurnView(i + 1, t.speaker, t.start, t.end, t.text)
            for i, t in enumerate(transcript.turns)
        ]

    def feed(self, item: dict) -> None:
        """Record one graph update: {"turn_index", "node", "update"}."""
        turn = self.turns[item["turn_index"]]
        node, update = item["node"], item["update"]
        turn.timings = update.get("timings", turn.timings)
        if node == "remember":
            self._remember(turn, update)
        elif node == "analyze":
            self._analyze(turn, update)
        elif node == "decide":
            self._decide(turn, update)

    def finish(self) -> None:
        self.done = True

    def take_fresh(self) -> bool:
        """True once, in the first render after the escalation was found (the one animation)."""
        fresh, self.fresh_pending = self.fresh_pending, False
        return fresh

    def _remember(self, turn: TurnView, update: dict) -> None:
        turn.status = "reading"
        turn.text = update["turn"].text  # the cleaned text

    def _analyze(self, turn: TurnView, update: dict) -> None:
        nlp = update["nlp"]
        known = {(e.text, e.label) for e in self.entities}
        self.entities += [e for e in nlp.entities if (e.text, e.label) not in known]
        self.summary = nlp.summary
        verdict = verdict_from(nlp)
        turn.level = verdict["level"]
        turn.label_type = verdict["label_type"]
        turn.confidence = verdict["confidence"]

    def _decide(self, turn: TurnView, update: dict) -> None:
        decision = update["decision"]
        turn.reason = decision.reason
        if decision.auto_reply:
            turn.status = "normal"
        elif self.esc_turn is None:  # the first turn that is not answered is the escalation
            turn.status = "scam"
            self.esc_turn = turn.n
            self.esc_level = turn.level if turn.level != "safe" else "suspicious"
            self.esc_type, self.esc_confidence = turn.label_type, turn.confidence
            self.fresh_pending = True
        else:
            turn.status = "held"

    @property
    def total(self) -> int:
        return len(self.turns)

    @property
    def decided(self) -> int:
        return sum(1 for t in self.turns if t.status in {"normal", "scam", "held"})

    @property
    def model_label(self) -> str:
        if self.demo:
            return "Scripted demo transcript"
        return "Typhoon ASR, offline" if self.model in {"", "typhoon_asr"} else self.model

    def stage_time(self, node: str) -> float:
        return sum(t.timings.get(node, 0.0) for t in self.turns)

    def stages(self) -> list[Stage]:
        """The four pipeline steps with their state and time."""
        have_turns = bool(self.turns)
        running = not self.done and self.error is None
        has_decisions = self.decided > 0
        decide_state, decide_desc = self._decide_stage(running)
        return [
            Stage(
                1,
                "Listen",
                self.model_label,
                "done" if have_turns else "busy",
                fmt_seconds(self.asr_seconds) if have_turns else "",
            ),
            Stage(
                2,
                "Split speakers",
                diarizer_label(),
                "done" if have_turns else "busy",
                "with Listen" if have_turns else "",
            ),
            Stage(
                3,
                "Read",
                "Entities · summary · scam check, per turn",
                ("busy" if running else "done") if have_turns else "idle",
                fmt_seconds(self.stage_time("analyze")) if has_decisions else "",
            ),
            Stage(
                4,
                "Decide",
                decide_desc,
                decide_state,
                fmt_seconds(self.stage_time("decide")) if has_decisions else "",
            ),
        ]

    def _decide_stage(self, running: bool) -> tuple[str, str]:
        """State and description of the "Decide" step."""
        if self.esc_turn is not None:
            return "scam", f"Fixed rules. Escalated at turn {self.esc_turn}."
        if self.decided > 0:
            return "safe", (
                "Fixed rules. Reply allowed so far." if running else "Fixed rules. No escalation."
            )
        return "idle", "Fixed rules. Reply or warn."

    def columns(self) -> list[Column]:
        """The turn strip: one bar per turn."""
        shown_as = {"pending": "idle", "reading": "live"}
        return [Column(t.n, t.who, shown_as.get(t.status, t.status), t.height) for t in self.turns]

    def band(self, fresh: bool = False) -> Band | None:
        """The verdict band, or None until the first turn is decided."""
        if self.esc_turn is not None:
            return self._escalated_band(fresh)
        if self.decided == 0:
            return None
        return self._normal_band()

    def _escalated_band(self, fresh: bool) -> Band:
        confidence = confidence_word(self.esc_confidence)
        is_scam = self.esc_level == "scam"
        return Band(
            kind="scam" if is_scam else "unsure",
            word="SCAM" if is_scam else "UNSURE",
            subtype=type_name(self.esc_type) if is_scam else "Might be a scam. Type not known",
            instruction="Warn the user. No reply was spoken.",
            fact_label="Escalated at",
            fact_big=f"Turn {self.esc_turn}",
            fact_note=f"of {self.total}. Stays escalated until the call ends.",
            fact_extra=f"Confidence: {confidence.lower()}",
            inline_fact=(
                f"Escalated at turn {self.esc_turn} of {self.total}. Callas stopped replying."
            ),
            fresh=fresh,
        )

    def _normal_band(self) -> Band:
        count = self.decided
        turns = "turn" if count == 1 else "turns"
        if self.done:
            note = "The call is finished. No scam signs found."
        else:
            note = f"Turn {count + 1} is being read now. The verdict can still change."
        return Band(
            kind="safe",
            word="NORMAL",
            subtype=f"No scam signs in {count} {turns}",
            instruction="Callas may answer. Every reply passes the leak check first.",
            fact_label="Checked",
            fact_big=f"{count} {turns}",
            fact_note=note,
            inline_fact=note,
        )

    def log(self) -> list[LogEntry]:
        """Decision log: runs of turns with the same outcome, e.g. T1-T3 normal."""
        entries: list[LogEntry] = []
        for turn in self.turns:
            outcome = _log_outcome(turn)
            if outcome is None:
                continue
            kind, text, result = outcome
            if entries and (entries[-1].kind, entries[-1].text) == (kind, text):
                entries[-1].last = turn.n
            else:
                entries.append(LogEntry(kind, text, result, first=turn.n, last=turn.n))
        return entries


def _log_outcome(turn: TurnView) -> tuple[str, str, str] | None:
    """(kind, text, outcome) of a decided turn for the decision log, None while undecided."""
    if turn.status == "normal":
        return "ok", "normal", "reply allowed"
    if turn.status == "held":
        return "ok", "already escalated", "no reply"
    if turn.status == "scam":
        if "low-confidence" in turn.reason:
            return "alarm", "scam, low confidence", "escalate for review"
        return "alarm", "scam, high confidence", "escalate"
    return None


@dataclass
class Run:
    """One uploaded recording: its temp file, and the state of its event stream."""

    id: str
    path: str | None
    filename: str
    # A lambda, not time.monotonic itself: tests replace time.monotonic and it must be looked
    # up at call time.
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
        for run_id, run in list(self._runs.items()):
            if run.finished and now - run.finished_at > self.FINISHED_TTL:
                self.drop(run_id)
            elif not run.started and now - run.created > self.PENDING_TTL:
                self.drop(run_id)

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


async def iter_updates(
    app_runtime, path: str, stop: threading.Event
) -> AsyncIterator[tuple[str, Any]]:
    """Run CallSessions.simulate_call in one worker thread and yield what it produces.

    Yields ("transcript", (Transcript, seconds)) once, then ("item", update) per node, then
    ("end", None). A failure arrives as ("error", exception). Setting `stop` makes the worker
    thread stop after its current update, which ends the call session.

    The transcript is not part of the graph updates. runtime.py wraps the ASR function and
    stores its result in a thread-local, so _simulate must read it on the worker thread.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(kind: str, value: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

    worker = threading.Thread(
        target=_simulate, args=(app_runtime, path, stop, put), name="simulate-call", daemon=True
    )
    worker.start()
    try:
        while True:
            kind, value = await queue.get()
            yield kind, value
            if kind in {"end", "error"}:
                return
    finally:
        stop.set()


def _simulate(
    app_runtime, path: str, stop: threading.Event, put: Callable[[str, Any], None]
) -> None:
    """Worker thread of iter_updates: run the call and hand every update to `put`."""
    transcript_sent = False
    updates = None
    try:
        updates = app_runtime.sessions.simulate_call(path)
        for item in updates:
            if not transcript_sent:  # ASR has run by the time the first update arrives
                put("transcript", runtime.last_transcription())
                transcript_sent = True
            if stop.is_set():
                break
            put("item", item)
        if not transcript_sent:
            put("transcript", runtime.last_transcription())
        put("end", None)
    except Exception as exc:
        put("error", exc)
    finally:
        if updates is not None:
            updates.close()  # ends the call session even when stopped early
