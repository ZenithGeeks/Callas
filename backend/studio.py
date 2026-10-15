"""The /studio page: make a simulated call from the browser and watch it being written.

The form starts a background thread that runs the simulator's LangGraph graph
(data_pipeline.live.stream_call). Each step is rendered as an HTML piece and kept in the run's
event list; /studio/{run}/events sends the list to the browser as server-sent events and sends
new pieces as they arrive. The browser never has to stay connected: the thread finishes the call
and saves it to the same store /dataset reads, whether or not anyone is watching.

One run at a time per server. A second Start gets a 409 with a message. The Stop button asks the
thread to stop after the step it is working on; the call stays in the store as unfinished.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import dspy
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from data_pipeline import live
from data_pipeline.config import BENIGN, CALL_TYPES, Config, ConfigError, ReceiverProfile
from data_pipeline.config import load_config as load_default_config
from data_pipeline.store import Store
from llm_serving.client import LLMUnavailable

from . import dataset, views
from .web import form_error, page, render

log = logging.getLogger("callas.studio")
SHUTDOWN = threading.Event()  # set when the app stops, so open event streams end


@asynccontextmanager
async def _lifespan(app: Any) -> AsyncIterator[None]:
    """Runs with the app's own lifespan (include_router merges it)."""
    SHUTDOWN.clear()
    try:
        yield
    finally:
        SHUTDOWN.set()
        for run in BOARD.all():
            run.cancel.set()  # a running call stops after its step; it stays saved and resumable


router = APIRouter(lifespan=_lifespan)

RUN_ID_RE = re.compile(r"[0-9a-f]{12}")
KEPT_RUNS = 5
CUSTOM = "__custom__"
CUSTOM_KEY = "custom"  # the key in the call id of a call with a described receiver
MAX_SEED = 99_999
MAX_TEXT = 200

# Form field -> (label, ReceiverProfile pool). The pool fills the suggestions list.
DETAIL_FIELDS = [
    ("occupation", "Job", "occupations"),
    ("education", "Education", "education"),
    ("region", "Lives in", "regions"),
    ("family", "Family", "family"),
    ("digital_literacy", "Phone use", "digital_literacy"),
    ("scam_awareness", "Knows about scams", "scam_awareness"),
    ("temperament", "Temperament", "temperaments"),
    ("ending", "When unsure", "endings"),
    ("speech", "Talks", "speech"),
]


class FormProblem(Exception):
    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


# ---- runs ------------------------------------------------------------------------------------


@dataclass
class StudioRun:
    id: str
    spec: live.CallSpec
    title: str
    events: list[tuple[str, str]] = field(default_factory=list)  # (event name, HTML)
    finished: bool = False
    started: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)

    def push(self, name: str, html: str) -> None:
        self.events.append((name, html))  # list.append is atomic; readers only index it


class Board:
    """The runs of this server: at most one running, the last few kept so a page can replay."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, StudioRun] = {}

    def start(self, make) -> StudioRun:
        """Register the run `make(run_id)` builds, unless one is already running."""
        with self._lock:
            if (busy := self.running()) is not None:
                raise FormProblem(
                    f"A call is already being written: {busy.title}, running for "
                    f"{_elapsed(time.monotonic() - busy.started)}. Wait for it to finish, or "
                    "press Stop on it. Stop works between steps, so a model call that is stuck "
                    "keeps the run going until the model server times out.",
                    status=409,
                )
            run = make(uuid.uuid4().hex[:12])
            self._runs[run.id] = run
            for old in list(self._runs)[:-KEPT_RUNS]:
                del self._runs[old]
            return run

    def all(self) -> list[StudioRun]:
        return list(self._runs.values())

    def running(self) -> StudioRun | None:
        return next((r for r in self._runs.values() if not r.finished), None)

    def get(self, run_id: str) -> StudioRun | None:
        return self._runs.get(run_id) if RUN_ID_RE.fullmatch(run_id) else None


BOARD = Board()


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


@cache
def _config() -> Config:
    return load_default_config()


def _demo(request: Request) -> bool:
    return request.app.state.rt.is_demo


def _type_options() -> list[tuple[str, str]]:
    return [(key, dataset.CALL_TYPE_NAMES[key]) for key in CALL_TYPES]


def _profile_label(profile: ReceiverProfile) -> str:
    return f"{profile.key}: {profile.gender}, {profile.age}, {profile.occupation}"


# ---- the page and the form -------------------------------------------------------------------


@router.get("/studio", response_class=HTMLResponse)
def studio_page(request: Request) -> HTMLResponse:
    config = _config()
    pools = config.scenarios.receivers
    scenarios = [
        (s.key, call_type, f"{s.key}: {s.role}")
        for call_type in CALL_TYPES
        for s in (
            config.scenarios.legit if call_type == BENIGN else config.scenarios.scams[call_type]
        )
    ]
    run = BOARD.running()
    return page(
        request,
        "studio.html",
        "studio",
        types=_type_options(),
        scenarios=scenarios,
        profiles=[(p.key, _profile_label(p)) for p in pools.profiles],
        genders=pools.genders,
        detail_fields=[(key, label, getattr(pools, pool)) for key, label, pool in DETAIL_FIELDS],
        age_range=pools.age,
        run_view=_run_view(run) if run else None,
        db=str(dataset.db_path()),
    )


def parse_form(form: Any, config: Config, taken_ids: set[str]) -> live.CallSpec:
    """Check the submitted form and turn it into a CallSpec. Raises FormProblem."""
    call_type = str(form.get("call_type", ""))
    if call_type not in CALL_TYPES:
        raise FormProblem("Choose the kind of call.")
    scenario = None
    if key := str(form.get("scenario", "")):
        pool = config.scenarios.legit if call_type == BENIGN else config.scenarios.scams[call_type]
        scenario = next((s for s in pool if s.key == key), None)
        if scenario is None:
            raise FormProblem("That scenario does not belong to this kind of call.")
    receiver = _parse_receiver(form, config)
    name = _clean(form.get("receiver_name", ""))
    seed_text = str(form.get("seed", "")).strip()
    if seed_text:
        if not (seed_text.isascii() and seed_text.isdigit()) or int(seed_text) > MAX_SEED:
            raise FormProblem(f"The seed is a whole number from 0 to {MAX_SEED}, or empty.")
        seed = int(seed_text)
        spec = live.CallSpec(call_type, seed, scenario, receiver, name)
        if spec.record_id in taken_ids:
            raise FormProblem(
                f"{spec.record_id} is already in the dataset. "
                "Choose another seed, or leave the seed empty to take the next free one."
            )
        return spec
    seed = live.next_free_seed(call_type, taken_ids, receiver)
    return live.CallSpec(call_type, seed, scenario, receiver, name)


def _clean(value: Any) -> str:
    """One line of text for a prompt: control and format characters are dropped, any run of
    white space (including line breaks) becomes one space. Thai letters are kept."""
    text = "".join(
        c for c in str(value) if unicodedata.category(c) not in ("Cc", "Cf") or c.isspace()
    )
    return " ".join(text.split())[:MAX_TEXT]


def _parse_receiver(form: Any, config: Config) -> ReceiverProfile | None:
    choice = str(form.get("receiver", ""))
    if not choice:
        return None
    if choice != CUSTOM:
        profile = next((p for p in config.scenarios.receivers.profiles if p.key == choice), None)
        if profile is None:
            raise FormProblem("That saved person does not exist.")
        return profile
    pools = config.scenarios.receivers
    values = {key: _clean(form.get(f"r_{key}", "")) for key, _, _ in DETAIL_FIELDS}
    gender = str(form.get("r_gender", ""))
    if gender not in pools.genders:
        raise FormProblem(f"Gender is one of: {', '.join(pools.genders)}.")
    if missing := [label for key, label, _ in DETAIL_FIELDS if not values[key]]:
        raise FormProblem(f"Fill in every detail about the person. Missing: {', '.join(missing)}.")
    try:
        return ReceiverProfile(
            key=CUSTOM_KEY, gender=gender, age=int(str(form.get("r_age", "")) or 0), **values
        )
    except (ValidationError, ValueError):
        raise FormProblem("The age is a whole number from 10 to 100.") from None


def _taken_ids() -> set[str]:
    """Calls that cannot be started again: the finished ones. An unfinished call (a stopped
    run) can be resumed with the same seed; the store checks that the receiver matches."""
    db = dataset.store()
    return {r["id"] for r in db.summaries() if r["status"] != "in_progress"} if db else set()


@router.post("/studio/start", response_class=HTMLResponse)
async def studio_start(request: Request) -> Response:
    form = await request.form()  # the store and the templates are used in a worker thread
    return await run_in_threadpool(_begin, form, _demo(request), request.app.state.pace)


def _begin(form: Any, demo: bool, pace: float) -> Response:
    try:
        spec = parse_form(form, _config(), _taken_ids())
        run = BOARD.start(lambda run_id: StudioRun(run_id, spec, _title(spec)))
    except FormProblem as problem:
        return form_error(str(problem), problem.status)
    threading.Thread(
        target=_work,
        args=(run, demo, pace, bool(form.get("speak")), dataset.db_path()),
        name=f"studio-{run.id}",
        daemon=True,
    ).start()
    return HTMLResponse(render("partials/studio_run.html", run=_run_view(run), demo=demo))


def _title(spec: live.CallSpec) -> str:
    return f"{dataset.CALL_TYPE_NAMES[spec.call_type]} #{spec.seed}"


def _run_view(run: StudioRun) -> dict:
    spec = run.spec
    return {
        "id": run.id,
        "title": _title(spec),
        "record_id": spec.record_id,
        "label_scam": spec.call_type != BENIGN,
        "receiver": spec.receiver,
        "receiver_name": spec.receiver_name,
    }


# ---- server-sent events ----------------------------------------------------------------------


def _sse(name: str, html: str, event_id: int) -> str:
    data = "\n".join(f"data: {line}" for line in html.splitlines() or [""])
    return f"id: {event_id}\nevent: {name}\n{data}\n\n"


@router.get("/studio/{run_id}/events")
async def studio_events(request: Request, run_id: str) -> Response:
    run = BOARD.get(run_id)
    if run is None:
        return JSONResponse({"detail": "Unknown run."}, status_code=404)
    last = request.headers.get("last-event-id", "")
    first = int(last) + 1 if last.isdigit() else 0

    async def stream() -> AsyncIterator[str]:
        yield "retry: 3000\n\n"
        sent, quiet = first, 0
        while True:
            while sent < len(run.events):
                name, html = run.events[sent]
                yield _sse(name, html, sent)
                sent, quiet = sent + 1, 0
            if (run.finished and sent >= len(run.events)) or SHUTDOWN.is_set():
                return
            if await request.is_disconnected():
                return  # the call keeps running and is saved; a new page can replay it
            await asyncio.sleep(0.1)
            quiet += 1
            if quiet % 150 == 0:
                yield ": still running\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get("/studio/{run_id}/state")
def studio_state(run_id: str) -> Response:
    """Lets the page tell a lost connection from a run the server no longer knows."""
    run = BOARD.get(run_id)
    if run is None:
        return JSONResponse({"detail": "Unknown run."}, status_code=404)
    return JSONResponse({"finished": run.finished})


@router.post("/studio/{run_id}/stop")
def studio_stop(run_id: str) -> Response:
    run = BOARD.get(run_id)
    if run is None:
        return JSONResponse({"detail": "Unknown run."}, status_code=404)
    run.cancel.set()
    return Response(status_code=204)


# ---- the background thread -------------------------------------------------------------------


def _work(run: StudioRun, demo: bool, pace: float, speak: bool, db_path: Path) -> None:
    """Run the call and push one HTML piece per step. Never raises: problems become an event."""
    outcome = "stopped"
    try:
        store = Store(db_path)
        config = _config()
        if demo:
            agents = live.fake_agents(config, run.spec.call_type, delay=2 * pace)
            voices = live.SilentVoices() if speak else None
        else:
            from data_pipeline.agents import Agents
            from data_pipeline.tts import Voices

            _sync_wiki(store)
            agents = Agents(config)
            voices = Voices() if speak else None
        capture = live.Capture(agents)
        with dspy.context(callbacks=[capture]):
            for event in live.stream_call(
                run.spec, store, agents, config, voices, capture, run.cancel
            ):
                run.push(event.kind, _render_event(run, event))
                if event.kind == "verdict":
                    outcome = "finished"
    except Exception as exc:
        outcome = "failed"
        run.push("failed", render("partials/studio_failed.html", message=_explain(run, exc)))
    finally:
        try:
            html = render("partials/studio_done.html", run=_run_view(run), outcome=outcome)
        except Exception:
            log.exception("studio run %s: could not render the last message", run.id)
            html = "<p>The run has ended.</p>"
        run.push("done", html)
        run.finished = True  # last: a new run may start from here on


def _sync_wiki(store: Store) -> None:
    from data_pipeline.wiki import DEFAULT_WIKI, WikiError, sync

    if DEFAULT_WIKI.is_dir():
        try:
            sync(store, DEFAULT_WIKI)
        except WikiError as exc:
            log.warning("wiki sync skipped: %s", exc)


def _explain(run: StudioRun, exc: Exception) -> str:
    """A message for the page. The call stays in the store; model errors reject it like the CLI."""
    if isinstance(exc, LLMUnavailable):
        return f"{exc}. The call is saved as far as it got; start it again with the same seed."
    if isinstance(exc, ConfigError):
        return f"Config problem: {exc}"
    if isinstance(exc, live.CallConflict):
        return str(exc)
    log.exception("studio run %s failed", run.id)
    try:
        live.mark_model_error(Store(dataset.db_path()), run.spec.record_id, exc)
    except Exception:
        log.exception("could not mark %s as rejected", run.spec.record_id)
    return (
        f"The call stopped with an error ({type(exc).__name__}). "
        "What was written so far is in the dataset."
    )


# ---- turning steps into HTML -----------------------------------------------------------------


def _tag(modules: list[live.ModuleRun], agent: str, fallback_s: float | None = None) -> dict | None:
    """The small label under a message: agent, DSPy module, model, time taken."""
    if modules:
        main = modules[-1]
        return {
            "agent": main.agent,
            "module": main.module,
            "model": main.model,
            "seconds": main.seconds,
            "calls": [_lm_view(c) for m in modules for c in m.lm_calls],
        }
    if fallback_s is not None:  # a resumed turn: no callbacks, only the stored time
        return {"agent": agent, "module": "", "model": "", "seconds": fallback_s, "calls": []}
    return None


def _lm_view(call: live.LmCall) -> dict:
    return {
        "model": call.model,
        "messages": call.messages,
        "answer": call.answer,
        "error": call.error,
    }


def _flags(turn: dict) -> list[str]:
    flags = [dataset.reason_text(p) for p in turn.get("problems", [])]
    words = [w for w in turn.get("spelling", []) if len(w.strip()) > 1]
    if words:
        flags.append(f"Check spelling: {', '.join(words[:3])}")
    return flags


def _leaked(turn: dict) -> str:
    """What the turn leaked, if the turn says so. Detection is not done here."""
    labels = turn.get("leaked") or []  # label names, e.g. ["OTP", "ID_CARD"]
    return ", ".join(views.LABEL_TEXT.get(label, label) for label in labels)


def _audio_url(record_id: str, turn: dict) -> str | None:
    return f"/dataset/audio/{record_id}/{Path(turn['audio']).name}" if turn.get("audio") else None


def _render_event(run: StudioRun, event: live.Event) -> str:
    record, generation = event.record, event.record["generation"]
    if event.kind == "thinking":
        return render(
            "partials/studio_thinking.html",
            agent=event.agent,
            n=len(generation["turns"]) + 1,
        )
    if event.kind == "plan":
        briefing = generation["briefing"]
        return render(
            "partials/studio_plan.html",
            plan=briefing["plan"],
            notes=briefing["notes_used"],
            reasoning=briefing.get("reasoning", ""),
            tag=_tag(event.modules, "planner"),
        )
    if event.kind == "turn":
        turn = generation["turns"][event.index]
        return render(
            "partials/studio_turn.html",
            n=event.index + 1,
            turn=turn,
            side="left" if turn["speaker"] == "A" else "right",
            who=dataset.SPEAKER_NAMES[turn["speaker"]],
            flags=_flags(turn),
            leaked=_leaked(turn),
            audio=_audio_url(record["id"], turn),
            tag=_tag(event.modules, turn["agent"], turn.get("latency_s")),
        )
    if event.kind == "ended":  # replaces the "thinking" bubble of a turn that was never written
        return render(
            "partials/studio_ended.html",
            n=event.index + 1,
            reasons=[dataset.reason_text(r) for r in generation["reasons"]],
        )
    judge = generation["judge"]  # None when the call ended early and was not judged
    return render(
        "partials/studio_verdict.html",
        judge=judge,
        accepted=generation["status"] == "accepted",
        reasons=[dataset.reason_text(r) for r in generation["reasons"]],
        scam_start=record["scam_start_turn"],
        summary=record["reference_summary"],
        label_scam=record["label_scam"],
        tag=_tag(event.modules, "judge"),
    )
