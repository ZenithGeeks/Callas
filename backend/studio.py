"""The /studio page: make a simulated call from the browser and watch it being written.

POST /studio/start starts a background thread that runs the simulator graph
(data_pipeline.live.stream_call) and saves the call to the store /dataset reads. Each step is
rendered as an HTML piece and added to the run's event list. GET /studio/{run}/events sends
that list as server-sent events, so a page that opens late, or again, sees the whole call.
One run at a time per server. Stop asks the thread to stop after the step it is working on.
"""

from __future__ import annotations

import asyncio
import logging
import random
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
from data_pipeline.scenarios import sample_config, sample_receiver
from data_pipeline.store import Store
from llm_serving.client import LLMUnavailable

from . import dataset
from .web import ROOT, form_error, page, render

log = logging.getLogger("callas.studio")
SHUTDOWN = threading.Event()  # set when the app stops, so open event streams end

RUN_ID_RE = re.compile(r"[0-9a-f]{12}")
KEPT_RUNS = 5
CUSTOM = "__custom__"
CUSTOM_KEY = "custom"  # the key in the call id of a call with a described receiver
MAX_SEED = 99_999
MAX_TEXT = 200

# (form field, label, ReceiverProfile pool). The form field is sent as r_<field>; the pool fills
# the suggestions list.
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
    """A form value the user has to fix. `status` is the HTTP code (409 when a run is busy)."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class StudioRun:
    """One call being written: what was asked for, the HTML pieces so far, and the stop flag."""

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


@router.get("/studio", response_class=HTMLResponse)
def studio_page(request: Request) -> HTMLResponse:
    """The form, with the run that is going now (if any) already opened."""
    config = _config()
    pools = config.scenarios.receivers
    scenarios = [
        (scenario.key, call_type, f"{scenario.key}: {scenario.role}")
        for call_type in CALL_TYPES
        for scenario in _scenarios_of(config, call_type)
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
        db=_shown_path(dataset.db_path()),
    )


def _shown_path(path: Path) -> str:
    """The store's path as the page shows it: relative to the repo, or only the file name.

    The page can be opened from other machines (tailscale), so it does not print this machine's
    folder layout.
    """
    try:
        return path.resolve().relative_to(ROOT.parent).as_posix()
    except ValueError:
        return path.name


@cache
def _config() -> Config:
    return load_default_config()


def _scenarios_of(config: Config, call_type: str) -> list:
    """The scenarios the config has for this kind of call."""
    if call_type == BENIGN:
        return config.scenarios.legit
    return config.scenarios.scams[call_type]


def _type_options() -> list[tuple[str, str]]:
    return [(key, dataset.CALL_TYPE_NAMES[key]) for key in CALL_TYPES]


def _profile_label(profile: ReceiverProfile) -> str:
    return f"{profile.key}: {profile.gender}, {profile.age}, {profile.occupation}"


@router.post("/studio/start", response_class=HTMLResponse)
async def studio_start(request: Request) -> Response:
    """Check the form and start a run, or answer with the problem."""
    form = await request.form()  # the store and the templates are used in a worker thread
    demo = request.app.state.app_runtime.is_demo
    return await run_in_threadpool(_begin, form, demo, request.app.state.pace)


def _begin(form: Any, demo: bool, pace: float) -> Response:
    """Register the run and start its thread (runs in the worker pool: it uses the store)."""
    try:
        config = _config()
        spec = parse_form(form, config, _taken_ids(config))
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


def parse_form(form: Any, config: Config, taken_ids: set[str]) -> live.CallSpec:
    """Check the submitted form and turn it into a CallSpec. Raises FormProblem."""
    call_type = str(form.get("call_type", ""))
    if call_type not in CALL_TYPES:
        raise FormProblem("Choose the kind of call.")
    scenario = _parse_scenario(form, config, call_type)
    receiver = _parse_receiver(form, config)
    name = _clean(form.get("receiver_name", ""))
    seed_text = str(form.get("seed", "")).strip()
    if not seed_text:
        seed = live.next_free_seed(call_type, taken_ids, receiver)
        return live.CallSpec(call_type, seed, scenario, receiver, name)
    spec = live.CallSpec(call_type, _parse_seed(seed_text), scenario, receiver, name)
    if spec.record_id in taken_ids:
        raise FormProblem(
            f"{spec.record_id} is already in the dataset. "
            "Choose another seed, or leave the seed empty to take the next free one."
        )
    return spec


def _parse_scenario(form: Any, config: Config, call_type: str) -> Any:
    """The chosen scenario of this kind of call, or None when the form leaves it empty."""
    key = str(form.get("scenario", ""))
    if not key:
        return None
    scenario = next((s for s in _scenarios_of(config, call_type) if s.key == key), None)
    if scenario is None:
        raise FormProblem("That scenario does not belong to this kind of call.")
    return scenario


def _parse_seed(text: str) -> int:
    if not (text.isascii() and text.isdigit()) or int(text) > MAX_SEED:
        raise FormProblem(f"The seed is a whole number from 0 to {MAX_SEED}, or empty.")
    return int(text)


def _clean(value: Any) -> str:
    """One line of text for a prompt: control and format characters are dropped, any run of
    white space (including line breaks) becomes one space. Thai letters are kept."""
    text = "".join(
        c for c in str(value) if unicodedata.category(c) not in ("Cc", "Cf") or c.isspace()
    )
    return " ".join(text.split())[:MAX_TEXT]


def _parse_receiver(form: Any, config: Config) -> ReceiverProfile | None:
    """The chosen saved person, the described one, or None when the form leaves it empty."""
    choice = str(form.get("receiver", ""))
    if not choice:
        return None
    if choice == CUSTOM:
        return _described_receiver(form, config.scenarios.receivers)
    profile = next((p for p in config.scenarios.receivers.profiles if p.key == choice), None)
    if profile is None:
        raise FormProblem("That saved person does not exist.")
    return profile


def _described_receiver(form: Any, pools: Any) -> ReceiverProfile:
    """A receiver from the r_* fields. Any field left on "Random" (empty) is filled from a
    coherently sampled base, so a half-described person is still a believable one."""
    base = sample_receiver(random.Random(), pools)
    gender = str(form.get("r_gender", ""))
    if gender and gender not in pools.genders:
        raise FormProblem(f"Gender is one of: {', '.join(pools.genders)}.")
    gender = gender or base.gender
    age_text = str(form.get("r_age", "")).strip()
    try:
        age = int(age_text) if age_text else base.age
    except ValueError:
        raise FormProblem("The age is a whole number from 10 to 100.") from None
    values = {
        key: _clean(form.get(f"r_{key}", "")) or getattr(base, key) for key, _, _ in DETAIL_FIELDS
    }
    try:
        return ReceiverProfile(key=CUSTOM_KEY, gender=gender, age=age, **values)
    except (ValidationError, ValueError):
        raise FormProblem("The age is a whole number from 10 to 100.") from None


def _taken_ids(config: Config) -> set[str]:
    """Calls an empty seed must skip.

    A stopped call made with this config stays available so an ordinary Start can resume it.
    One made with another config cannot resume safely, so automatic selection skips it instead
    of handing the user a config-conflict error after a configuration change.
    """
    db = dataset.store()
    if db is None:
        return set()
    taken = set()
    for row in db.summaries():
        if row["status"] != "in_progress":
            taken.add(row["id"])
            continue
        record = db.load(row["id"])
        if record and record["generation"].get("config_hash") != config.hash:
            taken.add(row["id"])
    return taken


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
        # the outcome locked for this call (scenarios.sample_config draws it 50/50 by seed); shown
        # as a badge. "" when no outcome is locked.
        "scam_outcome": _locked_outcome(spec),
    }


def _locked_outcome(spec: live.CallSpec) -> str:
    """This call's locked scam outcome ("scammed"/"safe"/""). Sampling is deterministic for
    (call_type, seed, receiver), so this is the outcome the run's receiver was built to lead to."""
    return sample_config(spec.call_type, spec.seed, _config(), spec.receiver).receiver.scam_outcome


def _elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def _sse(name: str, html: str, event_id: int) -> str:
    data = "\n".join(f"data: {line}" for line in html.splitlines() or [""])
    return f"id: {event_id}\nevent: {name}\n{data}\n\n"


def _unknown_run() -> JSONResponse:
    return JSONResponse({"detail": "Unknown run."}, status_code=404)


@router.get("/studio/{run_id}/events")
async def studio_events(request: Request, run_id: str) -> Response:
    """The run's pieces as server-sent events, from the start or after Last-Event-ID."""
    run = BOARD.get(run_id)
    if run is None:
        return _unknown_run()
    last_id = request.headers.get("last-event-id", "")
    first = int(last_id) + 1 if last_id.isdigit() else 0
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(
        _event_stream(request, run, first), media_type="text/event-stream", headers=headers
    )


async def _event_stream(request: Request, run: StudioRun, first: int) -> AsyncIterator[str]:
    """Yield the run's events from index `first`, then new ones until the run is finished."""
    yield "retry: 3000\n\n"
    sent, quiet_ticks = first, 0
    while True:
        while sent < len(run.events):
            name, html = run.events[sent]
            yield _sse(name, html, sent)
            sent, quiet_ticks = sent + 1, 0
        if (run.finished and sent >= len(run.events)) or SHUTDOWN.is_set():
            return
        if await request.is_disconnected():
            return  # the call keeps running and is saved; a new page can replay it
        await asyncio.sleep(0.1)
        quiet_ticks += 1
        if quiet_ticks % 150 == 0:  # about every 15 s, so proxies keep the connection open
            yield ": still running\n\n"


@router.get("/studio/{run_id}/state")
def studio_state(run_id: str) -> Response:
    """Lets the page tell a lost connection from a run the server no longer knows."""
    run = BOARD.get(run_id)
    if run is None:
        return _unknown_run()
    return JSONResponse({"finished": run.finished})


@router.post("/studio/{run_id}/stop")
def studio_stop(run_id: str) -> Response:
    """Ask the run's thread to stop after the step it is working on."""
    run = BOARD.get(run_id)
    if run is None:
        return _unknown_run()
    run.cancel.set()
    return Response(status_code=204)


def _work(run: StudioRun, demo: bool, pace: float, speak: bool, db_path: Path) -> None:
    """Run the call and push one HTML piece per step. Never raises: problems become an event."""
    outcome = "stopped"
    try:
        store = Store(db_path)
        config = _config()
        agents, voices = _make_agents(config, run.spec.call_type, store, demo, pace, speak)
        capture = live.Capture(agents)
        with dspy.context(callbacks=[capture]):
            for event in live.stream_call(
                run.spec, store, agents, config, voices, capture, run.cancel
            ):
                run.push(event.kind, _render_event(event))
                if event.kind == "verdict":
                    outcome = "finished"
    except Exception as exc:
        outcome = "failed"
        run.push(
            "failed", render("partials/studio_failed.html", message=_failure_message(run, exc))
        )
    finally:
        run.push("done", _done_html(run, outcome))
        run.finished = True  # last: a new run may start from here on


def _make_agents(
    config: Config, call_type: str, store: Store, demo: bool, pace: float, speak: bool
) -> tuple[Any, Any]:
    """The agents that write the call, and the voices (None unless the form asked to speak)."""
    if demo:
        agents = live.fake_agents(config, call_type, delay=2 * pace)
        return agents, live.SilentVoices() if speak else None
    from data_pipeline.agents import Agents
    from data_pipeline.tts import Voices

    _sync_wiki(store)
    return Agents(config), Voices() if speak else None


def _done_html(run: StudioRun, outcome: str) -> str:
    """The closing message of a run. A page that fails to render must not keep the run open."""
    try:
        return render("partials/studio_done.html", run=_run_view(run), outcome=outcome)
    except Exception:
        log.exception("studio run %s: could not render the last message", run.id)
        return "<p>The run has ended.</p>"


def _sync_wiki(store: Store) -> None:
    from data_pipeline.wiki import DEFAULT_WIKI, WikiError, sync

    if DEFAULT_WIKI.is_dir():
        try:
            sync(store, DEFAULT_WIKI)
        except WikiError as exc:
            log.warning("wiki sync skipped: %s", exc)


def _failure_message(run: StudioRun, exc: Exception) -> str:
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


def _render_event(event: live.Event) -> str:
    """The HTML piece for one step of the call. The event name is the step's kind."""
    match event.kind:
        case "thinking":
            return _thinking_html(event)
        case "plan":
            return _plan_html(event)
        case "turn":
            return _turn_html(event)
        case "ended":  # replaces the "thinking" bubble of a turn that was never written
            return _ended_html(event)
        case _:
            return _verdict_html(event)


def _thinking_html(event: live.Event) -> str:
    turns_so_far = len(event.record["generation"]["turns"])
    return render("partials/studio_thinking.html", agent=event.agent, n=turns_so_far + 1)


def _plan_html(event: live.Event) -> str:
    briefing = event.record["generation"]["briefing"]
    return render(
        "partials/studio_plan.html",
        plan=briefing["plan"],
        notes=briefing["notes_used"],
        reasoning=briefing.get("reasoning", ""),
        tag=_tag(event.modules, "planner"),
    )


def _turn_html(event: live.Event) -> str:
    turn = event.record["generation"]["turns"][event.index]
    return render(
        "partials/studio_turn.html",
        n=event.index + 1,
        turn=turn,
        side="left" if turn["speaker"] == dataset.GENERATED_CALLER else "right",
        who=dataset.SPEAKER_NAMES[turn["speaker"]],
        flags=dataset.turn_flags(turn),
        leaked=dataset.leaked_text(turn),
        audio=_audio_url(event.record["id"], turn),
        tag=_tag(event.modules, turn["agent"], turn.get("latency_s")),
    )


def _ended_html(event: live.Event) -> str:
    reasons = event.record["generation"]["reasons"]
    return render(
        "partials/studio_ended.html",
        n=event.index + 1,
        reasons=[dataset.reason_text(r) for r in reasons],
    )


def _verdict_html(event: live.Event) -> str:
    record = event.record
    generation = record["generation"]
    judge = generation.get("judge")
    confidence = judge.get("confidence") if judge else None
    return render(
        "partials/studio_verdict.html",
        call_id=record.get("id"),
        judge=judge,  # None when the call ended early and was not judged
        accepted=generation["status"] == "accepted",
        confidence=confidence,
        review=record.get("review"),
        reasons=[dataset.reason_text(r) for r in generation.get("reasons", [])],
        scam_start=record.get("scam_start_turn"),
        summary=record.get("reference_summary"),
        label_scam=record.get("label_scam"),
        tag=_tag(event.modules, "judge"),
    )


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
    """One model call as the template shows it."""
    return {
        "model": call.model,
        "messages": call.messages,
        "answer": call.answer,
        "error": call.error,
    }


def _audio_url(record_id: str, turn: dict) -> str | None:
    """The URL of a turn's audio, if the turn has any (the file is not checked here)."""
    return f"/dataset/audio/{record_id}/{Path(turn['audio']).name}" if turn.get("audio") else None
