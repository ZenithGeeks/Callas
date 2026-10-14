"""Callas web demo. Run: uvicorn backend.main:app

Every endpoint is a thin wrapper over the orchestrator. No model or business logic lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "modules"))  # run without PYTHONPATH

import asyncio  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Form, Request, UploadFile, WebSocket  # noqa: E402
from fastapi.responses import (  # noqa: E402
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402
from markupsafe import Markup  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from . import call_run, demo, live_ws, views  # noqa: E402
from . import runtime as rt_mod  # noqa: E402

log = logging.getLogger("callas.web")

MAX_MESSAGE_CHARS = 2000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
DEMO_PACE_SECONDS = 0.45  # demo mode only: pause after each turn so the page visibly streams

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True
templates.env.globals.update(
    fmt_time=views.fmt_time, who=views.who, segments=views.segments, entity_rows=views.entity_rows
)


# Uploaded recordings live only in this folder. It is emptied at startup and shutdown, so a
# crash or restart never leaves call audio behind.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "callas-uploads"


def clear_uploads() -> None:
    if UPLOAD_DIR.is_dir():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    clear_uploads()
    app.state.rt = rt_mod.build()
    app.state.runs = call_run.RunRegistry()
    app.state.pace = float(os.environ.get("CALLAS_FAKE_DELAY", DEMO_PACE_SECONDS))
    log.info("Callas web demo started in %s mode", app.state.rt.mode)
    try:
        yield
    finally:
        app.state.rt.close()
        clear_uploads()


app = FastAPI(title="Callas", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def render(name: str, **ctx) -> str:
    return templates.env.get_template(name).render(**ctx)


def page(request: Request, name: str, active: str, **ctx) -> HTMLResponse:
    rt = request.app.state.rt
    html = render(name, request=request, active=active, demo=rt.is_demo, **ctx)
    return HTMLResponse(html)


def form_error(message: str, status: int = 422) -> HTMLResponse:
    """A short message for the #form-error slot. htmx puts it there (HX-Retarget)."""
    return HTMLResponse(
        render("partials/form_error.html", message=message),
        status_code=status,
        headers={"HX-Retarget": "#form-error", "HX-Reswap": "innerHTML"},
    )


# ---- basics -------------------------------------------------------------------------------


@app.get("/")
async def index() -> RedirectResponse:
    return RedirectResponse("/message", status_code=307)


@app.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    rt = request.app.state.rt
    reachable = await run_in_threadpool(rt.llm_reachable)
    return JSONResponse(
        {
            "status": "ok",
            "mode": rt.mode,
            "llm_reachable": reachable,
            "live_call": live_ws.live_backend_name(rt),
        }
    )


# ---- check a message ----------------------------------------------------------------------


@app.get("/message", response_class=HTMLResponse)
async def message_page(request: Request) -> HTMLResponse:
    return page(request, "message.html", "message", max_chars=MAX_MESSAGE_CHARS)


@app.post("/message/check", response_class=HTMLResponse)
async def message_check(request: Request, text: str = Form("")) -> HTMLResponse:
    from llm_serving.client import LLMUnavailable
    from orchestrator import ToolCallingUnavailable

    rt = request.app.state.rt
    text = text.strip()
    if not text:
        return form_error("Paste a message first.")
    if len(text) > MAX_MESSAGE_CHARS:
        return form_error(
            f"The message is {len(text)} characters. The limit is {MAX_MESSAGE_CHARS}. "
            "Shorten it and try again."
        )
    try:
        result = await run_in_threadpool(rt.sessions.analyze_message, text)
    except LLMUnavailable as exc:
        return form_error(f"Cannot check the message. {exc}", 503)
    except Exception:
        log.exception("message check failed")
        return form_error("Something went wrong while checking the message.", 500)

    try:
        why, why_error = await run_in_threadpool(rt.explain, text), None
    except ToolCallingUnavailable:
        why, why_error = None, "Explanation unavailable: the LLM server does not allow tool calls."
    except LLMUnavailable as exc:
        why, why_error = None, f"Explanation unavailable: {exc}"
    except Exception:
        log.exception("explain failed")
        why, why_error = None, "Explanation unavailable: the LLM server did not answer."

    nlp = result["nlp"]
    html = render(
        "partials/message_result.html",
        band=views.message_band(result["verdict"]),
        text=text,
        entities=nlp.entities,
        summary=nlp.summary,
        why=why,
        why_error=why_error,
        timings=result["timings"],
    )
    return HTMLResponse(html)


# ---- simulate a call ----------------------------------------------------------------------


@app.get("/call", response_class=HTMLResponse)
async def call_page(request: Request) -> HTMLResponse:
    return page(request, "call.html", "call", max_mb=MAX_UPLOAD_BYTES // (1024 * 1024))


def _looks_like_audio(ext: str, head: bytes) -> bool:
    if ext == ".wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == ".ogg":
        return head[:4] == b"OggS"
    if ext == ".m4a":
        return head[4:8] == b"ftyp"
    return head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


async def _save_upload(upload: UploadFile) -> tuple[str | None, str | None]:
    """Save to a temp file. Returns (path, error message)."""
    name = Path(upload.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        return None, "That file is not audio. Use a .wav, .mp3, .m4a or .ogg recording."
    fd, path = tempfile.mkstemp(suffix=ext, prefix="callas-", dir=UPLOAD_DIR)
    size, head = 0, b""
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(1024 * 1024):
                if not head:
                    head = chunk[:16]
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    Path(path).unlink(missing_ok=True)
                    return None, "The recording is larger than 20 MB. Use a shorter one."
                out.write(chunk)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    if not _looks_like_audio(ext, head):
        Path(path).unlink(missing_ok=True)
        return None, f"That file does not look like a {ext[1:].upper()} recording."
    return path, None


REGIONS = ("stages", "verdict", "strip", "transcript", "entities", "summary", "log")


def _render_regions(view: call_run.RunView, fresh: bool) -> dict[str, str]:
    mod = templates.env.get_template("partials/call_regions.html").module
    return {name: str(getattr(mod, name)(view, fresh)) for name in REGIONS}


def _sse(event: str, data: str) -> str:
    return f"event: {event}\n" + "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


@app.post("/call/simulate", response_class=HTMLResponse)
async def call_simulate(request: Request, audio: UploadFile | None = None) -> HTMLResponse:
    app_rt = request.app.state.rt
    has_file = audio is not None and bool(audio.filename)
    if not has_file and not app_rt.is_demo:
        return form_error("Choose a recording first.")
    path: str | None = None
    filename = demo.DEMO_FILENAME
    if has_file:
        path, error = await _save_upload(audio)
        if error:
            return form_error(error)
        filename = Path(audio.filename).name
    else:  # demo mode: the fake pipeline ignores the file but still wants a path
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="callas-", dir=UPLOAD_DIR)
        os.close(fd)
    run = request.app.state.runs.create(path, filename)
    view = call_run.RunView(filename, app_rt.is_demo)
    html = render(
        "partials/call_shell.html",
        run=run,
        view=view,
        regions={k: Markup(v) for k, v in _render_regions(view, False).items()},
    )
    return HTMLResponse(html)


@app.get("/call/{run_id}/events")
async def call_events(request: Request, run_id: str) -> Response:
    runs: call_run.RunRegistry = request.app.state.runs
    run = runs.get(run_id)
    if run is None:
        return JSONResponse({"detail": "Unknown run."}, status_code=404)
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    if run.finished:  # a reconnect after the end: replay the final picture, then close

        async def replay() -> AsyncIterator[str]:
            for name, html in run.snapshot.items():
                yield _sse(name, html)
            yield _sse("done", "done")

        return StreamingResponse(replay(), media_type="text/event-stream", headers=headers)
    if run.started:
        return JSONResponse({"detail": "This run is already streaming."}, status_code=409)
    run.started = True
    return StreamingResponse(
        _stream_run(request, run), media_type="text/event-stream", headers=headers
    )


async def _stream_run(request: Request, run: call_run.Run) -> AsyncIterator[str]:
    app = request.app
    app_rt = app.state.rt
    view = call_run.RunView(run.filename, app_rt.is_demo)
    sent: dict[str, str] = {}
    stop = threading.Event()

    def changed(fresh: bool) -> list[str]:
        out = []
        for name, html in _render_regions(view, fresh).items():
            key = html.replace("is-fresh", "")  # the animation class alone is not a change
            if sent.get(name) != key:
                sent[name] = key
                out.append(_sse(name, html))
        return out

    try:
        yield ": open\n\n"
        async for kind, value in call_run.iter_updates(app_rt, run.path, stop):
            if kind == "transcript":
                if value:
                    view.set_transcript(*value)
            elif kind == "item":
                view.feed(value)
            elif kind == "error":
                log.error("call run failed: %s", type(value).__name__)
                from llm_serving.client import LLMUnavailable

                view.error = (
                    str(value)
                    if isinstance(value, LLMUnavailable)
                    else "Something went wrong while reading the call."
                )
            elif kind == "end":
                view.finish()
            for chunk in changed(view.take_fresh()):
                yield chunk
            if kind == "item" and app_rt.is_demo and app.state.pace:
                if value["node"] in call_run.TERMINAL_NODES:
                    await asyncio.sleep(
                        app.state.pace * (2 if view.esc_turn == value["turn_index"] + 1 else 1)
                    )
        view.finish()
        run.snapshot = _render_regions(view, False)
        run.finished = True
        run.finished_at = time.monotonic()
        yield _sse("done", "done")
    finally:
        stop.set()
        if run.path:
            Path(run.path).unlink(missing_ok=True)
            run.path = None


# ---- live call ----------------------------------------------------------------------------


@app.get("/live", response_class=HTMLResponse)
async def live_page(request: Request) -> HTMLResponse:
    sample = {"word": "", "subtype": "", "instruction": "", "fact_label": "", "fact_big": ""}
    bands = {
        kind: views.Band(kind, fact_note="", inline_fact="-", **sample)
        for kind in ("safe", "scam", "unsure")
    }
    return page(
        request,
        "live.html",
        "live",
        bands=bands,
        high_conf=views.HIGH_CONFIDENCE,
        types_json=json.dumps(views.TYPE_NAMES),
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    # The first real call loads the streaming ASR model. Off the event loop, so other pages
    # keep answering while it loads.
    call = await run_in_threadpool(live_ws.make_live_call, ws.app.state.rt)
    await live_ws.serve(ws, call)
    try:
        await ws.close()
    except RuntimeError:  # already closed by the browser
        pass
