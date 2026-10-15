"""The "Simulate a call" page: upload a recording, watch it analysed live.

URLs: GET /call (the form), POST /call/simulate (starts a run, returns the page shell),
GET /call/{run_id}/events (server-sent events that fill the shell in as the run goes).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from markupsafe import Markup

from . import call_run, demo, web
from .web import form_error, page, render, templates

log = logging.getLogger("callas.web")

router = APIRouter()

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")


@router.get("/call", response_class=HTMLResponse)
async def call_page(request: Request) -> HTMLResponse:
    """The page with the upload form."""
    return page(request, "call.html", "call", max_mb=MAX_UPLOAD_BYTES // (1024 * 1024))


def _looks_like_audio(ext: str, head: bytes) -> bool:
    """True if the first bytes match the format the file name claims."""
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
    fd, path = tempfile.mkstemp(suffix=ext, prefix="callas-", dir=web.UPLOAD_DIR)
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
    """Each result region of the page as HTML, by name."""
    mod = templates.env.get_template("partials/call_regions.html").module
    return {name: str(getattr(mod, name)(view, fresh)) for name in REGIONS}


def _sse(event: str, data: str) -> str:
    """One server-sent event; every line of `data` needs its own `data:` prefix."""
    return f"event: {event}\n" + "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


@router.post("/call/simulate", response_class=HTMLResponse)
async def call_simulate(request: Request, audio: UploadFile | None = None) -> HTMLResponse:
    """Save the recording, register a run, and return the empty result shell."""
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
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="callas-", dir=web.UPLOAD_DIR)
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


@router.get("/call/{run_id}/events")
async def call_events(request: Request, run_id: str) -> Response:
    """The run's event stream (404 unknown run, 409 already streaming, replay once finished)."""
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
    """Run the analysis and yield an event whenever a region of the page changes."""
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
