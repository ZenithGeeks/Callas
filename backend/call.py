"""The "Simulate a call" page: upload a recording, watch it analysed live.

URLs: GET /call (the form), POST /call/simulate (starts a run, returns the page shell),
GET /call/{run_id}/events (server-sent events that fill the shell in as the run goes).
The run itself (RunView, RunRegistry, iter_updates) is in call_run.py.
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
from typing import Any, BinaryIO

from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from markupsafe import Markup

from llm_serving.client import LLMUnavailable

from . import call_run, demo
from .web import form_error, page, render, templates

log = logging.getLogger("callas.web")

router = APIRouter()

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# The result regions of the page, in the order partials/call_regions.html defines them.
REGIONS = ("stages", "verdict", "strip", "transcript", "entities", "summary", "log")

# Uploaded recordings live only in this folder. It is emptied at startup and shutdown, so a
# crash or restart never leaves call audio behind. Always read it as this module's global:
# tests point it at a temp folder, and a copy imported elsewhere would not follow.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "callas-uploads"


class BadUpload(Exception):
    """The recording cannot be used. The message tells the user why."""


def clear_uploads() -> None:
    """Empty the upload folder (and create it if it is missing)."""
    if UPLOAD_DIR.is_dir():
        for upload in UPLOAD_DIR.iterdir():
            if upload.is_file():
                upload.unlink(missing_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


@router.get("/call", response_class=HTMLResponse)
async def call_page(request: Request) -> HTMLResponse:
    """The page with the upload form."""
    return page(request, "call.html", "call", max_mb=MAX_UPLOAD_MB)


@router.post("/call/simulate", response_class=HTMLResponse)
async def call_simulate(request: Request, audio: UploadFile | None = None) -> HTMLResponse:
    """Save the recording, register a run, and return the empty result shell."""
    app_rt = request.app.state.rt
    if audio is not None and audio.filename:
        try:
            path = await _save_upload(audio)
        except BadUpload as problem:
            return form_error(str(problem))
        filename = Path(audio.filename).name
    elif app_rt.is_demo:
        path = _placeholder_recording()
        filename = demo.DEMO_FILENAME
    else:
        return form_error("Choose a recording first.")
    run = request.app.state.runs.create(path, filename)
    view = call_run.RunView(filename, app_rt.is_demo)
    regions = {name: Markup(html) for name, html in _render_regions(view, False).items()}
    return HTMLResponse(render("partials/call_shell.html", run=run, view=view, regions=regions))


@router.get("/call/{run_id}/events")
async def call_events(request: Request, run_id: str) -> Response:
    """The run's event stream (404 unknown run, 409 already streaming, replay once finished)."""
    run = request.app.state.runs.get(run_id)
    if run is None:
        return JSONResponse({"detail": "Unknown run."}, status_code=404)
    if run.finished:  # a reconnect after the end: replay the final picture, then close
        return StreamingResponse(_replay(run), media_type="text/event-stream", headers=SSE_HEADERS)
    if run.started:
        return JSONResponse({"detail": "This run is already streaming."}, status_code=409)
    run.started = True
    return StreamingResponse(
        _stream_run(request, run), media_type="text/event-stream", headers=SSE_HEADERS
    )


async def _save_upload(upload: UploadFile) -> str:
    """Copy the upload into UPLOAD_DIR and return the path. Raises BadUpload."""
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in AUDIO_EXTENSIONS:
        raise BadUpload("That file is not audio. Use a .wav, .mp3, .m4a or .ogg recording.")
    fd, path = tempfile.mkstemp(suffix=ext, prefix="callas-", dir=UPLOAD_DIR)
    try:
        with os.fdopen(fd, "wb") as out:
            head = await _copy_within_limit(upload, out)
        if not _looks_like_audio(ext, head):
            raise BadUpload(f"That file does not look like a {ext[1:].upper()} recording.")
    except BaseException:  # includes a dropped connection: no half-written file stays behind
        Path(path).unlink(missing_ok=True)  # after the with block, so the file is closed first
        raise
    return path


async def _copy_within_limit(upload: UploadFile, out: BinaryIO) -> bytes:
    """Write the upload to `out` and return its first 16 bytes. Raises BadUpload if too big."""
    head, size = b"", 0
    while chunk := await upload.read(1024 * 1024):
        head = head or chunk[:16]
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise BadUpload(f"The recording is larger than {MAX_UPLOAD_MB} MB. Use a shorter one.")
        out.write(chunk)
    return head


def _looks_like_audio(ext: str, head: bytes) -> bool:
    """True if the first bytes match the format the file name claims."""
    if ext == ".wav":
        return head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if ext == ".ogg":
        return head[:4] == b"OggS"
    if ext == ".m4a":
        return head[4:8] == b"ftyp"
    return head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0)


def _placeholder_recording() -> str:
    """An empty .wav path for demo mode: the fake pipeline ignores the file but wants a path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="callas-", dir=UPLOAD_DIR)
    os.close(fd)
    return path


async def _stream_run(request: Request, run: call_run.Run) -> AsyncIterator[str]:
    """Run the analysis and yield an event whenever a region of the page changes."""
    app = request.app
    app_rt = app.state.rt
    view = call_run.RunView(run.filename, app_rt.is_demo)
    sent: dict[str, str] = {}
    stop = threading.Event()
    try:
        yield ": open\n\n"
        async for kind, value in call_run.iter_updates(app_rt, run.path, stop):
            _apply_update(view, kind, value)
            for event in _changed_regions(view, sent, fresh=view.take_fresh()):
                yield event
            if kind == "item" and app_rt.is_demo and app.state.pace:
                await _pause_demo(app.state.pace, view, value)
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


def _apply_update(view: call_run.RunView, kind: str, value: Any) -> None:
    """Record one update from call_run.iter_updates in the view."""
    if kind == "transcript" and value:
        view.set_transcript(*value)
    elif kind == "item":
        view.feed(value)
    elif kind == "error":
        log.error("call run failed: %s", type(value).__name__)
        if isinstance(value, LLMUnavailable):
            view.error = str(value)
        else:
            view.error = "Something went wrong while reading the call."
    elif kind == "end":
        view.finish()


def _changed_regions(view: call_run.RunView, sent: dict[str, str], fresh: bool) -> list[str]:
    """One event for each region whose HTML differs from the last one sent (kept in `sent`)."""
    events = []
    for name, html in _render_regions(view, fresh).items():
        comparable = html.replace("is-fresh", "")  # the animation class alone is not a change
        if sent.get(name) != comparable:
            sent[name] = comparable
            events.append(_sse(name, html))
    return events


async def _pause_demo(pace: float, view: call_run.RunView, item: dict) -> None:
    """Demo mode only: wait after a turn is decided so the page visibly streams.

    The turn that escalates waits twice as long, so the alarm has time to show.
    """
    if item["node"] not in call_run.TERMINAL_NODES:
        return
    escalates_here = view.esc_turn == item["turn_index"] + 1
    await asyncio.sleep(pace * (2 if escalates_here else 1))


async def _replay(run: call_run.Run) -> AsyncIterator[str]:
    """The final picture of a finished run, for a browser that reconnects."""
    for name, html in run.snapshot.items():
        yield _sse(name, html)
    yield _sse("done", "done")


def _render_regions(view: call_run.RunView, fresh: bool) -> dict[str, str]:
    """Each result region of the page as HTML, by name."""
    macros = templates.env.get_template("partials/call_regions.html").module
    return {name: str(getattr(macros, name)(view, fresh)) for name in REGIONS}


def _sse(event: str, data: str) -> str:
    """One server-sent event; every line of `data` needs its own `data:` prefix."""
    return f"event: {event}\n" + "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"
