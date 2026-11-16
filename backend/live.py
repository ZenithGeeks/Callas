"""The "Live call" page and its WebSocket.

URLs: GET /live (the page), GET /live/script (a test-split call to read aloud, test_calls.py),
WebSocket /ws/live (audio in, analysis events out). The socket loop lives in live_ws.py.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from nlp_core import registry

from . import live_ws, test_calls, views
from .web import page, render

log = logging.getLogger("callas.web")

router = APIRouter()

# Registry names (NLP_CLASSIFIER) as the page shows them. Other names are shown as they are.
CLASSIFIER_NAMES = {"llm": "LLM", "phayathaibert": "fine-tuned PhayaThaiBERT"}


@router.get("/live", response_class=HTMLResponse)
def live_page(request: Request) -> HTMLResponse:
    """The live-call page. Sync, so reading the test calls runs in a thread."""
    sample = {"word": "", "subtype": "", "instruction": "", "fact_label": "", "fact_big": ""}
    bands = {
        kind: views.Band(kind, fact_note="", inline_fact="-", **sample)
        for kind in ("safe", "scam", "unsure")
    }
    is_demo = request.app.state.app_runtime.is_demo
    return page(
        request,
        "live.html",
        "live",
        bands=bands,
        high_conf=views.HIGH_CONFIDENCE,
        types_json=json.dumps(views.TYPE_NAMES),
        classifier=CLASSIFIER_NAMES.get(name := registry.default_name("classifier"), name),
        call=None if is_demo else test_calls.random_test_call(),
    )


@router.get("/live/script", response_class=HTMLResponse)
def live_script() -> HTMLResponse:  # sync: FastAPI runs it in a thread, off the event loop
    """Another random test-split call to read aloud: the inner part of the script section."""
    return HTMLResponse(render("partials/live_script.html", call=test_calls.random_test_call()))


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Start a live call and pass audio and events through until the browser leaves."""
    await ws.accept()
    # The first real call loads the streaming ASR model. Off the event loop, so other pages
    # keep answering while it loads.
    try:
        call = await run_in_threadpool(live_ws.make_live_call, ws.app.state.app_runtime)
    except Exception:
        # A model that will not load (bad ASR_MODEL, missing files) must say so on screen,
        # not just close the socket. The cause is in the server log.
        log.exception("live call could not start")
        await ws.send_json(
            {"type": "error", "turn": None, "message": "The live call could not start."}
        )
        await ws.close()
        return
    await live_ws.serve(ws, call)
    try:
        await ws.close()
    except RuntimeError:  # already closed by the browser
        pass
