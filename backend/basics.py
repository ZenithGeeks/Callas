"""The root redirect and the health check.

URLs: GET /, GET /healthz.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from . import live_ws

router = APIRouter()


@router.get("/")
async def index() -> RedirectResponse:
    """Send the root to the message page."""
    return RedirectResponse("/message", status_code=307)


@router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    """Mode, whether the LLM answers, and which live-call backend is in use."""
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
