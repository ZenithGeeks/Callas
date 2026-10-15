"""Callas web demo. Run: uvicorn backend.main:app

Every endpoint is a thin wrapper over the orchestrator. No model or business logic lives here.
This file builds the app; each page's routes are in its own router module.
"""

from __future__ import annotations

import sys

from .web import ROOT

sys.path.insert(0, str(ROOT.parent / "modules"))  # run without PYTHONPATH

import logging  # noqa: E402
import os  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from callas_tracing import setup_tracing  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from callas_settings import get_settings  # noqa: E402

from . import basics, call, call_run, live, message, review, studio  # noqa: E402
from . import runtime as rt_mod  # noqa: E402

log = logging.getLogger("callas.web")

DEMO_PACE_SECONDS = 0.45  # demo mode only: pause after each turn so the page visibly streams


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the runtime and run registry; on shutdown close the runtime and delete uploads."""
    setup_tracing("callas-web", call_text=get_settings().trace_call_text)
    call.clear_uploads()
    app.state.rt = rt_mod.build()
    app.state.runs = call_run.RunRegistry()
    app.state.pace = float(os.environ.get("CALLAS_FAKE_DELAY", DEMO_PACE_SECONDS))
    log.info("Callas web demo started in %s mode", app.state.rt.mode)
    try:
        yield
    finally:
        app.state.rt.close()
        call.clear_uploads()


app = FastAPI(title="Callas", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

app.include_router(basics.router)
app.include_router(message.router)
app.include_router(call.router)
app.include_router(review.router)
app.include_router(studio.router)
app.include_router(live.router)
