"""WebSocket glue for the live call.

The browser sends PCM16 LE mono 16 kHz frames (binary) and JSON controls
{"type": "pause" | "resume" | "end"}. The server forwards LiveCall.poll_events() as JSON.

The LiveCall interface (feed, pause, resume, poll_events, end) comes from orchestrator.live
when it exists; otherwise, and in demo mode, DemoLiveCall plays a scripted call.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .live_demo import DemoLiveCall

log = logging.getLogger(__name__)

POLL_SECONDS = 0.1


def live_backend_name(app_runtime) -> str:
    """Return "real" when orchestrator.live can be imported and we are not in demo mode."""
    if app_runtime.is_demo:
        return "demo"
    try:
        from orchestrator import live  # noqa: F401
    except ImportError:
        return "demo"
    return "real"


def make_live_call(app_runtime, classifier: str | None = None) -> Any:
    """The live call for one WebSocket: the real LiveCall, or the scripted demo.

    `classifier` is a registry name picked on the page; None keeps the one in use.
    """
    if live_backend_name(app_runtime) == "real":
        from nlp_core import registry
        from orchestrator.live import LiveCall

        if classifier:
            registry.get("classifier", classifier)  # load it now, so a bad model fails here
            # shortcut: one Runtime for every call, so the last call started picks the models
            # for all calls in progress; give each LiveCall its own Runtime if two people share
            # the server.
            app_runtime.runtime.classifier = classifier
        return LiveCall(app_runtime.sessions)
    return DemoLiveCall(speed=float(os.environ.get("CALLAS_LIVE_SPEED", "1")))


async def serve(ws: WebSocket, call: Any) -> None:
    """Pass audio and controls from the browser to `call` and its events back, until it ends."""
    forwarder = asyncio.create_task(_forward_events(ws, call))
    incoming: asyncio.Future | None = None
    try:
        while True:
            if incoming is None:
                incoming = asyncio.ensure_future(ws.receive())
            await asyncio.wait({incoming, forwarder}, return_when=asyncio.FIRST_COMPLETED)
            if not incoming.done():
                break  # the forwarder finished (call ended or failed)
            message, incoming = incoming.result(), None
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                await run_in_threadpool(call.feed, message["bytes"])
            elif not await _handle_control(ws, call, message.get("text")):
                break
        # after end(), poll_events returns the closing events, including "ended"
        await asyncio.wait_for(forwarder, timeout=5)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if incoming is not None:
            incoming.cancel()
        forwarder.cancel()
        with contextlib.suppress(Exception):
            await run_in_threadpool(call.end)


async def _forward_events(ws: WebSocket, call: Any) -> None:
    """Send the call's events to the browser until the call reports "ended" or the socket fails."""
    ended = False
    while not ended:
        try:
            for event in await run_in_threadpool(call.poll_events):
                await ws.send_json(event)
                ended = ended or event.get("type") == "ended"
        except (WebSocketDisconnect, RuntimeError):
            return
        except Exception:
            log.exception("live call failed")
            with contextlib.suppress(Exception):
                await ws.send_json(
                    {"type": "error", "turn": None, "message": "The live call stopped."}
                )
            return
        if not ended:
            await asyncio.sleep(POLL_SECONDS)


async def _handle_control(ws: WebSocket, call: Any, text: str | None) -> bool:
    """Apply one JSON control message. Returns False when the browser ended the call."""
    try:
        kind = json.loads(text or "").get("type")
    except (ValueError, AttributeError):
        kind = None
    if kind == "pause":
        await run_in_threadpool(call.pause)
    elif kind == "resume":
        await run_in_threadpool(call.resume)
    elif kind == "end":
        await run_in_threadpool(call.end)
        return False
    else:
        await ws.send_json({"type": "error", "turn": None, "message": "Unknown message."})
    return True
