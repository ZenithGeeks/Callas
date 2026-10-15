"""The "Check a message" page.

URLs: GET /message (the form), POST /message/check (the htmx result panel).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from llm_serving.client import LLMUnavailable
from orchestrator import ToolCallingUnavailable

from . import views
from .web import form_error, page, render

log = logging.getLogger("callas.web")

router = APIRouter()

MAX_MESSAGE_CHARS = 2000


@router.get("/message", response_class=HTMLResponse)
async def message_page(request: Request) -> HTMLResponse:
    """The page with the message form."""
    return page(request, "message.html", "message", max_chars=MAX_MESSAGE_CHARS)


@router.post("/message/check", response_class=HTMLResponse)
async def message_check(request: Request, text: str = Form("")) -> HTMLResponse:
    """The verdict panel for one message, or a form error."""
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
