"""The "Live call" page and its WebSocket.

URLs: GET /live (the page), GET /live/script (a test-split call to read aloud, test_calls.py),
GET /live/script/{call_id}/{line}.wav (that line spoken by the TTS, for the page's TTS caller),
WebSocket /ws/live?gate=<encoder>&type=<encoder> (audio in, analysis events out). The socket
loop lives in live_ws.py.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool

from nlp_core import registry

from . import live_ws, test_calls, views
from .web import page, render

log = logging.getLogger("callas.web")

router = APIRouter()

METRICS = Path(__file__).resolve().parents[1] / "models" / "scam_classifier_metrics.json"
TASKS = {"gate": "scam_gate", "type": "scam_type"}
# PyThaiTTS refuses more than 510 phonemes, about 250 Thai characters (2 phonemes each).
TTS_CHUNK_CHARS = 120


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
        model_choices=model_choices(
            request.app.state.app_runtime.runtime.classifier or registry.default_name("classifier")
        ),
        models=request.app.state.app_runtime.models(),
        call=None if is_demo else test_calls.random_test_call(),
    )


def model_choices(classifier: str) -> dict[str, list[dict]] | None:
    """The gate and type dropdowns: each encoder with its test macro-F1, `classifier` selected.

    None when `classifier` is not a fine-tuned one (NLP_CLASSIFIER=llm, say).
    """
    pair = registry.fine_tuned_pair(classifier)
    if pair is None:
        return None
    f1 = {}
    if METRICS.exists():
        results = json.loads(METRICS.read_text(encoding="utf-8"))["results"]
        f1 = {(r["task"], r["model"]): r["f1"] for r in results}
    return {
        task: [
            {
                "value": encoder,
                "label": label,
                "f1": f1.get((TASKS[task], encoder)),
                "best": encoder == registry.BEST_FINE_TUNED[i],
                "selected": encoder == pair[i],
            }
            for encoder, label in registry.FINE_TUNED.items()
        ]
        for i, task in enumerate(TASKS)
    }


def chosen_classifier(gate: str | None, type_: str | None) -> str | None:
    """The registry name for the page's gate and type choice; None keeps the default."""
    name = f"{gate}+{type_}"
    return name if registry.fine_tuned_pair(name) else None


@router.get("/live/script", response_class=HTMLResponse)
def live_script() -> HTMLResponse:  # sync: FastAPI runs it in a thread, off the event loop
    """Another random test-split call to read aloud: the inner part of the script section."""
    return HTMLResponse(render("partials/live_script.html", call=test_calls.random_test_call()))


@router.get("/live/script/{call_id}/{line}.wav")
def live_script_line(request: Request, call_id: str, line: int) -> Response:
    """One caller line of a test-split call as WAV, for the page's TTS caller."""
    call = test_calls.find_test_call(call_id)
    if call is None or not 0 <= line < len(call.caller_lines):
        raise HTTPException(404, "No such test call line.")
    speaker = request.app.state.app_runtime.runtime.speaker
    return Response(speak_line(speaker, call.caller_lines[line]), media_type="audio/wav")


def speak_line(speaker, text: str) -> bytes:
    """`text` as one WAV, spoken in chunks short enough for the TTS."""
    # The caller's lines are dataset text, not Callas's reply: their numbers are the scam.
    chunks = tts_chunks(text, TTS_CHUNK_CHARS)
    wavs = [speaker.synthesize_bytes(chunk, check_leaks=False) for chunk in chunks]
    if len(wavs) == 1:
        return wavs[0]
    parts = [sf.read(io.BytesIO(wav), dtype="int16") for wav in wavs]
    out = io.BytesIO()
    sf.write(out, np.concatenate([audio for audio, _ in parts]), parts[0][1], format="WAV")
    return out.getvalue()


def tts_chunks(text: str, limit: int) -> list[str]:
    """`text` split at spaces into chunks of up to `limit` characters; a longer word stays whole."""
    chunks: list[str] = []
    for word in text.split():
        if chunks and len(chunks[-1]) + 1 + len(word) <= limit:
            chunks[-1] += " " + word
        else:
            chunks.append(word)
    return chunks or [text]


@router.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    """Start a live call and pass audio and events through until the browser leaves."""
    await ws.accept()
    # The first real call loads the streaming ASR model. Off the event loop, so other pages
    # keep answering while it loads.
    app_runtime = ws.app.state.app_runtime
    classifier = chosen_classifier(ws.query_params.get("gate"), ws.query_params.get("type"))
    try:
        call = await run_in_threadpool(live_ws.make_live_call, app_runtime, classifier)
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
