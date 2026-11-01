"""The "Review the dataset" page: browse simulated calls and approve or reject them.

URLs: GET /dataset, GET /dataset/call/{id}, POST /dataset/call/{id}/review,
GET /dataset/audio/{id} and /dataset/audio/{id}/{name}. The store reading lives in dataset.py.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from data_pipeline.usage import summary

from . import dataset
from .web import page, render

router = APIRouter()


def _known_type(value: str) -> str:
    """The call type filter if it names a real type, else "" (meaning all types)."""
    return value if value in dataset.CALL_TYPE_NAMES else ""


def _from_this_site(request: Request) -> bool:
    """True for a request our own page made with htmx. The app has no login, so a form on
    another site must not be able to post a review: browsers add Origin to cross-site posts,
    and a plain form cannot set HX-Request."""
    if request.headers.get("HX-Request") != "true":
        return False
    origin = request.headers.get("origin")
    return origin is None or urlsplit(origin).netloc == request.headers.get("host")


# These handlers are plain `def` on purpose: SQLite calls block (busy_timeout is 30 s), and
# FastAPI runs a plain def in its threadpool instead of on the event loop.
@router.get("/dataset", response_class=HTMLResponse)
def dataset_page(
    request: Request, type: str = "", view: str = dataset.DEFAULT_VIEW, id: str = ""
) -> HTMLResponse:
    """The review page, or just the queue when htmx asks for it."""
    db = dataset.store()
    call_type = _known_type(type)
    listing = dataset.listing(db, call_type, view) if db else dataset.empty_listing(view)
    ctx = dict(
        listing=listing,
        call_type=call_type,
        views=dataset.VIEWS,
        type_names=dataset.CALL_TYPE_NAMES,
        folder=dataset.db_path(),
        missing=db is None,
        selected=id,
    )
    # The type filter and view tabs swap only the queue.
    if request.headers.get("HX-Target") == "queue":
        return HTMLResponse(render("partials/dataset_queue.html", **ctx))
    # Full page: open the requested call, else the first one in the queue.
    open_id = id or (listing.rows[0].id if listing.rows else "")
    record = dataset.load(db, open_id) if db and open_id else None
    ctx["selected"] = record["id"] if record else ""
    ctx["d"] = dataset.detail(db, record) if record else None
    return page(request, "dataset.html", "dataset", **ctx)


@router.get("/dataset/tokens", response_class=HTMLResponse)
def dataset_tokens(type: str = "") -> HTMLResponse:
    """The token-usage block: per agent and per call type, over the stored calls (htmx)."""
    db = dataset.store()
    call_type = _known_type(type)
    records = list(db.call_dicts(call_type)) if db else []
    return HTMLResponse(
        render(
            "partials/dataset_tokens.html",
            s=summary(records),
            call_type=call_type,
            type_names=dataset.CALL_TYPE_NAMES,
        )
    )


@router.get("/dataset/call/{record_id}", response_class=HTMLResponse)
def dataset_call(request: Request, record_id: str) -> Response:
    """One call's detail panel (htmx), or a redirect to the full page for a plain request."""
    db = dataset.store()
    record = dataset.load(db, record_id) if db else None
    if record is None:
        return HTMLResponse("Call not found.", status_code=404)
    if request.headers.get("HX-Request") != "true":
        return RedirectResponse(f"/dataset?id={record_id}", status_code=303)
    return HTMLResponse(render("partials/dataset_call.html", d=dataset.detail(db, record)))


@router.post("/dataset/call/{record_id}/review", response_class=HTMLResponse)
def dataset_review(
    request: Request,
    record_id: str,
    status: str = Form(...),
    note: str = Form(""),
    keep_note: str = Form(""),
    type: str = Form(""),
    view: str = Form(dataset.DEFAULT_VIEW),
) -> HTMLResponse:
    """Save a review and return the review bar plus the out-of-band queue updates."""
    if not _from_this_site(request):
        return HTMLResponse("Not allowed.", status_code=403)
    db = dataset.store()
    if status not in dataset.REVIEW_STATES:
        return HTMLResponse("Unknown review status.", status_code=422)
    # FastAPI reads an empty field as missing, so "no note sent" needs its own flag. Only the
    # toast's Undo sets it; the bar's own form always sends the note box, empty or not.
    keep = status == "pending" and bool(keep_note)
    record = dataset.save_review(db, record_id, status, None if keep else note) if db else None
    if record is None:
        return HTMLResponse("Call not found.", status_code=404)
    # The review bar, plus the queue row, progress line and view counts, swapped out-of-band.
    # They follow the type and view the page is showing (the form includes both).
    shown_type = _known_type(type)
    progress = dataset.listing(db, shown_type, view)
    return HTMLResponse(
        render(
            "partials/dataset_review.html",
            d=dataset.detail(db, record),
            oob=True,
            listing=progress,
            listing_type=shown_type,
            views=dataset.VIEWS,
            in_view=any(r.id == record["id"] for r in progress.rows),
        )
    )


@router.get("/dataset/audio/{record_id}")
@router.get("/dataset/audio/{record_id}/{name}")
def dataset_audio(record_id: str, name: str | None = None) -> Response:
    """The call's audio file (whole call, or one turn by name)."""
    db = dataset.store()
    path = dataset.audio_file(db, record_id, name) if db else None
    if path is None:
        return Response(status_code=404)
    return FileResponse(path, media_type="audio/wav")
