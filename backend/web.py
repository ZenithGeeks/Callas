"""Things every page shares: the templates, `asset()`, the page renderers and the upload folder.

Routers import from here. Nothing here imports a router, so there is no import cycle.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import views

ROOT = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(ROOT / "templates"))
templates.env.trim_blocks = True
templates.env.lstrip_blocks = True


def asset(path: str) -> str:
    """URL of a static file with its modification time attached, so the browser fetches a
    rebuilt app.css or edited script instead of serving its cached copy. Path only (no host),
    so the page works behind an https proxy."""
    try:
        version = int((ROOT / "static" / path).stat().st_mtime)
    except OSError:
        version = 0
    return f"/static/{path}?v={version}"


templates.env.globals.update(
    fmt_time=views.fmt_time,
    who=views.who,
    segments=views.segments,
    entity_rows=views.entity_rows,
    asset=asset,
)


# Uploaded recordings live only in this folder. It is emptied at startup and shutdown, so a
# crash or restart never leaves call audio behind.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "callas-uploads"


def clear_uploads() -> None:
    """Empty the upload folder (and create it if it is missing)."""
    if UPLOAD_DIR.is_dir():
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


def render(name: str, **ctx) -> str:
    """A template as a string, for partials and for code that builds its own response."""
    return templates.env.get_template(name).render(**ctx)


def page(request: Request, name: str, active: str, **ctx) -> HTMLResponse:
    """A full page. `active` marks the nav link; `demo` shows the demo badge."""
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
