"""Load versioned prompt files: prompts/<job>.v<N>.txt, highest N wins.

Lines starting with `#` are comments and are removed. `{context}` is filled with
str.replace (not str.format), so braces in call text cannot break a prompt.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"
PLACEHOLDER = "{context}"


def strip_comments(raw: str) -> str:
    lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines).strip("\n") + "\n"


@cache
def load_prompt(job: str, prompt_dir: Path = PROMPT_DIR) -> tuple[str, str]:
    """Return (prompt text with {context} placeholder, version like 'v1')."""
    pattern = re.compile(rf"^{re.escape(job)}\.v(\d+)\.txt$")
    found = [(int(m.group(1)), p) for p in prompt_dir.iterdir() if (m := pattern.match(p.name))]
    if not found:
        raise FileNotFoundError(f"no prompt file for job {job!r} in {prompt_dir}")
    number, path = max(found, key=lambda t: t[0])
    text = strip_comments(path.read_text(encoding="utf-8"))
    if PLACEHOLDER not in text:
        raise ValueError(f"{path.name} has no {PLACEHOLDER} placeholder")
    return text, f"v{number}"


def render(template: str, context: str) -> str:
    return template.replace(PLACEHOLDER, context)
