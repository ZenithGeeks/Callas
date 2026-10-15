"""The one interface every LLM call goes through."""

from __future__ import annotations

from typing import Protocol

# Job name -> max output tokens. Adding a job also needs a prompt file and a temperature.
JOBS: dict[str, int] = {"summary": 200, "classify": 20, "reply": 80}


class LLM(Protocol):
    """What the rest of the app needs from an LLM: one text answer per job."""

    def complete(self, job: str, context: str) -> str: ...
