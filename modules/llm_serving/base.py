"""The one interface every LLM call goes through."""

from __future__ import annotations

from typing import Protocol

# job -> max output tokens
JOBS: dict[str, int] = {"summary": 200, "classify": 20, "reply": 80}


class LLM(Protocol):
    def complete(self, job: str, context: str) -> str: ...
