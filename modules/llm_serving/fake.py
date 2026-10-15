"""Scripted LLM for tests in every module. No network."""

from __future__ import annotations

from collections.abc import Callable

from .base import JOBS

Response = str | list[str] | Callable[[str], str]

DEFAULTS: dict[str, str] = {
    "summary": "สรุปการสนทนา",
    "classify": "normal",
    "reply": "ขอบคุณที่โทรมาค่ะ",
}


class FakeLLM:
    """responses: job -> str, list of str (used in order, last repeats), or callable(context)."""

    def __init__(self, responses: dict[str, Response] | None = None) -> None:
        self.responses: dict[str, Response] = {**DEFAULTS, **(responses or {})}
        self.calls: list[tuple[str, str]] = []
        self._index: dict[str, int] = {}

    def complete(self, job: str, context: str) -> str:
        if job not in JOBS:
            raise ValueError(f"unknown job {job!r}")
        self.calls.append((job, context))
        response = self.responses[job]
        if callable(response):
            return response(context)
        if isinstance(response, list):
            position = self._index.get(job, 0)
            self._index[job] = position + 1
            return response[min(position, len(response) - 1)]
        return response

    def calls_for(self, job: str) -> list[str]:
        """Return the contexts the given job was called with, in order."""
        return [context for call_job, context in self.calls if call_job == job]
