"""The LLM interface used by the nlp, orchestrator and reply code (not by the DSPy programs)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class JobSettings:
    """How one job calls the model."""

    max_tokens: int
    temperature: float


# Adding a job also needs a prompt file (prompts/<job>.v1.txt). Classify and summary are
# deterministic; the reply gets a little variety.
JOB_SETTINGS: dict[str, JobSettings] = {
    "summary": JobSettings(max_tokens=200, temperature=0.0),
    "classify": JobSettings(max_tokens=20, temperature=0.0),
    "reply": JobSettings(max_tokens=80, temperature=0.3),
}
# Job name -> max output tokens, kept for the code that imports JOBS.
JOBS: dict[str, int] = {job: s.max_tokens for job, s in JOB_SETTINGS.items()}


class LLM(Protocol):
    """What the rest of the app needs from an LLM: one text answer per job."""

    def complete(self, job: str, context: str) -> str: ...
