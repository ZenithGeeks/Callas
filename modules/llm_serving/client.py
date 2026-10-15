"""OpenAI-compatible client for the vLLM server. Every LLM call goes through here."""

from __future__ import annotations

import logging
import re
import time

import openai
from openai import OpenAI

from callas_settings import get_settings

from .base import JOBS
from .prompts import load_prompt, render

log = logging.getLogger("llm_serving")

LABELS: tuple[str, ...] = (
    "normal",
    "fake_authority",
    "romance",
    "investment",
    "phishing_otp",
    "lottery",
)
_LABEL_RES = {label: re.compile(rf"(?<![a-z_]){label}(?![a-z_])") for label in LABELS}

# Per job. Classify and summary are deterministic; the reply gets a little variety.
TEMPERATURE = {"classify": 0.0, "summary": 0.0, "reply": 0.3}


class LLMUnavailable(RuntimeError):
    """The LLM server could not be reached (connection error or timeout)."""


def parse_label(raw: str) -> str:
    """Map model output to one of LABELS, or "unknown". Never guesses.

    Accepts quotes, punctuation or extra words only when exactly one known label appears.
    """
    text = (raw or "").strip().lower()
    hits = [label for label, pattern in _LABEL_RES.items() if pattern.search(text)]
    return hits[0] if len(hits) == 1 else "unknown"


class LLMClient:
    """Chat client for the vLLM server. `model` is looked up from the server when not set."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = base_url or settings.vllm_base_url
        self._model = model if model is not None else settings.vllm_model
        key = api_key if api_key is not None else settings.vllm_api_key.get_secret_value()
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=key or "EMPTY",
            timeout=timeout if timeout is not None else settings.vllm_timeout_s,
            max_retries=0,
        )

    def list_models(self) -> list[str]:
        """Return the ids of the models the server serves."""
        try:
            return [m.id for m in self._client.models.list().data]
        except (openai.APIConnectionError, openai.APITimeoutError) as error:
            raise LLMUnavailable(f"cannot reach LLM server at {self.base_url}: {error}") from error

    @property
    def model(self) -> str:
        """The configured model name, or the first model the server lists."""
        if not self._model:
            models = self.list_models()
            if not models:
                raise LLMUnavailable(f"LLM server at {self.base_url} lists no models")
            self._model = models[0]
        return self._model

    def complete(self, job: str, context: str) -> str:
        """Run one job ("summary", "classify" or "reply") on the call context."""
        if job not in JOBS:
            raise ValueError(f"unknown job {job!r}; expected one of {sorted(JOBS)}")
        template, version = load_prompt(job)
        prompt = render(template, context)
        model = self.model
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE[job],
                max_tokens=JOBS[job],
                extra_body=get_settings().llm_extra_body() or None,
            )
        except (openai.APIConnectionError, openai.APITimeoutError) as error:
            raise LLMUnavailable(f"cannot reach LLM server at {self.base_url}: {error}") from error
        latency = time.perf_counter() - started
        output = (response.choices[0].message.content or "").strip()
        # lengths and timings only, never call text or reply text
        log.info(
            "llm call model=%s job=%s prompt=%s context_chars=%d output_chars=%d latency_s=%.3f",
            model,
            job,
            version,
            len(context),
            len(output),
            latency,
        )
        return output
