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
_LABEL_RES = {lab: re.compile(rf"(?<![a-z_]){lab}(?![a-z_])") for lab in LABELS}

TEMPERATURE = {"classify": 0.0, "summary": 0.0, "reply": 0.3}


class LLMUnavailable(RuntimeError):
    """The LLM server could not be reached (connection error or timeout)."""


def parse_label(raw: str) -> str:
    """Map model output to one of LABELS, or "unknown". Never guesses.

    Accepts quotes, punctuation or extra words only when exactly one known label appears.
    """
    text = (raw or "").strip().lower()
    hits = [lab for lab, rx in _LABEL_RES.items() if rx.search(text)]
    return hits[0] if len(hits) == 1 else "unknown"


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        s = get_settings()
        self.base_url = base_url or s.vllm_base_url
        self._model = model if model is not None else s.vllm_model
        key = api_key if api_key is not None else s.vllm_api_key.get_secret_value()
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=key or "EMPTY",
            timeout=timeout if timeout is not None else s.vllm_timeout_s,
            max_retries=0,
        )

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMUnavailable(f"cannot reach LLM server at {self.base_url}: {e}") from e

    @property
    def model(self) -> str:
        if not self._model:
            models = self.list_models()
            if not models:
                raise LLMUnavailable(f"LLM server at {self.base_url} lists no models")
            self._model = models[0]
        return self._model

    def complete(self, job: str, context: str) -> str:
        if job not in JOBS:
            raise ValueError(f"unknown job {job!r}; expected one of {sorted(JOBS)}")
        template, version = load_prompt(job)
        prompt = render(template, context)
        model = self.model
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE[job],
                max_tokens=JOBS[job],
            )
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            raise LLMUnavailable(f"cannot reach LLM server at {self.base_url}: {e}") from e
        latency = time.perf_counter() - t0
        out = (resp.choices[0].message.content or "").strip()
        log.info(
            "llm call model=%s job=%s prompt=%s context_chars=%d output_chars=%d latency_s=%.3f",
            model,
            job,
            version,
            len(context),
            len(out),
            latency,
        )
        return out
