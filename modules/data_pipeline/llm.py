"""The Typhoon LM for simulator agents whose agents.yaml `lm.model` is a typhoon-... model.

Which agent uses which model is set in config/agents.yaml. agents.build_lms checks each
agent's model: a typhoon-... model is built here, every other model (or an empty one, meaning
VLLM_MODEL) by llm_serving.dspy_programs.build_lm on the team's vLLM server.

Typhoon's server and key come from your own .env:

    TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1
    TYPHOON_API_KEY=...
    TYPHOON_TIMEOUT_S=30          (optional, else VLLM_TIMEOUT_S)

Without TYPHOON_BASE_URL and TYPHOON_API_KEY a typhoon-... agent runs on the vLLM server with
VLLM_MODEL instead, with no error: teammates without a Typhoon key need no setup.
temperature and max_tokens come from agents.yaml whichever server is used.

Each call's record stores the model of every agent (Agents.models), so Typhoon calls can be told apart.
"""

from __future__ import annotations

import os
from typing import Any

TYPHOON_PREFIX = "typhoon"  # model names that start with this run on Typhoon's server


def is_typhoon(model: str | None) -> bool:
    """True for a typhoon-... model name from agents.yaml."""
    return (model or "").startswith(TYPHOON_PREFIX)


def env(name: str) -> str:
    """A .env value from the environment, else from .env; "" when it is not set.

    Settings (callas_settings) ignores .env lines it does not declare, so .env is read here.
    """
    from dotenv import dotenv_values

    from callas_settings import _ENV_FILE

    return os.environ.get(name) or dotenv_values(_ENV_FILE).get(name) or ""


def build_typhoon_lm(model: str, max_tokens: int, temperature: float) -> Any:
    """The dspy.LM for a typhoon-... model: on Typhoon's server when its .env lines are set,
    else VLLM_MODEL on the vLLM server (which does not serve Typhoon models)."""
    base_url, key = env("TYPHOON_BASE_URL"), env("TYPHOON_API_KEY")
    if not (base_url and key):
        from llm_serving.dspy_programs import build_lm

        return build_lm(model=None, max_tokens=max_tokens, temperature=temperature)

    import dspy

    from callas_settings import get_settings

    # the model from agents.yaml, the server and key from .env.
    # Same options as the vLLM LM, minus the vLLM-only extra_body (thinking switch).
    return dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=float(env("TYPHOON_TIMEOUT_S") or get_settings().vllm_timeout_s),
        num_retries=0,
        cache=False,
    )
