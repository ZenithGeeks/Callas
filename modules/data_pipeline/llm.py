"""The Typhoon LM for simulator agents whose agents.yaml `lm.model` is a typhoon-... model.

Which agent uses which model is set in config/agents.yaml. agents.build_lms checks each
agent's model: a typhoon-... model is built here, every other model (or an empty one, meaning
VLLM_MODEL) by llm_serving.dspy_programs.build_lm on the team's vLLM server.

Typhoon's server and key come from your own .env (callas_settings.Settings):

    TYPHOON_BASE_URL=https://api.opentyphoon.ai/v1
    TYPHOON_API_KEY=...
    TYPHOON_TIMEOUT_S=30          (optional, else VLLM_TIMEOUT_S)

Without TYPHOON_BASE_URL and TYPHOON_API_KEY a typhoon-... agent raises an error instead of
quietly running another model: the config hash only names the model, so a silent fallback
would let the same hash mean different models. To run without a Typhoon key, leave that
agent's lm.model empty in agents.yaml (VLLM_MODEL). temperature and max_tokens come from
agents.yaml.

Each call's record stores the model of every agent (Agents.models), so Typhoon calls can be
told apart.
"""

from __future__ import annotations

from typing import Any

TYPHOON_PREFIX = "typhoon"  # model names that start with this run on Typhoon's server


class TyphoonNotConfigured(RuntimeError):
    """agents.yaml asks for a typhoon-... model but .env has no Typhoon server or key."""


def is_typhoon(model: str | None) -> bool:
    """True for a typhoon-... model name from agents.yaml."""
    return (model or "").startswith(TYPHOON_PREFIX)


def build_typhoon_lm(model: str, max_tokens: int, temperature: float) -> Any:
    """The dspy.LM for a typhoon-... model on Typhoon's server.

    Raises:
        TyphoonNotConfigured: TYPHOON_BASE_URL or TYPHOON_API_KEY is not set.
    """
    from callas_settings import get_settings

    settings = get_settings()
    base_url = settings.typhoon_base_url
    key = settings.typhoon_api_key.get_secret_value()
    if not (base_url and key):
        raise TyphoonNotConfigured(
            f"agents.yaml asks for {model}, but TYPHOON_BASE_URL and TYPHOON_API_KEY are not "
            "set in .env. Set them, or leave that agent's lm.model empty to use the vLLM server."
        )

    import dspy

    # the model from agents.yaml, the server and key from .env.
    # Same options as the vLLM LM, minus the vLLM-only extra_body (thinking switch).
    return dspy.LM(
        f"openai/{model}",
        api_base=base_url,
        api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.typhoon_timeout_s or settings.vllm_timeout_s,
        num_retries=0,
        cache=False,
    )
