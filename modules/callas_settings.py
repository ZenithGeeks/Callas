"""Runtime settings, read from environment variables and the gitignored .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives at the repo root; resolve it so CLIs work from any working directory.
_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

_REPLY_ENGINES = ("prompt", "dspy")
_LEAK_JUDGES = ("off", "on")
DIARIZERS = ("nemotron", "pyannote_segmentation")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = ""
    vllm_api_key: SecretStr = SecretStr("EMPTY")
    vllm_timeout_s: float = 30.0
    # Thinking models (Qwen3 and similar) spend the whole small token budget on hidden
    # reasoning and return no answer. Off sends chat_template_kwargs.enable_thinking=false.
    vllm_thinking: bool = False

    def llm_extra_body(self) -> dict:
        """Extra request fields for every vLLM chat call."""
        return {} if self.vllm_thinking else {"chat_template_kwargs": {"enable_thinking": False}}

    # How the voice reply is written: "prompt" (llm_serving prompt) or "dspy" (ChainOfThought).
    reply_engine: str = "prompt"
    # Extra leak judge on top of the rule check: "off" or "on". It can only add blocks.
    leak_judge: str = "off"

    asr_model: str = "typhoon_asr"

    # Speaker-split model: "nemotron" (Nemotron-3 diarization) or "pyannote_segmentation".
    diarizer: str = "nemotron"

    tts_pretrained: str = "fastthaig2p"
    tts_speaker: str = "thai_som"

    @field_validator("reply_engine", "leak_judge", "diarizer", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("reply_engine")
    @classmethod
    def _check_engine(cls, v: str) -> str:
        if v not in _REPLY_ENGINES:
            raise ValueError(f"REPLY_ENGINE must be one of {_REPLY_ENGINES}, got {v!r}")
        return v

    @field_validator("diarizer")
    @classmethod
    def _check_diarizer(cls, v: str) -> str:
        if v not in DIARIZERS:
            raise ValueError(f"DIARIZER must be one of {DIARIZERS}, got {v!r}")
        return v

    @field_validator("leak_judge")
    @classmethod
    def _check_judge(cls, v: str) -> str:
        if v not in _LEAK_JUDGES:
            raise ValueError(f"LEAK_JUDGE must be one of {_LEAK_JUDGES}, got {v!r}")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
