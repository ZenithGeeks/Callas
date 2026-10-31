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
_AUTOCORRECTS = ("off", "pythainlp")
_SPELL_REPAIRS = ("off", "on")
_TRACINGS = ("off", "phoenix")
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

    # Fix misspelled Thai words in each transcript turn: "off" or "pythainlp" (see nlp_core.text).
    autocorrect: str = "off"
    # Last autocorrect step: fix the marks of words still unknown, scored in the sentence by
    # WangchanBERTa: "on" or "off" (see docs/sentence-level-benchmark.md). Off: about 1 change in
    # 5 to 12 was wrong on new calls, so it is kept only as an option.
    spell_repair: str = "off"

    # Speaker-split model: "nemotron" (Nemotron-3 diarization) or "pyannote_segmentation".
    diarizer: str = "nemotron"

    # nlp-core models by registry name; empty means the built-in default (see nlp_core.registry).
    nlp_ner: str = ""
    nlp_classifier: str = ""
    nlp_summarizer: str = ""

    # Local traces: "off" or "phoenix" (see callas_tracing). The endpoint must be on this machine.
    tracing: str = "off"
    phoenix_collector_endpoint: str = "http://localhost:6006"
    # Keep prompts and answers in web traces. Off hides them, so real calls leave no text.
    trace_call_text: bool = False

    tts_pretrained: str = "fastthaig2p"
    tts_speaker: str = "thai_som"

    @field_validator(
        "reply_engine",
        "leak_judge",
        "diarizer",
        "autocorrect",
        "spell_repair",
        "tracing",
        mode="before",
    )
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

    @field_validator("autocorrect")
    @classmethod
    def _check_autocorrect(cls, v: str) -> str:
        if v not in _AUTOCORRECTS:
            raise ValueError(f"AUTOCORRECT must be one of {_AUTOCORRECTS}, got {v!r}")
        return v

    @field_validator("spell_repair")
    @classmethod
    def _check_spell_repair(cls, v: str) -> str:
        if v not in _SPELL_REPAIRS:
            raise ValueError(f"SPELL_REPAIR must be one of {_SPELL_REPAIRS}, got {v!r}")
        return v

    @field_validator("tracing")
    @classmethod
    def _check_tracing(cls, v: str) -> str:
        if v not in _TRACINGS:
            raise ValueError(f"TRACING must be one of {_TRACINGS}, got {v!r}")
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
