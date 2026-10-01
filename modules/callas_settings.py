"""Runtime settings, read from environment variables and the gitignored .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = ""
    vllm_api_key: SecretStr = SecretStr("EMPTY")
    vllm_timeout_s: float = 30.0

    asr_model: str = "typhoon_asr"

    tts_pretrained: str = "fastthaig2p"
    tts_speaker: str = "thai_som"


@lru_cache
def get_settings() -> Settings:
    return Settings()
