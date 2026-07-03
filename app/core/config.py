from __future__ import annotations

from functools import lru_cache
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    app_name: str = "pantra-backend"
    app_version: str = "0.1.0"
    app_env: str = "development"
    app_debug: bool = True
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    supabase_url: AnyHttpUrl | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(default=None, alias="SUPABASE_SERVICE_ROLE_KEY")

    google_genai_api_key: str | None = Field(
        default=None,
        alias="GOOGLE_GENERATIVE_AI_API_KEY",
    )
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.0
    gemini_max_tokens: int = 1000
    gemini_max_retries: int = 2
    gemini_embeddings_model: str = "gemini-embedding-001"
    gemini_embeddings_output_dimensionality: int = 768


@lru_cache
def get_settings() -> Settings:
    return Settings()
