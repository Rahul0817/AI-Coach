"""Typed application configuration.

Every tunable in Oviora AI is funnelled through this single settings object so
that no module ever reads ``os.environ`` directly. That gives us three things
that matter in production:

* **Validation at boot.** A malformed port or a missing secret fails the
  process on startup rather than at 3am inside a request handler.
* **One place to audit.** Security reviews only have to read this file to know
  every external dependency and credential the service touches.
* **Testability.** Tests override the cached ``get_settings()`` dependency
  instead of mutating global environment state.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]
LLMProviderName = Literal["openai", "local"]
EmbeddingProviderName = Literal["openai", "local"]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Oviora AI"
    environment: Environment = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # ------------------------------------------------------------- security
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(64))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    bcrypt_rounds: int = 12

    backend_cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ------------------------------------------------------------- database
    postgres_user: str = "oviora"
    postgres_password: str = "oviora_dev_password"
    postgres_db: str = "oviora"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # ---------------------------------------------------------------- redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # --------------------------------------------------------- rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    chat_rate_limit_requests: int = 20
    chat_rate_limit_window_seconds: int = 60

    # ------------------------------------------------------------------ llm
    llm_provider: LLMProviderName = "local"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    llm_max_tokens: int = 1200
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # ------------------------------------------------------------------ rag
    embedding_provider: EmbeddingProviderName = "local"
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "./chroma_store"
    chroma_collection: str = "oviora_pcos_knowledge"
    rag_top_k: int = 5
    rag_min_score: float = 0.15

    # ------------------------------------------------------------------- ml
    ml_artifact_dir: str = "./ml/artifacts"
    ml_model_file: str = "pcos_risk_model.joblib"

    # --------------------------------------------------------------- memory
    memory_window_messages: int = 12
    memory_summary_trigger: int = 20
    memory_ttl_seconds: int = 86_400

    # -------------------------------------------------------------- storage
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""
    local_upload_dir: str = "./uploads"
    max_upload_mb: int = 10

    # ------------------------------------------------------------------ ocr
    tesseract_cmd: str = "tesseract"

    # ------------------------------------------------------------ validators
    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @field_validator("llm_temperature")
    @classmethod
    def _valid_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return value

    @model_validator(mode="after")
    def _guard_production(self) -> "Settings":
        """Refuse to boot a production process with development defaults.

        The most common way a student project becomes a security incident is
        shipping the sample signing key. Failing loudly here makes that
        impossible.
        """
        if self.environment == "production":
            if "change-me" in self.secret_key or len(self.secret_key) < 32:
                raise ValueError(
                    "SECRET_KEY must be a strong random value in production."
                )
            if self.debug:
                raise ValueError("DEBUG must be false in production.")
            if self.llm_provider == "openai" and not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        return self

    # -------------------------------------------------------- derived values
    @property
    def sqlalchemy_uri(self) -> str:
        """Async SQLAlchemy URI, assembled from parts unless fully overridden."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_sqlalchemy_uri(self) -> str:
        """Synchronous URI — Alembic migrations run outside the event loop."""
        return self.sqlalchemy_uri.replace("+asyncpg", "+psycopg2")

    @property
    def redis_uri(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.backend_cors_origins.split(",") if o.strip()]

    @property
    def cloudinary_enabled(self) -> bool:
        return bool(
            self.cloudinary_cloud_name
            and self.cloudinary_api_key
            and self.cloudinary_api_secret
        )

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton.

    ``lru_cache`` means the ``.env`` file is parsed exactly once. Tests clear
    the cache with ``get_settings.cache_clear()`` when they need a fresh view.
    """
    return Settings()


settings = get_settings()
