"""Environment-based application settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    index_path: Path | None = None
    log_level: str = "INFO"
    port: int = Field(default=8000, ge=1, le=65535)
    search_mode: str = Field(default="def", pattern="^(def|async)$")
    app_name: str = "findex"