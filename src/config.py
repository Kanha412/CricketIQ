"""
CricketIQ — Application Configuration
======================================
All settings are loaded from environment variables (or .env file).
Pydantic validates types at startup — the app will not start if
required variables are missing or have wrong types.

Usage:
    from src.config import settings
    db_url = settings.database_url
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — the directory containing this file's parent
PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    All fields map directly to a variable in .env file.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # DATABASE_URL == database_url
        extra="ignore",         # Don't raise error for unknown env vars
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        description="PostgreSQL connection string. "
                    "Format: postgresql://user:password@host:port/dbname"
    )

    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="gemini",
        description="LLM provider to use. Options: gemini, groq, ollama"
    )
    llm_model: str = Field(
        default="gemini-2.0-flash",
        description="Model name within the chosen provider"
    )
    llm_api_key: str = Field(
        default="",
        description="API key for LLM provider. Not required for ollama."
    )

    # ── Embeddings (for RAG — V2) ─────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model name. Runs locally, no API key needed."
    )
    embedding_provider: str = Field(
        default="sentence_transformers",
        description="Embedding provider. Options: sentence_transformers"
    )

    # ── Simulation ────────────────────────────────────────────────────────────
    n_simulations: int = Field(
        default=10_000,
        ge=100,
        le=50_000,
        description="Number of Monte Carlo simulation trials. Capped at 50,000."
    )
    simulation_random_seed: int = Field(
        default=42,
        description="Random seed for reproducible simulations."
    )

    # ── Application ───────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level. Options: DEBUG, INFO, WARNING, ERROR"
    )
    environment: str = Field(
        default="development",
        description="Runtime environment. Options: development, production"
    )
    app_name: str = Field(
        default="CricketIQ",
        description="Application name shown in UI and logs"
    )

    # ── Data Paths ────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "raw",
        description="Directory for raw downloaded Cricsheet files"
    )
    data_processed_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "processed",
        description="Directory for processed/intermediate files"
    )
    data_samples_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "samples",
        description="Directory for sample match files used in tests"
    )
    models_dir: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Directory for trained ML model artifacts"
    )

    # ── Streamlit / UI ────────────────────────────────────────────────────────
    streamlit_theme: str = Field(
        default="dark",
        description="Streamlit theme. Options: light, dark"
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"gemini", "groq", "ollama"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"environment must be one of {allowed}, got '{v}'")
        return v.lower()

    # ── Computed Properties ───────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        """True when running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """True when running in development environment."""
        return self.environment == "development"

    @property
    def cricsheet_ipl_dir(self) -> Path:
        """Convenience path to raw IPL data directory."""
        return self.data_raw_dir / "cricsheet" / "ipl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Cached after first call — settings are loaded once at startup.

    Use this function everywhere instead of instantiating Settings() directly.
    This makes it easy to override settings in tests.

    Example:
        from src.config import get_settings
        settings = get_settings()
    """
    return Settings()


# Module-level convenience alias
# Import this directly: from src.config import settings
settings = get_settings()
