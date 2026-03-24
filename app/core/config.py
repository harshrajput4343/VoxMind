"""
Configuration management using Pydantic Settings.

Loads environment variables from .env file with sensible defaults.
All API keys are optional — the app works fully without them.
"""

from pydantic_settings import BaseSettings
from pydantic import computed_field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # Gemini TTS (optional — PRIMARY TTS, free at aistudio.google.com)
    gemini_api_key: str = ""

    # ElevenLabs (optional)
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"

    # HuggingFace (optional)
    hf_token: str = ""

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    output_dir: str = "outputs"
    model_cache_dir: str = "model_cache"
    max_text_length: int = 500

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @computed_field
    @property
    def output_path(self) -> Path:
        """Return resolved Path to outputs directory, creating it if needed."""
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @computed_field
    @property
    def model_cache_path(self) -> Path:
        """Return resolved Path for HuggingFace model cache."""
        p = Path(self.model_cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def is_gemini_configured(self) -> bool:
        """Return True only if a non-empty Gemini API key is set."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    def is_elevenlabs_configured(self) -> bool:
        """Return True only if a non-empty ElevenLabs API key is set."""
        return bool(self.elevenlabs_api_key and self.elevenlabs_api_key.strip())

    def is_development(self) -> bool:
        """Return True if the app is running in development mode."""
        return self.app_env == "development"


settings = Settings()
