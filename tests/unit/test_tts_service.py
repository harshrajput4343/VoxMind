"""
Unit tests for TTSService.

Tests:
  - Provider availability checks
  - Provider naming
  - Synthesis output and fallback behaviour
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from app.services.tts_service import GTTSProvider, Pyttsx3Provider, TTSService


class TestProviders:
    """Tests for individual TTS providers."""

    def test_gtts_available(self) -> None:
        """GTTSProvider should report available when gtts is installed."""
        provider = GTTSProvider()
        # gtts should be installed per requirements.txt
        assert provider.is_available() is True

    def test_pyttsx3_provider_name(self) -> None:
        """Pyttsx3Provider should have name 'pyttsx3'."""
        provider = Pyttsx3Provider()
        assert provider.name == "pyttsx3"


class TestTTSService:
    """Tests for the TTSService fallback chain."""

    def test_service_returns_provider(self, sample_wav: str, tmp_path) -> None:
        """Synthesize should return a tuple of (path, provider_name)."""
        service = TTSService()
        out = str(tmp_path / "out.wav")
        result = service.synthesize("Hello world", out)
        assert isinstance(result, tuple)
        assert len(result) == 2
        path, name = result
        assert isinstance(path, str)
        assert isinstance(name, str)

    def test_output_file_created(self, tmp_path) -> None:
        """Synthesize should create a .wav file at the given output path."""
        service = TTSService()
        out = str(tmp_path / "created.wav")
        service.synthesize("Test audio output.", out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_falls_back_to_pyttsx3(self, tmp_path) -> None:
        """If gTTS fails, pyttsx3 should be used as fallback."""
        service = TTSService()
        out = str(tmp_path / "fallback.wav")

        # Mock gTTS to fail
        for p in service.providers:
            if p.name == "gTTS":
                p.synthesize = MagicMock(side_effect=RuntimeError("gTTS mock failure"))

        _, name = service.synthesize("Fallback test", out)
        assert name != "gTTS"

    def test_all_fail_raises(self) -> None:
        """If all providers fail, RuntimeError should be raised."""
        service = TTSService()
        for p in service.providers:
            p.synthesize = MagicMock(side_effect=RuntimeError("Mock failure"))
            p.is_available = MagicMock(return_value=True)

        with pytest.raises(RuntimeError, match="All TTS providers exhausted"):
            service.synthesize("This should fail", "/tmp/should_not_exist.wav")
