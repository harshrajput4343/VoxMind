"""
Unit tests for AudioService.

Tests:
  - Voice parameter lookup for specific emotion+intensity combos
  - Complete coverage of the emotion→voice mapping table
  - Audio modulation creates valid output files
  - Fallback on unknown emotion
"""

import os

import pytest

from app.services.audio_service import AudioService, EMOTION_VOICE_MAPPING
from app.models.schemas import EmotionResult, VoiceParameters


@pytest.fixture
def audio_service() -> AudioService:
    """AudioService instance for testing."""
    return AudioService()


def _make_emotion(emotion: str, intensity: float = 0.85) -> EmotionResult:
    """Helper to create EmotionResult for testing."""
    label = "high" if intensity >= 0.7 else ("medium" if intensity >= 0.4 else "low")
    return EmotionResult(
        emotion=emotion,
        intensity=intensity,
        all_scores={emotion: intensity},
        model_used="test",
        intensity_label=label,
    )


class TestVoiceParams:
    """Tests for emotion → voice parameter lookup."""

    def test_voice_params_joy_high(self, audio_service: AudioService) -> None:
        """Joy with high intensity should have elevated rate and pitch."""
        result = audio_service.get_voice_params(_make_emotion("joy", 0.9))
        assert result.rate > 1.2
        assert result.pitch > 3.0

    def test_voice_params_sadness_high(self, audio_service: AudioService) -> None:
        """Sadness with high intensity should have low rate and negative pitch."""
        result = audio_service.get_voice_params(_make_emotion("sadness", 0.9))
        assert result.rate < 0.8
        assert result.pitch < -3.0

    def test_voice_params_neutral(self, audio_service: AudioService) -> None:
        """Neutral should have default parameters (rate=1.0, pitch=0.0)."""
        result = audio_service.get_voice_params(_make_emotion("neutral", 0.5))
        assert result.rate == 1.0
        assert result.pitch == 0.0

    def test_all_emotions_mapped(self) -> None:
        """Every canonical emotion should have entries in the mapping table."""
        emotions = {"joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"}
        mapped_emotions = {key[0] for key in EMOTION_VOICE_MAPPING}
        assert emotions.issubset(mapped_emotions)

    def test_fallback_on_unknown_emotion(self, audio_service: AudioService) -> None:
        """Unknown emotion should return neutral-like params without raising."""
        result = audio_service.get_voice_params(_make_emotion("confusion", 0.5))
        assert result.rate == 1.0
        assert result.pitch == 0.0


class TestAudioModulation:
    """Tests for audio file modulation."""

    def test_modulate_creates_file(self, audio_service: AudioService, sample_wav: str, tmp_path) -> None:
        """Modulation should produce an output file."""
        out = str(tmp_path / "modulated.wav")
        params = VoiceParameters(rate=1.0, pitch=0.0, volume=1.0, emotion="neutral", intensity_label="medium")
        audio_service.modulate_audio(sample_wav, params, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_volume_change_applied(self, audio_service: AudioService, sample_wav: str, tmp_path) -> None:
        """Volume modulation should create valid output without error."""
        out = str(tmp_path / "loud.wav")
        params = VoiceParameters(rate=1.0, pitch=0.0, volume=1.4, emotion="anger", intensity_label="high")
        audio_service.modulate_audio(sample_wav, params, out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
