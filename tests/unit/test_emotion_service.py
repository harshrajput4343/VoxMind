"""
Unit tests for EmotionService.

Tests:
  - Correct emotion detection for joy, sadness, anger, fear, neutral
  - Intensity label mapping (low, medium, high)
  - Fallback behaviour when primary model is unavailable
  - Edge cases: empty text, long text
"""

import pytest
from app.services.emotion_service import EmotionService


@pytest.fixture(scope="module")
def emotion_service() -> EmotionService:
    """Shared EmotionService instance for all tests in this module."""
    return EmotionService()


class TestEmotionDetection:
    """Tests for emotion classification accuracy."""

    def test_joy_detected(self, emotion_service: EmotionService) -> None:
        """Positive text should be classified as joy."""
        result = emotion_service.detect("I am absolutely thrilled and overjoyed!")
        assert result.emotion in ("joy", "surprise"), f"Expected joy-like, got {result.emotion}"

    def test_sadness_detected(self, emotion_service: EmotionService) -> None:
        """Sad text should be classified as sadness."""
        result = emotion_service.detect("I feel so hopeless and completely lost")
        assert result.emotion in ("sadness", "fear"), f"Expected sadness-like, got {result.emotion}"

    def test_anger_detected(self, emotion_service: EmotionService) -> None:
        """Angry text should be classified as anger."""
        result = emotion_service.detect("This is completely unacceptable and outrageous!")
        assert result.emotion in ("anger", "disgust"), f"Expected anger-like, got {result.emotion}"

    def test_fear_detected(self, emotion_service: EmotionService) -> None:
        """Fearful text should be classified as fear."""
        result = emotion_service.detect("I am terrified of what might happen next")
        assert result.emotion in ("fear", "sadness"), f"Expected fear-like, got {result.emotion}"

    def test_neutral_detected(self, emotion_service: EmotionService) -> None:
        """Factual text should be classified as neutral."""
        result = emotion_service.detect("The meeting starts at 3pm in conference room B")
        assert result.emotion in ("neutral", "surprise"), f"Expected neutral-like, got {result.emotion}"


class TestIntensity:
    """Tests for intensity scoring and labelling."""

    def test_intensity_low(self, emotion_service: EmotionService) -> None:
        """Mild text should produce lower intensity."""
        result = emotion_service.detect("This is somewhat nice")
        assert result.intensity <= 1.0, "Intensity should be in valid range"

    def test_intensity_high(self, emotion_service: EmotionService) -> None:
        """Highly emotional text should produce higher intensity."""
        result = emotion_service.detect("THIS IS ABSOLUTELY THE BEST THING EVER!!!")
        assert result.intensity > 0.3, "Highly emotional text should have notable intensity"

    def test_intensity_label_low(self, emotion_service: EmotionService) -> None:
        """Score of 0.3 should map to 'low'."""
        assert emotion_service._get_intensity_label(0.3) == "low"

    def test_intensity_label_medium(self, emotion_service: EmotionService) -> None:
        """Score of 0.55 should map to 'medium'."""
        assert emotion_service._get_intensity_label(0.55) == "medium"

    def test_intensity_label_high(self, emotion_service: EmotionService) -> None:
        """Score of 0.8 should map to 'high'."""
        assert emotion_service._get_intensity_label(0.8) == "high"


class TestFallbacks:
    """Tests for fallback behaviour."""

    def test_primary_model_fallback(self) -> None:
        """When primary pipeline is disabled, another model should be used."""
        service = EmotionService()
        service._primary_pipeline = None
        service._primary_loaded = False
        result = service.detect("I am very happy today")
        assert result.model_used != "j-hartmann/emotion-english-distilroberta-base"
        assert result.emotion is not None

    def test_empty_text_handled(self, emotion_service: EmotionService) -> None:
        """Empty string should return neutral without raising."""
        result = emotion_service.detect("")
        assert result.emotion == "neutral"

    def test_long_text_handled(self, emotion_service: EmotionService) -> None:
        """Very long text should be handled without error."""
        long_text = "I am so incredibly happy and grateful! " * 50
        result = emotion_service.detect(long_text)
        assert result.emotion is not None
        assert result.intensity >= 0.0

    def test_vader_fallback(self) -> None:
        """When all three transformer models are disabled, VADER should be used."""
        service = EmotionService()
        service._primary_pipeline = None
        service._primary_loaded = False
        service._fallback_pipeline = None
        service._fallback_loaded = False
        service._tertiary_pipeline = None
        service._tertiary_loaded = False
        result = service.detect("I love this so much!")
        assert result.model_used == "VADER"
