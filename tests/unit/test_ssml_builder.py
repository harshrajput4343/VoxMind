"""
Unit tests for SSMLBuilder.

Tests:
  - Valid SSML structure (<speak>, <prosody> tags)
  - Emotion-specific markup (emphasis levels, pauses)
  - SSML stripping returns plain text
"""

import pytest

from app.services.ssml_builder import SSMLBuilder
from app.models.schemas import VoiceParameters


@pytest.fixture
def builder() -> SSMLBuilder:
    """SSMLBuilder instance for testing."""
    return SSMLBuilder()


def _make_params(emotion: str = "neutral", rate: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> VoiceParameters:
    """Create VoiceParameters for testing."""
    return VoiceParameters(
        rate=rate, pitch=pitch, volume=volume,
        emotion=emotion, intensity_label="medium",
    )


class TestSSMLGeneration:
    """Tests for SSML document building."""

    def test_ssml_contains_speak_tag(self, builder: SSMLBuilder) -> None:
        """Output should start with <speak>."""
        result = builder.build("Hello", _make_params())
        assert result.startswith("<speak>")
        assert result.endswith("</speak>")

    def test_ssml_contains_prosody(self, builder: SSMLBuilder) -> None:
        """Output should contain <prosody rate= attribute."""
        result = builder.build("Hello", _make_params(rate=1.2))
        assert '<prosody rate="' in result

    def test_ssml_joy_strong_emphasis(self, builder: SSMLBuilder) -> None:
        """Joy emotion should produce strong emphasis."""
        result = builder.build("I am happy!", _make_params(emotion="joy"))
        assert 'emphasis level="strong"' in result

    def test_ssml_sadness_pause(self, builder: SSMLBuilder) -> None:
        """Sadness emotion should include a <break> pause tag."""
        result = builder.build("I am sad.", _make_params(emotion="sadness"))
        assert "<break time=" in result

    def test_ssml_neutral_no_emphasis(self, builder: SSMLBuilder) -> None:
        """Neutral emotion should not include <emphasis> tag."""
        result = builder.build("Just a fact.", _make_params(emotion="neutral"))
        assert "<emphasis" not in result

    def test_strip_ssml(self, builder: SSMLBuilder) -> None:
        """Stripping SSML should return only plain text."""
        ssml = '<speak><prosody rate="100%">Hello</prosody></speak>'
        assert builder.strip_ssml(ssml) == "Hello"
