"""
Pytest fixtures shared across all test modules.

Provides:
  - test_client:        FastAPI TestClient (session-scoped)
  - sample_texts:       Representative text per emotion
  - mock_emotion_result: Factory for EmotionResult objects
  - tmp_output_dir:     Temporary directory for audio output
  - sample_wav:         Minimal valid WAV file for audio tests
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from pydub import AudioSegment

from app.main import app
from app.models.schemas import EmotionResult


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """FastAPI test client — session-scoped for performance."""
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_texts() -> dict[str, str]:
    """One representative text per emotion for testing."""
    return {
        "joy": "I just got accepted to my dream university! This is the happiest day of my life!",
        "sadness": "I feel completely hopeless. Nothing will ever get better.",
        "anger": "This is absolutely unacceptable! I am furious beyond words!",
        "fear": "I'm terrified. Something horrible is about to happen and I can't stop it.",
        "surprise": "I cannot believe this just happened! I'm completely shocked!",
        "disgust": "That is absolutely revolting. I've never been more disgusted in my life.",
        "neutral": "The quarterly report will be available on Thursday at 3 PM.",
    }


@pytest.fixture
def mock_emotion_result() -> callable:
    """Factory fixture that creates EmotionResult for any emotion."""
    def _factory(emotion: str = "joy", intensity: float = 0.85) -> EmotionResult:
        """
        Create a test EmotionResult.

        Args:
            emotion: Emotion label.
            intensity: Confidence score 0-1.

        Returns:
            EmotionResult instance.
        """
        label = "high" if intensity >= 0.7 else ("medium" if intensity >= 0.4 else "low")
        return EmotionResult(
            emotion=emotion,
            intensity=intensity,
            all_scores={emotion: intensity, "neutral": 1.0 - intensity},
            model_used="test_model",
            intensity_label=label,
        )
    return _factory


@pytest.fixture
def tmp_output_dir(tmp_path) -> str:
    """Temporary directory for audio output during tests."""
    output = tmp_path / "outputs"
    output.mkdir()
    return str(output)


@pytest.fixture
def sample_wav(tmp_path) -> str:
    """Create a minimal valid WAV file for audio tests (1s 440Hz sine wave)."""
    sample_rate = 22050
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    tone = (np.sin(440 * 2 * np.pi * t) * 16384).astype(np.int16)
    audio = AudioSegment(
        tone.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    wav_path = str(tmp_path / "test_tone.wav")
    audio.export(wav_path, format="wav")
    return wav_path
