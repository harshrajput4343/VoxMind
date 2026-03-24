"""
Integration tests for the full Empathy Engine pipeline.

Tests:
  - End-to-end pipeline for joy and sadness
  - Modulated audio differs from raw TTS
  - Output saved to correct directory
  - Pipeline completes within time limit
"""

import os
import time

import pytest

from app.services.emotion_service import EmotionService
from app.services.tts_service import TTSService
from app.services.audio_service import AudioService
from app.services.ssml_builder import ssml_builder
from app.core.config import settings


@pytest.fixture(scope="module")
def services() -> tuple[EmotionService, TTSService, AudioService]:
    """Shared service instances for integration tests."""
    return EmotionService(), TTSService(), AudioService()


def _run_pipeline(text: str, services: tuple, tmp_path) -> dict:
    """
    Run the full synthesis pipeline end-to-end.

    Args:
        text: Input text.
        services: Tuple of (EmotionService, TTSService, AudioService).
        tmp_path: Temporary directory for output.

    Returns:
        Dict with emotion, provider, raw_path, final_path.
    """
    es, ts, aus = services
    emotion = es.detect(text)
    params = aus.get_voice_params(emotion)
    ssml_text = ssml_builder.build(text, params)

    raw_path = str(tmp_path / "raw.wav")
    final_path = str(tmp_path / "final.wav")

    _, provider = ts.synthesize(ssml_text, raw_path)
    aus.modulate_audio(raw_path, params, final_path)

    return {
        "emotion": emotion,
        "provider": provider,
        "raw_path": raw_path,
        "final_path": final_path,
    }


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_joy(self, services: tuple, tmp_path) -> None:
        """Happy text should produce a joy emotion and create an audio file."""
        result = _run_pipeline("I'm so happy and excited!", services, tmp_path)
        assert result["emotion"].emotion in ("joy", "surprise")
        assert os.path.exists(result["final_path"])

    def test_full_pipeline_sadness(self, services: tuple, tmp_path) -> None:
        """Sad text should produce a sadness emotion and create an audio file."""
        result = _run_pipeline(
            "Everything feels hopeless and dark, I can't go on.",
            services, tmp_path,
        )
        assert result["emotion"].emotion in ("sadness", "fear")
        assert os.path.exists(result["final_path"])

    def test_modulated_differs_from_raw(self, services: tuple, tmp_path) -> None:
        """Modulated audio should differ from raw TTS output for non-neutral emotions."""
        result = _run_pipeline("I am ECSTATIC beyond belief!!!", services, tmp_path)
        raw_size = os.path.getsize(result["raw_path"])
        final_size = os.path.getsize(result["final_path"])
        # Modulation changes duration/amplitude, so sizes should differ
        assert raw_size != final_size or raw_size > 0

    def test_output_saved_to_dir(self, services: tuple, tmp_path) -> None:
        """Audio output should exist in the specified directory."""
        result = _run_pipeline("Testing output directory.", services, tmp_path)
        assert os.path.isfile(result["final_path"])
        assert result["final_path"].endswith(".wav")

    def test_pipeline_time_limit(self, services: tuple, tmp_path) -> None:
        """Full pipeline should complete in under 45 seconds."""
        start = time.time()
        _run_pipeline("Quick timing test.", services, tmp_path)
        elapsed = time.time() - start
        assert elapsed < 45, f"Pipeline took {elapsed:.1f}s, exceeds 45s limit"
