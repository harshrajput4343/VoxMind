"""
API endpoint tests for the Empathy Engine.

Tests:
  - GET / returns HTML
  - POST /synthesize returns valid response
  - GET /audio/{filename} serves WAV files
  - GET /health returns system status
  - Validation errors for bad input
"""

import pytest
from fastapi.testclient import TestClient


class TestHomeEndpoint:
    """Tests for GET /."""

    def test_home_200(self, test_client: TestClient) -> None:
        """Home page should return 200 with HTML content."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestSynthesizeEndpoint:
    """Tests for POST /synthesize."""

    def test_synthesize_200(self, test_client: TestClient) -> None:
        """Valid text should return 200."""
        response = test_client.post(
            "/synthesize",
            json={"text": "I am feeling great today!"},
        )
        assert response.status_code == 200

    def test_response_has_emotion(self, test_client: TestClient) -> None:
        """Response JSON should contain the emotion key."""
        response = test_client.post(
            "/synthesize",
            json={"text": "This is wonderful news!"},
        )
        data = response.json()
        assert "emotion" in data
        assert "emotion" in data["emotion"]

    def test_response_has_filename(self, test_client: TestClient) -> None:
        """Response JSON should contain audio_filename key."""
        response = test_client.post(
            "/synthesize",
            json={"text": "Let me hear this audio."},
        )
        data = response.json()
        assert "audio_filename" in data
        assert data["audio_filename"].endswith(".wav")

    def test_empty_text_422(self, test_client: TestClient) -> None:
        """Empty text should return 422 validation error."""
        response = test_client.post("/synthesize", json={"text": ""})
        assert response.status_code == 422

    def test_too_long_text_422(self, test_client: TestClient) -> None:
        """Text exceeding 500 chars should return 422."""
        response = test_client.post(
            "/synthesize",
            json={"text": "x" * 501},
        )
        assert response.status_code == 422

    def test_response_schema(self, test_client: TestClient) -> None:
        """Response should match the SynthesisResponse schema exactly."""
        response = test_client.post(
            "/synthesize",
            json={"text": "Schema validation test."},
        )
        data = response.json()
        required_keys = {"audio_filename", "emotion", "voice_params", "tts_provider_used", "processing_time_ms"}
        assert required_keys.issubset(data.keys())
        assert "rate" in data["voice_params"]
        assert "pitch" in data["voice_params"]
        assert "volume" in data["voice_params"]


class TestAudioEndpoint:
    """Tests for GET /audio/{filename}."""

    def test_audio_served(self, test_client: TestClient) -> None:
        """A synthesized audio file should be servable."""
        # First create an audio file
        synth = test_client.post(
            "/synthesize",
            json={"text": "Audio serving test."},
        )
        filename = synth.json()["audio_filename"]

        response = test_client.get(f"/audio/{filename}")
        assert response.status_code == 200
        assert "audio/wav" in response.headers["content-type"]


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_200(self, test_client: TestClient) -> None:
        """Health endpoint should return 200 with status healthy."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
