"""
Pydantic v2 data models for the Empathy Engine API.

Models:
  - EmotionResult:      Output of emotion detection
  - VoiceParameters:    Rate/pitch/volume settings for TTS modulation
  - SynthesisRequest:   Incoming text payload
  - SynthesisResponse:  Full pipeline result returned to client
  - HealthResponse:     System health check payload
"""

from pydantic import BaseModel, Field, field_validator


class EmotionResult(BaseModel):
    """Result of emotion detection on input text."""

    emotion: str
    """Primary label: joy | sadness | anger | fear | disgust | surprise | neutral."""

    intensity: float
    """Confidence score from 0.0 to 1.0."""

    all_scores: dict[str, float]
    """Full label → score map for all emotions."""

    model_used: str
    """Which model/library produced this result."""

    intensity_label: str
    """Human-readable intensity: 'low' | 'medium' | 'high'."""


class VoiceParameters(BaseModel):
    """Voice modulation parameters derived from emotion analysis."""

    rate: float
    """Speech rate multiplier 0.5 → 2.0 (1.0 = normal)."""

    pitch: float
    """Semitone shift -5 → +5 (0 = no change)."""

    volume: float
    """Amplitude multiplier 0.5 → 1.5 (1.0 = normal)."""

    emotion: str
    """The emotion these parameters are tuned for."""

    intensity_label: str
    """The intensity level used to look up these parameters."""

    ssml: str = ""
    """SSML string generated for this emotion+intensity combo (Bonus B4)."""


class SynthesisRequest(BaseModel):
    """Incoming request payload for the /synthesize endpoint."""

    text: str = Field(..., min_length=1, max_length=500)

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        """Ensure text is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("text must not be empty or whitespace")
        return v.strip()


class SynthesisResponse(BaseModel):
    """Complete pipeline response returned to the client."""

    audio_filename: str
    """Name of the generated audio file (UUID-based)."""

    emotion: EmotionResult
    """Detected emotion and scores."""

    voice_params: VoiceParameters
    """Voice modulation parameters that were applied."""

    tts_provider_used: str
    """Which TTS provider generated the raw audio."""

    processing_time_ms: float
    """Total pipeline processing time in milliseconds."""


class HealthResponse(BaseModel):
    """System health check response."""

    status: str
    """Overall system status."""

    elevenlabs_configured: bool
    """Whether an ElevenLabs API key is present."""

    models_loaded: dict[str, bool]
    """Which emotion detection backends are available."""

    tts_providers_available: list[str]
    """List of TTS provider names that are currently usable."""
