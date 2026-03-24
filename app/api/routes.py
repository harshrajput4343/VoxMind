"""
API route handlers for the Empathy Engine.

Endpoints:
  GET  /              → Web UI (Jinja2 template)
  POST /synthesize    → Full emotion→voice→audio pipeline
  GET  /audio/{name}  → Serve generated audio files
  GET  /health        → System health check
"""

import os
import re
import shutil
import time
import uuid

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from app.services.emotion_service import EmotionService
from app.services.tts_service import TTSService
from app.services.audio_service import AudioService
from app.services.ssml_builder import ssml_builder
from app.models.schemas import SynthesisRequest, SynthesisResponse, HealthResponse
from app.core.config import settings
from app.core.logger import logger

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# ── Singleton service instances (loaded once at startup) ────────────
emotion_service = EmotionService()
tts_service = TTSService()
audio_service = AudioService()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Serve the main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/synthesize", response_model=SynthesisResponse)
async def synthesize(body: SynthesisRequest) -> SynthesisResponse:
    """
    Main pipeline endpoint.

    Steps:
    1. Validate text input (Pydantic handles this)
    2. Detect emotion + intensity
    3. Retrieve voice parameters from emotion→voice map
    4. Build SSML string (Bonus B4)
    5. Generate raw TTS audio
    6. Modulate audio with voice params (rate, pitch, volume)
    7. Save to outputs/ with UUID filename
    8. Return SynthesisResponse with all metadata

    Args:
        body: SynthesisRequest with text field.

    Returns:
        SynthesisResponse with audio filename, emotion, voice params, etc.
    """
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Synthesize request: '{body.text[:60]}...'")

    try:
        # ── 1. Emotion detection ─────────────────────────────
        emotion_result = emotion_service.detect(body.text)
        logger.info(
            f"[{request_id}] Emotion: {emotion_result.emotion} "
            f"({emotion_result.intensity_label}, {emotion_result.intensity:.2f}) "
            f"via {emotion_result.model_used}"
        )

        # ── 2. Voice parameters ──────────────────────────────
        voice_params = audio_service.get_voice_params(emotion_result)
        logger.info(
            f"[{request_id}] Voice params: rate={voice_params.rate}, "
            f"pitch={voice_params.pitch}, vol={voice_params.volume}"
        )

        # ── 3. Build SSML (Bonus B4) ─────────────────────────
        ssml_text = ssml_builder.build(body.text, voice_params)
        voice_params = voice_params.model_copy(update={"ssml": ssml_text})

        # ── 4. Generate raw TTS audio ────────────────────
        raw_filename = f"raw_{request_id}.wav"
        raw_path = str(settings.output_path / raw_filename)
        _, provider_used = tts_service.synthesize(
            ssml_text, raw_path,
            emotion=emotion_result.emotion,
            intensity=emotion_result.intensity_label,
        )

        # ── 5. Modulate audio ────────────────────────────────
        # Skip modulation for Gemini TTS — it handles emotion natively via prompt
        final_filename = f"audio_{request_id}.wav"
        final_path = str(settings.output_path / final_filename)
        if provider_used == "Gemini TTS":
            # Gemini already produced emotion-aware audio — just rename
            shutil.move(raw_path, final_path)
            logger.info(f"[{request_id}] Gemini TTS — skipping post-modulation (native emotion)")
        else:
            audio_service.modulate_audio(raw_path, voice_params, final_path)
            # Clean up raw TTS file
            if os.path.exists(raw_path):
                os.remove(raw_path)

        # ── 6. Return response ───────────────────────────────
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.success(
            f"[{request_id}] Pipeline complete in {elapsed_ms:.0f}ms via {provider_used}"
        )

        return SynthesisResponse(
            audio_filename=final_filename,
            emotion=emotion_result,
            voice_params=voice_params,
            tts_provider_used=provider_used,
            processing_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        logger.exception(f"[{request_id}] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@router.get("/audio/{filename}")
async def get_audio(filename: str) -> FileResponse:
    """
    Serve a generated audio file from the outputs directory.
    Validates filename to prevent path traversal attacks.

    Args:
        filename: Name of the audio file (must match UUID pattern).

    Returns:
        FileResponse with audio/wav content type.
    """
    if not re.match(r"^audio_[a-f0-9]{8}\.wav$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = settings.output_path / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint — reports model and provider status.

    Returns:
        HealthResponse with system status details.
    """
    return HealthResponse(
        status="healthy",
        elevenlabs_configured=settings.is_elevenlabs_configured(),
        models_loaded=emotion_service.models_loaded(),
        tts_providers_available=tts_service.available_providers(),
    )
