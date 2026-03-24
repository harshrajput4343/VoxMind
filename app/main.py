"""
FastAPI application entry point for The Empathy Engine.

Configures CORS, lifespan events, exception handlers, and includes the API router.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup: pre-warm emotion models, log system info.
    Shutdown: log graceful exit.
    """
    logger.info("═══ Empathy Engine starting ═══")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Output dir  : {settings.output_path}")
    logger.info(
        f"ElevenLabs  : {'configured' if settings.is_elevenlabs_configured() else 'not configured (using gTTS/pyttsx3)'}"
    )

    # Pre-warm by importing (models loaded in EmotionService.__init__)
    from app.api.routes import emotion_service, tts_service

    ml = emotion_service.models_loaded()
    logger.info(f"Models loaded: {ml}")
    logger.info(f"TTS providers: {tts_service.available_providers()}")
    logger.success(
        f"Empathy Engine ready! Listening on http://{settings.app_host}:{settings.app_port}"
    )
    yield
    logger.info("Empathy Engine shutting down")


app = FastAPI(
    title="The Empathy Engine",
    description="Emotion-aware text-to-speech synthesis. Detects emotion, modulates voice.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


# ── Exception Handlers ──────────────────────────────────────────────

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic validation errors with a clean JSON response."""
    logger.warning(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors() if hasattr(exc, "errors") else str(exc)},
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle internal server errors with a clean JSON response."""
    logger.error(f"Internal server error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Check server logs."},
    )
