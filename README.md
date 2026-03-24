# 🎭 The Empathy Engine

> **Emotion-aware text-to-speech synthesis** — Detects how text *feels*, then speaks it with matching vocal emotion.

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## Overview

The Empathy Engine is a production-grade AI service built for the **Darvish AI Campus Challenge**. It takes plain text as input, detects the underlying emotion using state-of-the-art transformer models, then generates speech audio with vocal parameters (rate, pitch, volume) modulated to match the detected emotion and intensity.

**All 4 bonus features are implemented:**
- 🎭 **B1 — Granular Emotions**: 7 distinct emotions (not just positive/negative)
- 📈 **B2 — Intensity Scaling**: low/medium/high intensity → different voice parameters
- 🌐 **B3 — Web Interface**: Beautiful minimalist dark-theme UI with live audio player
- 🔤 **B4 — SSML Integration**: Full `<prosody>`, `<emphasis>`, and `<break>` tag generation

---

## Features

| Feature | Description |
|---------|-------------|
| 🎭 **7-Emotion Detection** | joy, sadness, anger, fear, disgust, surprise, neutral |
| 📊 **3-Level Emotion Fallback** | j-hartmann transformer → GoEmotions transformer → VADER |
| 🎙️ **3 TTS Providers** | ElevenLabs (optional) → gTTS (free) → pyttsx3 (offline) |
| 🎛️ **3 Vocal Parameters** | Rate + Pitch + Volume modulation per emotion |
| 📈 **Intensity Scaling** | Each emotion has low/medium/high parameter variants |
| 🔤 **SSML Integration** | `<prosody>`, `<emphasis>`, `<break>` tags |
| 🌐 **Minimalist Web UI** | Dark-theme, responsive, waveform visualizer, audio player |
| 🐳 **Docker Ready** | Dockerfile + docker-compose + Render.com config |
| ✅ **33+ Tests** | Unit, integration, and API endpoint tests |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP Request                         │
│                  POST /synthesize                       │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              EmotionService.detect()                     │
│  [j-hartmann] → [GoEmotions] → [VADER] (fallback chain) │
│  Returns: EmotionResult {emotion, intensity, scores}     │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│           AudioService.get_voice_params()                │
│    Looks up EMOTION_VOICE_MAPPING[(emotion, intensity)]  │
│    Returns: VoiceParameters {rate, pitch, volume, ssml}  │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              SSMLBuilder.build()  ← BONUS B4             │
│   Wraps text in <speak><prosody rate pitch volume>       │
│   Adds <emphasis> and <break> tags per emotion           │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              TTSService.synthesize()                     │
│   [ElevenLabs] → [gTTS] → [pyttsx3] (fallback chain)    │
│   Returns: raw .wav file path                            │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│            AudioService.modulate_audio()                 │
│   pydub: rate change → numpy pitch shift → volume gain   │
│   Returns: final modulated .wav file                     │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              SynthesisResponse (JSON)                    │
│   {audio_filename, emotion, voice_params, provider,      │
│    processing_time_ms}                                   │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Zero API keys needed** — works out of the box with free models.

```bash
# 1. Clone
git clone https://github.com/your-username/empathy-engine.git
cd empathy-engine

# 2. Virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (required for audio processing)
# Windows: choco install ffmpeg  OR  download from https://ffmpeg.org
# Mac: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# 5. Copy environment file (no edits needed for basic run)
copy .env.example .env

# 6. Run
uvicorn app.main:app --reload --port 8000
or
.\venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
# 7. Open browser
# http://localhost:8000
```

---

## API Documentation

### `GET /` — Web UI
Returns the interactive HTML interface.

### `POST /synthesize` — Main Pipeline
```json
// Request
{ "text": "I am so happy today!" }

// Response
{
  "audio_filename": "audio_a1b2c3d4.wav",
  "emotion": {
    "emotion": "joy",
    "intensity": 0.9432,
    "all_scores": {"joy": 0.9432, "surprise": 0.0312, ...},
    "model_used": "j-hartmann/emotion-english-distilroberta-base",
    "intensity_label": "high"
  },
  "voice_params": {
    "rate": 1.35, "pitch": 4.0, "volume": 1.2,
    "emotion": "joy", "intensity_label": "high",
    "ssml": "<speak><prosody rate=\"135%\" ...>...</prosody></speak>"
  },
  "tts_provider_used": "gTTS",
  "processing_time_ms": 2340.12
}
```

### `GET /audio/{filename}` — Audio File
Returns the generated `.wav` file.

### `GET /health` — Health Check
```json
{
  "status": "healthy",
  "elevenlabs_configured": false,
  "models_loaded": {
    "primary_transformer": true,
    "fallback_transformer": true,
    "vader": true
  },
  "tts_providers_available": ["gTTS", "pyttsx3"]
}
```

---

## Emotion → Voice Mapping Table

| Emotion  | Intensity | Rate  | Pitch  | Volume |
|----------|-----------|-------|--------|--------|
| joy      | low       | 1.10  | +1.5st | 1.00   |
| joy      | medium    | 1.20  | +2.5st | 1.10   |
| joy      | high      | 1.35  | +4.0st | 1.20   |
| sadness  | low       | 0.90  | -1.5st | 0.90   |
| sadness  | medium    | 0.80  | -2.5st | 0.85   |
| sadness  | high      | 0.70  | -4.0st | 0.75   |
| anger    | low       | 1.05  | +0.5st | 1.10   |
| anger    | medium    | 1.15  | -0.5st | 1.20   |
| anger    | high      | 1.30  | -1.0st | 1.40   |
| fear     | low       | 1.10  | +1.0st | 0.90   |
| fear     | medium    | 1.20  | +2.0st | 0.85   |
| fear     | high      | 1.35  | +3.0st | 0.80   |
| surprise | low       | 1.05  | +2.0st | 1.00   |
| surprise | medium    | 1.15  | +3.0st | 1.05   |
| surprise | high      | 1.25  | +3.5st | 1.10   |
| disgust  | low       | 0.95  | -1.0st | 1.00   |
| disgust  | medium    | 0.90  | -1.5st | 0.95   |
| disgust  | high      | 0.85  | -2.0st | 0.95   |
| neutral  | any       | 1.00  |  0.0st | 1.00   |

---

## Design Notes

### Why j-hartmann as Primary Model
`j-hartmann/emotion-english-distilroberta-base` is a fine-tuned DistilRoBERTa model trained on 6 emotion datasets. It classifies into exactly our 7 target emotions with high accuracy. It's free, publicly available on HuggingFace, and runs on CPU.

### Why 3-Level Fallback Chains
Both emotion detection and TTS use triple fallback chains to ensure the app **never fails**:
- **Emotion**: If the primary transformer can't load (e.g., no internet to download), GoEmotions is tried. If that fails too, VADER (which is 100% offline with no model download) handles the request.
- **TTS**: If ElevenLabs key expires or isn't set, gTTS (free, internet-only) takes over. If internet is down, pyttsx3 (fully offline) handles synthesis.

### Intensity Scaling Design
Each emotion has 3 intensity variants (low/medium/high) with progressively stronger parameter adjustments. This creates nuanced vocal expression — for example, mild joy slightly increases rate and pitch, while intense joy dramatically increases both.

### SSML Integration Approach
SSML markup is generated for every request with `<prosody>` (rate, pitch, volume), `<emphasis>` (emotion-appropriate levels), and `<break>` (pauses for sadness/fear). ElevenLabs receives the full SSML; gTTS and pyttsx3 receive the stripped plain text since they don't natively support SSML.

### Pitch Shift via Sample Rate Trick
Pitch shifting uses numpy interpolation: the audio samples are resampled at a ratio of `2^(semitones/12)`, effectively shifting the pitch without changing duration. This is combined with pydub's frame rate manipulation for speed changes.

---

## Running Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# API endpoint tests
pytest tests/api/ -v

# All tests with coverage
pytest --cov=app tests/ -v
```

---

## Docker

```bash
# Using docker-compose
docker-compose up --build

# Or standalone
docker build -t empathy-engine .
docker run -p 8000:8000 empathy-engine
```

---

## Deploy to Render.com

1. Push repo to GitHub
2. New Web Service → Connect GitHub repo
3. Select **Docker** runtime
4. Set env vars: `ELEVENLABS_API_KEY` (optional)
5. Deploy — `render.yaml` handles the rest

---

## Environment Variables

| Variable             | Required | Default       | Description              |
|----------------------|----------|---------------|--------------------------|
| `ELEVENLABS_API_KEY` | No       | `""`          | Optional premium TTS key |
| `ELEVENLABS_VOICE_ID`| No       | `21m00T...AM` | ElevenLabs voice ID      |
| `HF_TOKEN`           | No       | `""`          | Optional HuggingFace token |
| `APP_ENV`            | No       | `development` | Environment name         |
| `APP_PORT`           | No       | `8000`        | Server port              |
| `LOG_LEVEL`          | No       | `INFO`        | Logging level            |
| `OUTPUT_DIR`         | No       | `outputs`     | Audio output directory   |
| `MAX_TEXT_LENGTH`    | No       | `500`         | Max input characters     |
| `MODEL_CACHE_DIR`   | No       | `model_cache` | HuggingFace cache dir    |

---

## License

MIT License — Built for the Darvish AI Campus Challenge.
