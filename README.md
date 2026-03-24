<p align="center">
  <h1 align="center">🎭 The Empathy Engine</h1>
  <p align="center">
    <strong>Emotion-Aware Text-to-Speech Synthesis — Detects how text <em>feels</em>, then speaks it with matching vocal emotion.</strong>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#design-decisions">Design Decisions</a> •
    <a href="#deployment">Deployment</a>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11" />
    <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white" alt="Pydantic v2" />
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker" />
    <img src="https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white" alt="License: MIT" />
    <img src="https://img.shields.io/badge/Tests-33+-4CAF50?logo=pytest&logoColor=white" alt="Tests" />
  </p>
</p>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)
  - [Emotion Detection Strategy](#1-emotion-detection-strategy)
  - [Emotion-to-Voice Mapping Logic](#2-emotion-to-voice-mapping-logic)
  - [Intensity Scaling Design](#3-intensity-scaling-design)
  - [TTS Provider Strategy](#4-tts-provider-strategy)
  - [SSML Integration](#5-ssml-integration)
  - [Audio Modulation Pipeline](#6-audio-modulation-pipeline)
- [Emotion → Voice Parameter Reference](#emotion--voice-parameter-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration Reference](#configuration-reference)
- [Performance & Optimizations](#performance--optimizations)
- [License](#license)

---

## Overview

The Empathy Engine is a production-grade AI microservice built for the **Darwix AI Campus Challenge**. It implements a complete emotion-aware text-to-speech pipeline: given plain text, the system detects the underlying emotion using state-of-the-art NLP transformer models, maps the detected emotion and its intensity to specific vocal parameters (speech rate, pitch, volume), generates speech audio via a multi-provider TTS stack, and then applies post-processing modulation to produce expressive, emotionally congruent audio output.

The system is designed around **resilience** — every stage of the pipeline employs a multi-level fallback chain, so the application never fails to produce output regardless of network conditions, API availability, or model loading issues.

### Bonus Features Implemented

| Bonus | Feature | Implementation |
|-------|---------|----------------|
| **B1** | 🎭 Granular Emotions | 7 distinct emotions: joy, sadness, anger, fear, disgust, surprise, neutral |
| **B2** | 📈 Intensity Scaling | 3 intensity levels (low / medium / high) with distinct parameter variants per emotion |
| **B3** | 🌐 Web Interface | Responsive dark-theme UI with real-time waveform visualizer and audio playback |
| **B4** | 🔤 SSML Integration | Full `<prosody>`, `<emphasis>`, and `<break>` tag generation per emotion |

---

## Key Features

| Category | Feature | Details |
|----------|---------|---------|
| **Emotion Detection** | 7-Emotion Classification | joy, sadness, anger, fear, disgust, surprise, neutral |
| **Emotion Detection** | 4-Level Fallback Chain | j-hartmann → GoEmotions → bhadresh-savani → VADER |
| **TTS Synthesis** | 4 TTS Providers | Gemini 2.5 Flash (primary) → ElevenLabs → gTTS → pyttsx3 |
| **Voice Modulation** | 3 Vocal Parameters | Rate + Pitch + Volume — all three modulated per emotion × intensity |
| **Voice Modulation** | Intensity Scaling | Each of the 7 emotions has low/medium/high parameter variants (21 mappings) |
| **SSML** | Full SSML Markup | `<prosody>` (rate, pitch, volume), `<emphasis>`, `<break>` tags |
| **Web UI** | Minimalist Dark Theme | Responsive design, waveform visualizer, audio player, emotion badges |
| **Infrastructure** | Docker Ready | Multi-stage Dockerfile + docker-compose + Render.com config |
| **Quality** | 33+ Tests | Unit, integration, and API endpoint tests with mocking |

---

## Project Structure

```
empathy-engine/
├── app/                          # Application source code
│   ├── __init__.py
│   ├── main.py                   # FastAPI entry point, lifespan events, CORS, exception handlers
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # API endpoints: /, /synthesize, /audio/{file}, /health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Pydantic Settings — env vars, paths, feature flags
│   │   └── logger.py             # Loguru config — console (coloured) + rotating file handler
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Pydantic v2 models: EmotionResult, VoiceParameters, etc.
│   ├── services/
│   │   ├── __init__.py
│   │   ├── emotion_service.py    # 4-level emotion detection with graceful degradation
│   │   ├── tts_service.py        # 4-provider TTS with abstract base class + fallback chain
│   │   ├── audio_service.py      # Voice param lookup + pydub/numpy audio modulation
│   │   └── ssml_builder.py       # SSML markup generation (Bonus B4)
│   └── templates/
│       └── index.html            # Single-page dark-theme web UI (34 KB, fully self-contained)
├── tests/                        # Test suite
│   ├── conftest.py               # Shared fixtures: mock services, sample data
│   ├── unit/                     # Unit tests for each service
│   │   ├── test_emotion_service.py
│   │   ├── test_tts_service.py
│   │   ├── test_audio_service.py
│   │   └── test_ssml_builder.py
│   ├── integration/              # Full pipeline integration tests
│   │   └── test_empathy_pipeline.py
│   └── api/                      # HTTP endpoint tests via TestClient
│       └── test_empathy_api.py
├── outputs/                      # Generated audio files (gitignored)
├── logs/                         # Application logs (rotating, 10 MB cap)
├── model_cache/                  # HuggingFace model cache directory
├── Dockerfile                    # Production Docker image (python:3.11-slim)
├── docker-compose.yml            # Local container orchestration
├── render.yaml                   # Render.com deployment config
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # Environment variable template
└── .gitignore
```

---

## Architecture

The Empathy Engine follows a **linear pipeline architecture** with defensive fallback patterns at every stage. A single `POST /synthesize` request flows through five distinct processing stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                         CLIENT REQUEST                                 │
│                    POST /synthesize { "text": "..." }                  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 1 — Emotion Detection                               │
│                                                                        │
│  EmotionService.detect(text)                                           │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────┐  │
│  │  j-hartmann  │──▶│  GoEmotions  │──▶│  bhadresh-   │──▶│ VADER  │  │
│  │  DistilRoBERTa│  │  RoBERTa     │   │  savani      │   │(offline)│ │
│  │  (7 emotions) │  │  (28→7 map)  │   │  (6→7 map)   │   │        │  │
│  └──────────────┘   └──────────────┘   └──────────────┘   └────────┘  │
│                                                                        │
│  Output: EmotionResult { emotion, intensity, intensity_label, scores } │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 2 — Voice Parameter Lookup                          │
│                                                                        │
│  AudioService.get_voice_params(emotion_result)                         │
│  Looks up EMOTION_VOICE_MAPPING[(emotion, intensity_label)]            │
│  21 entries: 7 emotions × 3 intensity levels                           │
│                                                                        │
│  Output: VoiceParameters { rate, pitch, volume, emotion, intensity }   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 3 — SSML Generation (Bonus B4)                      │
│                                                                        │
│  SSMLBuilder.build(text, voice_params)                                 │
│  Generates: <speak>                                                    │
│               <break time="Nms"/>           ← emotion-based pause      │
│               <prosody rate="N%" pitch="Nst" volume="NdB">             │
│                 <emphasis level="strong|moderate|reduced">             │
│                   {text}                                                │
│                 </emphasis>                                             │
│               </prosody>                                               │
│             </speak>                                                   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 4 — Text-to-Speech Synthesis                        │
│                                                                        │
│  TTSService.synthesize(ssml_text, output_path, emotion, intensity)     │
│  ┌──────────────┐  ┌────────────┐  ┌────────┐  ┌──────────┐           │
│  │ Gemini 2.5   │─▶│ ElevenLabs │─▶│  gTTS  │─▶│ pyttsx3  │           │
│  │ Flash TTS    │  │ (optional) │  │ (free) │  │(offline) │           │
│  │ (emotion via │  │ (SSML)     │  │        │  │          │           │
│  │  prompt)     │  │            │  │        │  │          │           │
│  └──────────────┘  └────────────┘  └────────┘  └──────────┘           │
│                                                                        │
│  Output: raw .wav file + provider_name                                 │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              STAGE 5 — Audio Post-Processing                           │
│                                                                        │
│  AudioService.modulate_audio(raw_path, params, output_path)            │
│  ┌───────────────┐  ┌────────────────┐  ┌──────────────┐               │
│  │ pydub: Rate   │─▶│ numpy: Pitch   │─▶│ pydub: Volume│               │
│  │ (frame_rate   │  │ (interpolation │  │ (dB gain)    │               │
│  │  manipulation)│  │  resampling)   │  │              │               │
│  └───────────────┘  └────────────────┘  └──────────────┘               │
│                                                                        │
│  * Skipped for Gemini TTS (handles emotion natively via prompt)        │
│                                                                        │
│  Output: final modulated .wav (16-bit PCM)                             │
└───────────────────────────┬────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│              RESPONSE                                                  │
│                                                                        │
│  SynthesisResponse {                                                   │
│    audio_filename,        ← UUID-based .wav filename                   │
│    emotion,               ← full EmotionResult with scores             │
│    voice_params,          ← applied rate/pitch/volume + SSML           │
│    tts_provider_used,     ← which provider succeeded                   │
│    processing_time_ms     ← end-to-end latency                        │
│  }                                                                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

**Zero API keys needed** — the application works out of the box using free models and TTS providers.

### Prerequisites

| Requirement | Version | Purpose |
|-------------|---------|---------|
| **Python** | 3.11+ | Runtime |
| **pip** | Latest | Package manager |
| **ffmpeg** | Any | Audio format conversion (required by pydub) |
| **Git** | Any | Cloning the repository |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/empathy-engine.git
cd empathy-engine

# 2. Create and activate a virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install ffmpeg (required for audio processing)
# ┌──────────────────────────────────────────────────────────────────┐
# │ Windows:  choco install ffmpeg                                   │
# │           OR download from https://ffmpeg.org/download.html      │
# │ macOS:    brew install ffmpeg                                    │
# │ Linux:    sudo apt install ffmpeg                                │
# └──────────────────────────────────────────────────────────────────┘

# 5. Set up environment variables
#    Copy the template (no edits needed for basic run):
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
```

### Running the Application

```bash
# Start the development server
uvicorn app.main:app --reload --port 8000

# Alternative (explicit path on Windows):
.\venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open your browser and navigate to **http://localhost:8000** to access the web interface.

> **Note:** On first launch, the application will download transformer models from HuggingFace (~1 GB). Subsequent launches use the cached models from `model_cache/`.

---

## API Reference

### `GET /` — Web Interface

Returns the interactive single-page HTML application with a dark-theme UI, text input, emotion visualization, and audio playback controls.

### `POST /synthesize` — Main Synthesis Pipeline

The core endpoint that runs the full emotion → voice → audio pipeline.

**Request:**
```json
{
  "text": "I am so happy today!"
}
```

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `text` | `string` | 1–500 chars, non-whitespace | The text to synthesize |

**Response:**
```json
{
  "audio_filename": "audio_a1b2c3d4.wav",
  "emotion": {
    "emotion": "joy",
    "intensity": 0.9432,
    "all_scores": {
      "joy": 0.9432,
      "surprise": 0.0312,
      "neutral": 0.0156,
      "fear": 0.0042,
      "sadness": 0.0031,
      "disgust": 0.0018,
      "anger": 0.0009
    },
    "model_used": "j-hartmann/emotion-english-distilroberta-base",
    "intensity_label": "high"
  },
  "voice_params": {
    "rate": 1.35,
    "pitch": 4.0,
    "volume": 1.2,
    "emotion": "joy",
    "intensity_label": "high",
    "ssml": "<speak><prosody rate=\"135%\" pitch=\"+4.0st\" volume=\"+1.6dB\"><emphasis level=\"strong\">I am so happy today!</emphasis></prosody></speak>"
  },
  "tts_provider_used": "Gemini TTS",
  "processing_time_ms": 2340.12
}
```

### `GET /audio/{filename}` — Audio File Retrieval

Serves a generated `.wav` file from the `outputs/` directory. Filenames are validated against the pattern `audio_[a-f0-9]{8}.wav` to prevent path traversal attacks.

### `GET /health` — Health Check

Returns the system status, loaded models, and available TTS providers.

```json
{
  "status": "healthy",
  "elevenlabs_configured": false,
  "models_loaded": {
    "primary_transformer": true,
    "fallback_transformer": true,
    "tertiary_transformer": true,
    "vader": true
  },
  "tts_providers_available": ["Gemini TTS", "gTTS", "pyttsx3"]
}
```

---

## Design Decisions

### 1. Emotion Detection Strategy

**Choice:** A 4-level fallback chain of emotion classifiers with graceful degradation.

| Level | Model | Architecture | Emotions | Trade-off |
|-------|-------|-------------|----------|-----------|
| 1 (Primary) | `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa, fine-tuned on 6 datasets | 7 native | Highest accuracy; requires transformers + ~300 MB download |
| 2 (Fallback) | `SamLowe/roberta-base-go_emotions` | RoBERTa, trained on GoEmotions | 28 → mapped to 7 | Broad coverage; 28 granular labels aggregated into 7 canonical buckets |
| 3 (Tertiary) | `bhadresh-savani/distilbert-base-uncased-emotion` | DistilBERT | 6 → mapped to 7 | Lighter model; lacks "disgust" and "neutral" labels |
| 4 (Last Resort) | VADER (SentimentIntensityAnalyzer) | Lexicon + rule-based | pos/neg/neu → 3 | 100% offline, zero latency; only detects joy, sadness, or neutral |

**Rationale:** The primary model (`j-hartmann`) was chosen because it classifies into *exactly* our 7 target emotions with high accuracy and is freely available on HuggingFace. The fallback chain ensures the application **never fails** — even if all transformers fail to load (e.g., in a memory-constrained environment), VADER provides a deterministic, offline baseline.

**Label Normalization:** All models' labels are mapped to a canonical set of 7 emotions via `EMOTION_MAP`. For GoEmotions, 28 granular labels (e.g., `admiration`, `amusement`, `excitement`) are collapsed into the 7 canonical buckets. The `detect()` method *never raises* — it always returns a valid `EmotionResult`, falling through the chain until one succeeds or returning a neutral default.

---

### 2. Emotion-to-Voice Mapping Logic

**Choice:** A static lookup table (`EMOTION_VOICE_MAPPING`) keyed by `(emotion, intensity_label)` tuples, producing exact `VoiceParameters` values.

**Design Rationale:** Each mapping was designed based on psychoacoustic and phonetic research about how humans naturally express emotions through voice:

| Emotion | Rate Logic | Pitch Logic | Volume Logic |
|---------|-----------|-------------|--------------|
| **Joy** | ↑ Faster speech conveys excitement and energy | ↑ Higher pitch signals positivity and openness | ↑ Louder delivery projects enthusiasm |
| **Sadness** | ↓ Slower pace signals lethargy and weight | ↓ Lower pitch conveys heaviness and withdrawal | ↓ Quieter tone suggests vulnerability |
| **Anger** | ↑ Faster pace shows urgency and agitation | ↓ Slightly lower pitch conveys authority and force | ↑↑ Louder output (up to 1.4×) projects dominance |
| **Fear** | ↑ Faster pace mimics breathless, rushed speech | ↑ Higher pitch signals alarm and anxiety | ↓ Quieter volume reflects shrinking/withdrawal |
| **Surprise** | ↑ Slight acceleration shows spontaneous reaction | ↑↑ Highest pitch range signals widened vocal range | → Slight increase to project astonishment |
| **Disgust** | ↓ Slower pacing shows deliberation/contempt | ↓ Lower pitch conveys disdain | → Near-neutral volume, slightly dampened |
| **Neutral** | → 1.0× baseline with no modification | → 0 semitones, no shift | → 1.0× baseline |

**Why static, not dynamic?** A lookup table provides deterministic, reproducible behaviour and is trivially testable. Dynamic computation (e.g., linear interpolation on the intensity score) would introduce unpredictability and make the system harder to validate against assessment rubrics.

---

### 3. Intensity Scaling Design

**Choice:** Three discrete intensity levels (low / medium / high) with progressively stronger parameter adjustments per emotion.

**Thresholds:**
```
score < 0.4  →  "low"       (subtle vocal adjustment)
score < 0.7  →  "medium"    (noticeable vocal adjustment)
score ≥ 0.7  →  "high"      (dramatic vocal adjustment)
```

**Example — Joy Scaling:**
| Intensity | Rate | Pitch | Volume | Perceived Effect |
|-----------|------|-------|--------|-----------------|
| Low | 1.10× | +1.5 semitones | 1.00× | Mildly cheerful — slight uptick in pacing |
| Medium | 1.20× | +2.5 semitones | 1.10× | Clearly happy — brighter tone, louder |
| High | 1.35× | +4.0 semitones | 1.20× | Ecstatic — fast, bright, and projected |

This creates **nuanced vocal expression** — the same emotion at different confidence levels produces distinctly different audio output, making the system feel more natural and responsive.

---

### 4. TTS Provider Strategy

**Choice:** Environment-aware provider selection with a 4-level fallback chain.

```
Production mode (APP_ENV=production):
  Gemini 2.5 Flash → ElevenLabs → gTTS → pyttsx3

Development mode (APP_ENV=development):
  gTTS → pyttsx3
```

| Provider | Cost | API Key | Internet | Quality | Special Capability |
|----------|------|---------|----------|---------|-------------------|
| **Gemini 2.5 Flash** | Free (250 req/day) | Yes (free) | Yes | ★★★★★ | Emotion-aware via prompt injection |
| **ElevenLabs** | Freemium (10K chars/mo) | Yes | Yes | ★★★★★ | SSML prosody support |
| **gTTS** | Free (unlimited) | No | Yes | ★★★☆☆ | Slow mode for sadness |
| **pyttsx3** | Free (unlimited) | No | No | ★★☆☆☆ | 100% offline via SAPI5/eSpeak |

**Why Gemini 2.5 Flash as Primary:** Unlike traditional TTS engines that produce flat robotic speech, Gemini 2.5 Flash accepts an *emotion-style prompt* alongside the text. For each `(emotion, intensity)` pair, a curated natural-language prompt (e.g., *"Speak with genuine excitement and warmth, fast pace, bright and energetic tone"*) is injected, producing natively expressive audio that sounds significantly more human. This eliminates the need for post-processing modulation when Gemini is used.

**Why dev/prod split:** During development, API quotas (Gemini: 250/day, ElevenLabs: 10K chars/month) are precious. Development mode skips premium providers entirely, using only unlimited free providers (gTTS/pyttsx3) for rapid iteration.

---

### 5. SSML Integration

**Choice:** Generate a full SSML document for every request, then conditionally pass or strip it depending on the TTS provider's capabilities.

**SSML Structure Generated:**
```xml
<speak>
  <break time="400ms"/>                              <!-- Emotion-based pause (e.g., sadness) -->
  <prosody rate="70%" pitch="-4.0st" volume="-2.5dB"> <!-- Vocal parameters -->
    <emphasis level="reduced">                        <!-- Emotion-appropriate emphasis -->
      I feel so alone right now.
    </emphasis>
  </prosody>
</speak>
```

**Per-Provider Handling:**
| Provider | SSML Support | Behaviour |
|----------|-------------|-----------|
| ElevenLabs | ✅ Full | Receives complete SSML — prosody tags are interpreted natively |
| Gemini TTS | ❌ Not used | SSML is stripped; emotion is conveyed via prompt injection instead |
| gTTS | ❌ None | SSML tags are stripped via `SSMLBuilder.strip_ssml()` → plain text |
| pyttsx3 | ❌ None | SSML tags are stripped → plain text |

**Emphasis and Pause Mappings:**
| Emotion | `<emphasis>` Level | `<break>` Duration |
|---------|-------------------|-------------------|
| Joy | `strong` | 0 ms |
| Anger | `strong` | 100 ms |
| Surprise | `strong` | 200 ms |
| Fear | `moderate` | 300 ms |
| Disgust | `moderate` | 200 ms |
| Sadness | `reduced` | 400 ms |
| Neutral | `none` (omitted) | 0 ms |

---

### 6. Audio Modulation Pipeline

**Choice:** A three-step post-processing pipeline using `pydub` for rate/volume and `numpy` for pitch shifting.

**Step 1 — Rate Change (pydub frame-rate trick):**
The raw audio's frame rate is multiplied by the rate factor, then resampled back to the original frame rate. This produces a time-stretched or time-compressed version of the audio.
```python
new_rate = int(original_frame_rate * params.rate)
audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
audio = audio.set_frame_rate(original_frame_rate)
```

**Step 2 — Pitch Shift (numpy interpolation):**
Audio samples are resampled at a ratio of `2^(semitones/12)`, shifting the fundamental frequency without changing duration. This uses linear interpolation rather than FFT-based methods for simplicity and performance.
```python
ratio = 2 ** (params.pitch / 12.0)
new_length = int(len(samples) / ratio)
indices = np.linspace(0, len(samples) - 1, new_length)
pitched = np.interp(indices, np.arange(len(samples)), samples)
```

**Step 3 — Volume Change (dB gain):**
The volume multiplier is converted to decibels using the standard formula `dB = 20 × log₁₀(multiplier)`, then applied as gain.

**Important:** When Gemini TTS is the active provider, the entire modulation stage is **skipped** — Gemini produces emotion-aware audio natively through prompt injection, so post-processing would degrade the output quality.

---

## Emotion → Voice Parameter Reference

The complete mapping table used in `audio_service.py`. Pitch values are in semitones (st), rate/volume are multipliers.

| Emotion | Intensity | Rate | Pitch | Volume | Perceptual Intent |
|---------|-----------|-------|--------|--------|-------------------|
| **joy** | low | 1.10× | +1.5 st | 1.00× | Mildly cheerful, slight uptick |
| **joy** | medium | 1.20× | +2.5 st | 1.10× | Clearly happy, brighter tone |
| **joy** | high | 1.35× | +4.0 st | 1.20× | Ecstatic, fast and bright |
| **sadness** | low | 0.90× | −1.5 st | 0.90× | Slightly subdued, measured |
| **sadness** | medium | 0.80× | −2.5 st | 0.85× | Noticeably somber, slower |
| **sadness** | high | 0.70× | −4.0 st | 0.75× | Heavy grief, slow and quiet |
| **anger** | low | 1.05× | +0.5 st | 1.10× | Firm and serious |
| **anger** | medium | 1.15× | −0.5 st | 1.20× | Stern and assertive |
| **anger** | high | 1.30× | −1.0 st | 1.40× | Intense, loud, sharp delivery |
| **fear** | low | 1.10× | +1.0 st | 0.90× | Slight unease, cautious |
| **fear** | medium | 1.20× | +2.0 st | 0.85× | Worried, tense pacing |
| **fear** | high | 1.35× | +3.0 st | 0.80× | Panicked, breathless |
| **surprise** | low | 1.05× | +2.0 st | 1.00× | Mildly surprised |
| **surprise** | medium | 1.15× | +3.0 st | 1.05× | Genuinely astonished |
| **surprise** | high | 1.25× | +3.5 st | 1.10× | Wide-eyed shock |
| **disgust** | low | 0.95× | −1.0 st | 1.00× | Mild displeasure |
| **disgust** | medium | 0.90× | −1.5 st | 0.95× | Clear disapproval |
| **disgust** | high | 0.85× | −2.0 st | 0.95× | Strong distaste |
| **neutral** | any | 1.00× | 0.0 st | 1.00× | Flat baseline — no modulation |

---

## Testing

The test suite covers unit, integration, and API layers with 33+ test cases.

```bash
# Run all tests
pytest tests/ -v

# Run by layer
pytest tests/unit/ -v           # Unit tests (services in isolation)
pytest tests/integration/ -v    # Full pipeline integration tests
pytest tests/api/ -v            # HTTP endpoint tests via TestClient

# Run with coverage report
pytest --cov=app tests/ -v

# Run a specific test file
pytest tests/unit/test_emotion_service.py -v
```

### Test Structure

| Layer | File | Tests | What It Validates |
|-------|------|-------|-------------------|
| **Unit** | `test_emotion_service.py` | Emotion detection, label mapping, fallback logic, VADER |
| **Unit** | `test_tts_service.py` | Provider availability, synthesis, fallback chain |
| **Unit** | `test_audio_service.py` | Voice param lookup, audio modulation, edge cases |
| **Unit** | `test_ssml_builder.py` | SSML generation, tag correctness, stripping |
| **Integration** | `test_empathy_pipeline.py` | End-to-end pipeline from text to audio |
| **API** | `test_empathy_api.py` | HTTP status codes, response schemas, error handling |

---

## Deployment

### Option 1: Docker (Recommended)

```bash
# Using docker-compose (includes volume mounts, health checks)
docker-compose up --build

# Or standalone Docker
docker build -t empathy-engine .
docker run -p 8000:8000 \
  -e APP_ENV=production \
  -e GEMINI_API_KEY=your_key_here \
  empathy-engine
```

The Dockerfile uses a **model pre-baking strategy**: transformer models and VADER lexicons are downloaded at build time and baked into the image layers, ensuring **zero cold-start latency** when the container starts.

### Option 2: Render.com

1. Push the repository to GitHub
2. Create a **New Web Service** → Connect the GitHub repo
3. Select **Docker** as the runtime
4. Set environment variables: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` (optional)
5. Deploy — the `render.yaml` blueprint handles the rest (disk, env vars, health checks)

### Option 3: Local Development

```bash
# Ensure APP_ENV=development in .env (default)
uvicorn app.main:app --reload --port 8000
```

---

## Configuration Reference

All configuration is managed via environment variables (loaded from `.env`). **No variable is required** — the application runs with sensible defaults out of the box.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | `""` | Google AI Studio API key for Gemini 2.5 Flash TTS. Free at [aistudio.google.com](https://aistudio.google.com) |
| `ELEVENLABS_API_KEY` | No | `""` | ElevenLabs API key for premium voice synthesis |
| `ELEVENLABS_VOICE_ID` | No | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice ID (default: "Rachel") |
| `HF_TOKEN` | No | `""` | HuggingFace token for gated model access (not required for default models) |
| `APP_ENV` | No | `development` | Environment mode: `development` (free TTS only) or `production` (full TTS stack) |
| `APP_HOST` | No | `0.0.0.0` | Server bind host |
| `APP_PORT` | No | `8000` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `OUTPUT_DIR` | No | `outputs` | Directory for generated audio files |
| `MAX_TEXT_LENGTH` | No | `500` | Maximum allowed characters in synthesis request |
| `MODEL_CACHE_DIR` | No | `model_cache` | HuggingFace transformer model cache directory |

---

## Performance & Optimizations

| Optimization | How It Works |
|-------------|-------------|
| **Docker Model Pre-baking** | Transformer models are downloaded at `docker build` time and baked into image layers. Containers start with models already on disk — zero download latency at runtime. |
| **FastAPI Lifespan Pre-warming** | Models are loaded into memory during the `lifespan` startup event, not on the first request. The first user request hits a warm model. |
| **Singleton Services** | `EmotionService`, `TTSService`, and `AudioService` are instantiated once as module-level singletons in `routes.py` and reused across all requests. |
| **Environment-Aware TTS** | Development mode skips premium API providers entirely, conserving Gemini/ElevenLabs quotas for production demos. |
| **Path Traversal Prevention** | Audio filenames are validated with regex `^audio_[a-f0-9]{8}\.wav$` before serving, preventing directory traversal attacks. |
| **Rotating Log Files** | Loguru rotates `logs/app.log` at 10 MB with 7-day retention, preventing disk exhaustion in long-running deployments. |
| **Gemini Skip-Modulation** | When Gemini TTS produces the audio, it handles emotion natively via prompt injection — the post-processing modulation step is skipped entirely, reducing latency and preserving Gemini's natural-sounding output. |

---

## License

MIT License — Built for the **Darwix AI Campus Challenge**.
