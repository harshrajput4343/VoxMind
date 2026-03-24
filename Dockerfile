FROM python:3.11-slim

# System dependencies: ffmpeg (pydub), espeak (pyttsx3 offline TTS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    espeak \
    espeak-data \
    libespeak1 \
    libespeak-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download VADER lexicon at build time
RUN python -c "from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer; SentimentIntensityAnalyzer()"

# Pre-download emotion models at build time (bakes into image)
RUN python -c "\
from transformers import pipeline; \
import os; os.makedirs('model_cache', exist_ok=True); \
p = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', cache_dir='model_cache'); \
print('Primary model downloaded')"

COPY . .
RUN mkdir -p outputs logs model_cache

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
