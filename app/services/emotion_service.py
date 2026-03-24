"""
Emotion Detection Service — 4-level fallback chain.

Detection hierarchy:
  Level 1: j-hartmann/emotion-english-distilroberta-base (most accurate, 7 emotions)
  Level 2: SamLowe/roberta-base-go_emotions (28 emotions mapped to 7 canonical)
  Level 3: bhadresh-savani/distilbert-base-uncased-emotion (6 emotions mapped to 7)
  Level 4: VADER (offline, maps compound score to emotion)

Design: Never raises from detect() — always returns a valid EmotionResult.
"""

import os
from app.models.schemas import EmotionResult
from app.core.config import settings
from app.core.logger import logger


class EmotionService:
    """Multi-model emotion detection with graceful degradation."""

    # Maps any model-specific label → our 7 canonical labels
    EMOTION_MAP: dict[str, str] = {
        # Primary model (j-hartmann) labels
        "joy": "joy", "sadness": "sadness", "anger": "anger",
        "fear": "fear", "disgust": "disgust", "surprise": "surprise",
        "neutral": "neutral",
        # GoEmotions granular → canonical
        "admiration": "joy", "amusement": "joy", "excitement": "joy",
        "gratitude": "joy", "love": "joy", "optimism": "joy",
        "relief": "joy", "pride": "joy",
        "grief": "sadness", "remorse": "sadness", "disappointment": "sadness",
        "embarrassment": "sadness",
        "annoyance": "anger", "disapproval": "anger",
        "nervousness": "fear",
        "realization": "surprise",
        "confusion": "neutral", "curiosity": "surprise",
        "caring": "neutral", "desire": "joy",
        # VADER compound mappings
        "positive": "joy", "negative": "sadness",
    }

    def __init__(self) -> None:
        """Initialise emotion service, attempting to load all detection models."""
        self._primary_pipeline = None
        self._fallback_pipeline = None
        self._tertiary_pipeline = None
        self._vader_analyzer = None
        self._primary_loaded: bool = False
        self._fallback_loaded: bool = False
        self._tertiary_loaded: bool = False
        self._load_models()

    def _load_models(self) -> None:
        """
        Attempt to load all three detection backends.
        Failures are logged as warnings, never raised.
        Sets _primary_loaded and _fallback_loaded flags.
        """
        os.environ["TRANSFORMERS_CACHE"] = str(settings.model_cache_path)

        # Load primary — j-hartmann distilroberta
        try:
            from transformers import pipeline
            self._primary_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=-1,
            )
            self._primary_loaded = True
            logger.success("Primary emotion model loaded: j-hartmann/emotion-english-distilroberta-base")
        except Exception as e:
            logger.warning(f"Primary model failed to load: {e}")

        # Load fallback — SamLowe GoEmotions
        try:
            from transformers import pipeline
            self._fallback_pipeline = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=None,
                device=-1,
            )
            self._fallback_loaded = True
            logger.success("Fallback emotion model loaded: SamLowe/roberta-base-go_emotions")
        except Exception as e:
            logger.warning(f"Fallback model failed to load: {e}")

        # Load tertiary — bhadresh-savani distilbert-base-uncased-emotion
        try:
            from transformers import pipeline
            self._tertiary_pipeline = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                top_k=None,
                device=-1,
            )
            self._tertiary_loaded = True
            logger.success("Tertiary emotion model loaded: bhadresh-savani/distilbert-base-uncased-emotion")
        except Exception as e:
            logger.warning(f"Tertiary model failed to load: {e}")

        # Load VADER (always available once installed)
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader_analyzer = SentimentIntensityAnalyzer()
            logger.success("VADER sentiment analyzer loaded")
        except Exception as e:
            logger.error(f"VADER failed to load: {e}")

    def models_loaded(self) -> dict[str, bool]:
        """Return dict showing which backends are available."""
        return {
            "primary_transformer": self._primary_loaded,
            "fallback_transformer": self._fallback_loaded,
            "tertiary_transformer": self._tertiary_loaded,
            "vader": self._vader_analyzer is not None,
        }

    def detect(self, text: str) -> EmotionResult:
        """
        Detect emotion from text using a 3-level fallback chain.

        Level 1: j-hartmann transformer (most accurate)
        Level 2: SamLowe GoEmotions transformer
        Level 3: VADER (always available, 100% offline)

        Never raises — returns neutral EmotionResult on total failure.

        Args:
            text: Input text to analyse.

        Returns:
            EmotionResult with emotion label, intensity, and full scores.
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to detect(), returning neutral")
            return self._neutral_result()

        # Truncate to 512 tokens max (transformer limit)
        text = text[:512]

        if self._primary_loaded:
            try:
                result = self._detect_with_primary(text)
                logger.debug(f"Primary model: {result.emotion} ({result.intensity:.2f})")
                return result
            except Exception as e:
                logger.warning(f"Primary detection failed: {e}, falling back")

        if self._fallback_loaded:
            try:
                result = self._detect_with_fallback(text)
                logger.debug(f"Fallback model: {result.emotion} ({result.intensity:.2f})")
                return result
            except Exception as e:
                logger.warning(f"Fallback detection failed: {e}, falling back to tertiary")

        if self._tertiary_loaded:
            try:
                result = self._detect_with_tertiary(text)
                logger.debug(f"Tertiary model: {result.emotion} ({result.intensity:.2f})")
                return result
            except Exception as e:
                logger.warning(f"Tertiary detection failed: {e}, falling back to VADER")

        if self._vader_analyzer:
            try:
                result = self._detect_with_vader(text)
                logger.debug(f"VADER: {result.emotion} ({result.intensity:.2f})")
                return result
            except Exception as e:
                logger.error(f"VADER detection failed: {e}")

        logger.error("All emotion backends exhausted, returning neutral")
        return self._neutral_result()

    def _detect_with_primary(self, text: str) -> EmotionResult:
        """
        Run j-hartmann/emotion-english-distilroberta-base.
        Returns top emotion with full score dict.

        Args:
            text: Input text.

        Returns:
            EmotionResult from the primary model.
        """
        raw = self._primary_pipeline(text)[0]  # type: ignore[index]
        sorted_scores = sorted(raw, key=lambda x: x["score"], reverse=True)
        top = sorted_scores[0]
        emotion = self.EMOTION_MAP.get(top["label"].lower(), "neutral")
        all_scores = {
            self.EMOTION_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 4)
            for r in raw
        }
        return EmotionResult(
            emotion=emotion,
            intensity=round(top["score"], 4),
            all_scores=all_scores,
            model_used="j-hartmann/emotion-english-distilroberta-base",
            intensity_label=self._get_intensity_label(top["score"]),
        )

    def _detect_with_fallback(self, text: str) -> EmotionResult:
        """
        Run SamLowe/roberta-base-go_emotions (28 labels).
        Maps granular labels to our 7 canonical emotions.

        Args:
            text: Input text.

        Returns:
            EmotionResult from the fallback model.
        """
        raw = self._fallback_pipeline(text)[0]  # type: ignore[index]
        sorted_scores = sorted(raw, key=lambda x: x["score"], reverse=True)
        top = sorted_scores[0]
        emotion = self.EMOTION_MAP.get(top["label"].lower(), "neutral")
        intensity = top["score"]

        # Aggregate all 28 scores into our 7 canonical buckets
        canonical_scores: dict[str, float] = {
            e: 0.0 for e in ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
        }
        for r in raw:
            bucket = self.EMOTION_MAP.get(r["label"].lower(), "neutral")
            canonical_scores[bucket] = max(canonical_scores[bucket], r["score"])

        return EmotionResult(
            emotion=emotion,
            intensity=round(intensity, 4),
            all_scores={k: round(v, 4) for k, v in canonical_scores.items()},
            model_used="SamLowe/roberta-base-go_emotions",
            intensity_label=self._get_intensity_label(intensity),
        )

    def _detect_with_tertiary(self, text: str) -> EmotionResult:
        """
        Run bhadresh-savani/distilbert-base-uncased-emotion.
        Maps 6 labels (sadness, joy, love, anger, fear, surprise) to our 7 canonical.

        Args:
            text: Input text.

        Returns:
            EmotionResult from the tertiary model.
        """
        raw = self._tertiary_pipeline(text)[0]  # type: ignore[index]
        sorted_scores = sorted(raw, key=lambda x: x["score"], reverse=True)
        top = sorted_scores[0]

        # bhadresh-savani uses: sadness, joy, love, anger, fear, surprise
        bhadresh_map = {
            "sadness": "sadness", "joy": "joy", "love": "joy",
            "anger": "anger", "fear": "fear", "surprise": "surprise",
        }
        emotion = bhadresh_map.get(top["label"].lower(), "neutral")

        all_scores: dict[str, float] = {
            e: 0.0 for e in ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]
        }
        for r in raw:
            bucket = bhadresh_map.get(r["label"].lower(), "neutral")
            all_scores[bucket] = max(all_scores[bucket], r["score"])

        return EmotionResult(
            emotion=emotion,
            intensity=round(top["score"], 4),
            all_scores={k: round(v, 4) for k, v in all_scores.items()},
            model_used="bhadresh-savani/distilbert-base-uncased-emotion",
            intensity_label=self._get_intensity_label(top["score"]),
        )

    def _detect_with_vader(self, text: str) -> EmotionResult:
        """
        Use VADER SentimentIntensityAnalyzer as last resort.
        Maps compound score to joy/sadness/neutral with sub-scores.

        Args:
            text: Input text.

        Returns:
            EmotionResult from VADER analysis.
        """
        scores = self._vader_analyzer.polarity_scores(text)  # type: ignore[union-attr]
        compound = scores["compound"]

        if compound >= 0.05:
            emotion, intensity = "joy", min(abs(compound), 1.0)
        elif compound <= -0.05:
            emotion, intensity = "sadness", min(abs(compound), 1.0)
        else:
            emotion, intensity = "neutral", 1.0 - abs(compound)

        all_scores = {
            "joy": round(scores["pos"], 4),
            "sadness": round(scores["neg"], 4),
            "neutral": round(scores["neu"], 4),
            "anger": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "surprise": 0.0,
        }
        return EmotionResult(
            emotion=emotion,
            intensity=round(intensity, 4),
            all_scores=all_scores,
            model_used="VADER",
            intensity_label=self._get_intensity_label(intensity),
        )

    def _get_intensity_label(self, score: float) -> str:
        """
        Map a 0-1 score to a human label.

        Args:
            score: Confidence score between 0.0 and 1.0.

        Returns:
            'low', 'medium', or 'high'.
        """
        if score < 0.4:
            return "low"
        if score < 0.7:
            return "medium"
        return "high"

    def _neutral_result(self) -> EmotionResult:
        """Return a safe neutral EmotionResult for error / empty-text cases."""
        return EmotionResult(
            emotion="neutral",
            intensity=1.0,
            all_scores={
                "neutral": 1.0, "joy": 0.0, "sadness": 0.0,
                "anger": 0.0, "fear": 0.0, "disgust": 0.0, "surprise": 0.0,
            },
            model_used="fallback_default",
            intensity_label="high",
        )
