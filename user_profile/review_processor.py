#!/usr/bin/env python3
"""
user_profile/review_processor.py — Step 11: Dual-Path User Review Extractor & Profile Tuner

Extracts multi-dimensional taste signals from user ratings, surgical TUI checkboxes,
and natural language text reviews into structured `UserProfile` updates.

Features:
  • Path A (Surgical Checkboxes): Direct credit assignment for user-selected attributes.
  • Primary LLM Extractor (Gemini Flash): Structured JSON extraction matching Tier A card vocabulary.
  • Path B (Local Non-LLM Fallback): RapidFuzz fuzzy tag matching & sarcasm polarity heuristics when offline.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz import fuzz, process
from user_profile.schema import UserProfile

logger = logging.getLogger("cinevault.review_processor")

SARCASM_PATTERNS = [
    r"\b(oh wow|oh great|oh sure|yeah right|just what i needed|real genius|brilliant strategy)\b",
    r"\b(nobody|everyone) (asked for|needed|wanted) this\b",
    r"\b(real|truly|absolutely) (masterpiece|genius|brilliant)\b",
]

POSITIVE_WORDS = {"incredible", "amazing", "great", "masterpiece", "genius", "stunning", "loved", "brilliant", "excellent", "good"}
NEGATIVE_WORDS = {"terrible", "awful", "boring", "slow", "horrible", "waste", "trash", "bad", "disappointing", "predictable"}

REVIEW_LLM_PROMPT = """
You are the Taste Signal Analyzer for CineVault, a personalized movie recommendation system.
Analyze the user's movie rating and review text to extract structured taste signals.

Return ONLY a JSON object matching this exact schema:
{
  "sarcasm_detected": boolean,
  "sentiment_score": float between -1.0 and 1.0,
  "liked_aspects": list of strings (e.g. ["cinematography", "plot", "pacing", "acting", "directing", "soundtrack"]),
  "disliked_aspects": list of strings (e.g. ["pacing", "dialogue", "ending"]),
  "extracted_tags": list of descriptive keywords or themes extracted from the review,
  "director_sentiment": float between -1.0 and 1.0 or null,
  "standout_actors": list of strings (actor names praised or criticized)
}
"""


class NonLLMReviewProcessor:

    def __init__(self, tag_vocabulary: Optional[List[str]] = None):
        self.tag_vocabulary = tag_vocabulary or []

    def detect_sarcasm_risk(self, review_text: str, star_rating: float) -> float:
        """
        Calculates sarcasm risk (0.0 to 1.0) based on star rating vs text polarity contradiction.
        """
        text_lower = review_text.lower()
        lexical_hits = sum(bool(re.search(p, text_lower)) for p in SARCASM_PATTERNS)

        # Star rating normalized (1.0..5.0 -> 0.0..1.0)
        star_norm = (star_rating - 1.0) / 4.0

        pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words > 0:
            text_polarity = (pos_count - neg_count) / total_sentiment_words
        else:
            text_polarity = 0.0

        text_norm = (text_polarity + 1.0) / 2.0  # -1..+1 -> 0..1
        contradiction = abs(text_norm - star_norm)
        risk = min(1.0, contradiction * 0.7 + 0.3 * lexical_hits)
        return risk

    def extract_fuzzy_genome_tags(self, review_text: str, cutoff: int = 82) -> Dict[str, float]:
        """
        Extracts matching genome tags using fuzzy character similarity (RapidFuzz).
        Robust against typos and non-native English (e.g. "beautifull", "psycological").
        """
        if not self.tag_vocabulary:
            return {}

        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", review_text.lower())
        matched_tags = {}

        for token in tokens:
            match = process.extractOne(token, self.tag_vocabulary, scorer=fuzz.ratio, score_cutoff=cutoff)
            if match:
                tag, score, _ = match
                matched_tags[tag] = max(matched_tags.get(tag, 0.0), score / 100.0)

        return matched_tags

    def process_review(
        self,
        profile: UserProfile,
        movie_card: Dict[str, Any],
        star_rating: float,
        review_text: str = "",
        surgical_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes review text using local RapidFuzz heuristics and updates UserProfile.
        """
        sarcasm_risk = self.detect_sarcasm_risk(review_text, star_rating) if review_text else 0.0
        fuzzy_tags = self.extract_fuzzy_genome_tags(review_text) if review_text else {}
        text_confidence = max(0.1, 1.0 - sarcasm_risk)

        # 1. Base star rating update
        profile.apply_rating_update(movie_card, star_rating=star_rating, review_confidence=1.0)

        # 2. Surgical checkbox credit assignment (Path A)
        if surgical_aspects:
            mult = (star_rating - 3.0) / 2.0  # -1.0 to +1.0
            for aspect in surgical_aspects:
                aspect_clean = aspect.lower().strip()
                profile.tag_affinity[aspect_clean] = profile.tag_affinity.get(aspect_clean, 0.0) + mult * 0.4

        # 3. Fuzzy tag updates
        for tag, confidence in fuzzy_tags.items():
            current_aff = profile.tag_affinity.get(tag, 0.0)
            weight = (star_rating - 3.0) / 2.0 * confidence * text_confidence
            profile.tag_affinity[tag] = current_aff + weight * 0.3

        return {
            "mode": "non_llm_fallback",
            "sarcasm_risk": sarcasm_risk,
            "text_confidence": text_confidence,
            "extracted_fuzzy_tags": list(fuzzy_tags.keys()),
            "surgical_aspects": surgical_aspects or [],
            "star_rating": star_rating,
        }


class LLMReviewProcessor:
    """
    Dual-Path Review Processor combining Gemini LLM extraction with local RapidFuzz fallback.
    """

    def __init__(self, tag_vocabulary: Optional[List[str]] = None):
        self.tag_vocabulary = tag_vocabulary or []
        self.fallback = NonLLMReviewProcessor(tag_vocabulary)
        self.client = None

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self.client = genai.Client()
                logger.info("LLMReviewProcessor initialized with Gemini API client.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini genai client ({e}). LLM review extraction disabled.")
        else:
            logger.info("GEMINI_API_KEY not found. LLMReviewProcessor will use local Path B fallback.")

    def process_review(
        self,
        profile: UserProfile,
        movie_card: Dict[str, Any],
        star_rating: float,
        review_text: str = "",
        surgical_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Processes user rating, surgical checkboxes, and natural language review text.
        Fuses LLM extraction (Primary) with local fallback (Path B).
        """
        # Always apply surgical checkbox updates (Path A)
        mult = (star_rating - 3.0) / 2.0  # -1.0 to +1.0
        if surgical_aspects:
            for aspect in surgical_aspects:
                aspect_clean = aspect.lower().strip()
                profile.tag_affinity[aspect_clean] = profile.tag_affinity.get(aspect_clean, 0.0) + mult * 0.5

        # Base rating update
        profile.apply_rating_update(movie_card, star_rating=star_rating)

        # If no text provided or no API client available, use local path B
        if not review_text.strip() or not self.client:
            return self.fallback.process_review(profile, movie_card, star_rating, review_text, surgical_aspects)

        # Execute LLM Extraction
        try:
            t0 = time.time()
            prompt = (
                f"Movie Title: '{movie_card.get('title')}'\n"
                f"Genres: {movie_card.get('genres', [])}\n"
                f"Directors: {movie_card.get('directors', [])}\n"
                f"Star Rating: {star_rating} / 5.0\n"
                f"User Review Text: \"{review_text}\""
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[REVIEW_LLM_PROMPT, prompt],
                config={"response_mime_type": "application/json"},
            )

            llm_res = json.loads(response.text)
            latency_ms = round((time.time() - t0) * 1000, 2)

            # Apply LLM extracted signals to profile
            sarcasm = llm_res.get("sarcasm_detected", False)
            sentiment_score = llm_res.get("sentiment_score", mult)
            confidence = 0.2 if sarcasm else 1.0

            # Update liked / disliked aspects & tags
            for aspect in llm_res.get("liked_aspects", []):
                profile.tag_affinity[aspect.lower()] = profile.tag_affinity.get(aspect.lower(), 0.0) + 0.4 * confidence
            for aspect in llm_res.get("disliked_aspects", []):
                profile.tag_affinity[aspect.lower()] = profile.tag_affinity.get(aspect.lower(), 0.0) - 0.4 * confidence
            for tag in llm_res.get("extracted_tags", []):
                profile.tag_affinity[tag.lower()] = profile.tag_affinity.get(tag.lower(), 0.0) + (sentiment_score * 0.3 * confidence)

            # Director sentiment update
            dir_sent = llm_res.get("director_sentiment")
            if dir_sent is not None:
                for director in movie_card.get("directors", []):
                    profile.director_affinity[director] = profile.director_affinity.get(director, 0.0) + (dir_sent * profile.director_weight * confidence)

            # Actor sentiment updates
            for actor in llm_res.get("standout_actors", []):
                profile.actor_affinity[actor] = profile.actor_affinity.get(actor, 0.0) + (sentiment_score * profile.actor_weight * confidence)

            logger.info(f"LLM review processing completed in {latency_ms}ms for user='{profile.user_id}'.")
            return {
                "mode": "gemini_llm",
                "latency_ms": latency_ms,
                "llm_output": llm_res,
                "surgical_aspects": surgical_aspects or [],
                "star_rating": star_rating,
            }

        except Exception as e:
            logger.warning(f"LLM Review Processing failed ({e}). Falling back to local Path B processor.")
            return self.fallback.process_review(profile, movie_card, star_rating, review_text, surgical_aspects)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = LLMReviewProcessor()
    user = UserProfile(user_id="test_reviewer")
    movie = {
        "movie_id": 58559,
        "title": "The Dark Knight",
        "genres": ["Action", "Crime", "Drama"],
        "directors": ["Christopher Nolan"],
        "actors": ["Christian Bale", "Heath Ledger"]
    }

    result = processor.process_review(
        profile=user,
        movie_card=movie,
        star_rating=5.0,
        review_text="Brilliant cinematography and incredible acting by Heath Ledger, but a bit long.",
        surgical_aspects=["Visuals", "Performances"]
    )

    print("\nReview Processing Result Mode:", result.get("mode"))
    print("User Tag Affinities:", user.tag_affinity)
    print("User Director Affinities:", user.director_affinity)
