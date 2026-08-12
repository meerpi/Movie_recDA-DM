"""user_profile/review_processor.py — Dual-path review signal extraction.

Primary: Gemini Flash structured JSON extraction.
Fallback: local keyword + tag matching with sarcasm polarity heuristics.
"""

import json
import logging
import os
import re
import time


from user_profile.schema import UserProfile, _clamp

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


def _normalize_aspect(aspect):
    if isinstance(aspect, str):
        cleaned = aspect.strip().lower()
        if cleaned:
            return cleaned
    return None


class NonLLMReviewProcessor:

    def __init__(self, tag_vocabulary=None):
        self.tag_vocabulary = tag_vocabulary or []

    def detect_sarcasm_risk(self, review_text, star_rating):
        """Sarcasm risk (0-1) based on star rating vs text polarity contradiction."""
        text_lower = review_text.lower()
        lexical_hits = sum(bool(re.search(p, text_lower)) for p in SARCASM_PATTERNS)

        star_norm = (star_rating - 1.0) / 4.0

        pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)

        total = pos_count + neg_count
        if total > 0:
            text_polarity = (pos_count - neg_count) / total
        else:
            text_polarity = 0.0

        text_norm = (text_polarity + 1.0) / 2.0
        contradiction = abs(text_norm - star_norm)
        return min(1.0, contradiction * 0.7 + 0.3 * lexical_hits)

    def extract_fuzzy_genome_tags(self, review_text, cutoff=82):
        if not self.tag_vocabulary:
            return {}

        text_lower = review_text.lower()
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", text_lower))
        matched = {}

        for tag in self.tag_vocabulary:
            tc = tag.lower()
            if tc in tokens or (len(tc) > 4 and tc in text_lower):
                matched[tag] = 1.0

        return matched

    def process_review(self, profile, movie_card, star_rating, review_text="",
                       surgical_aspects=None):
        sarcasm_risk = self.detect_sarcasm_risk(review_text, star_rating) if review_text else 0.0
        fuzzy_tags = self.extract_fuzzy_genome_tags(review_text) if review_text else {}
        text_confidence = max(0.1, 1.0 - sarcasm_risk)

        profile.apply_rating_update(movie_card, star_rating=star_rating, review_confidence=text_confidence)

        aspect_deltas = {}

        if surgical_aspects:
            mult = (star_rating - 3.0) / 2.0
            for aspect in surgical_aspects:
                norm = _normalize_aspect(aspect)
                if norm and norm not in aspect_deltas:
                    aspect_deltas[norm] = mult * 0.4

        for tag, confidence in fuzzy_tags.items():
            norm = _normalize_aspect(tag)
            if norm and norm not in aspect_deltas:
                w = (star_rating - 3.0) / 2.0 * confidence * text_confidence
                aspect_deltas[norm] = w * 0.3

        for aspect, delta in aspect_deltas.items():
            profile.tag_affinity[aspect] = _clamp(profile.tag_affinity.get(aspect, 0.0) + delta)

        return {
            "mode": "non_llm_fallback",
            "sarcasm_risk": sarcasm_risk,
            "text_confidence": text_confidence,
            "extracted_fuzzy_tags": list(fuzzy_tags.keys()),
            "surgical_aspects": surgical_aspects or [],
            "star_rating": star_rating,
            "era_captured": bool(movie_card.get("year")),
            "language_captured": bool(movie_card.get("original_language")),
        }


class LLMReviewProcessor:

    def __init__(self, tag_vocabulary=None):
        self.tag_vocabulary = tag_vocabulary or []
        self.fallback = NonLLMReviewProcessor(tag_vocabulary)
        self.client = None

        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                self.client = genai.Client()
                logger.info("LLMReviewProcessor: Gemini client ready.")
            except Exception as e:
                logger.warning(f"Gemini client init failed ({e}). LLM extraction disabled.")
        else:
            logger.info("No GEMINI_API_KEY. Using local fallback for reviews.")

    def process_review(self, profile, movie_card, star_rating, review_text="",
                       surgical_aspects=None):
        if not review_text.strip() or not self.client:
            return self.fallback.process_review(profile, movie_card, star_rating, review_text, surgical_aspects)

        mult = (star_rating - 3.0) / 2.0

        try:
            t0 = time.time()
            prompt = (
                f"Movie Title: '{movie_card.get('title')}'\n"
                f"Genres: {movie_card.get('genres', [])}\n"
                f"Directors: {movie_card.get('directors', [])}\n"
                f"Star Rating: {star_rating} / 5.0\n"
                f"User Review Text: \"{review_text}\""
            )

            response = None
            for model in ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-lite-latest"]:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=[REVIEW_LLM_PROMPT, prompt],
                        config={"response_mime_type": "application/json"},
                    )
                    if response and response.text:
                        break
                except Exception as _e:
                    logger.debug(f"Review model {model} failed: {_e}")

            if not response or not response.text:
                return self.fallback.process_review(profile, movie_card, star_rating, review_text, surgical_aspects)

            llm_res = json.loads(response.text)
            latency_ms = round((time.time() - t0) * 1000, 2)

            sarcasm = llm_res.get("sarcasm_detected", False)
            sentiment = llm_res.get("sentiment_score", mult)
            confidence = 0.2 if sarcasm else 1.0

            profile.apply_rating_update(movie_card, star_rating=star_rating, review_confidence=confidence)

            aspect_deltas = {}

            if surgical_aspects:
                for aspect in surgical_aspects:
                    norm = _normalize_aspect(aspect)
                    if norm and norm not in aspect_deltas:
                        aspect_deltas[norm] = mult * 0.5 * confidence

            for aspect in (llm_res.get("liked_aspects") or []):
                norm = _normalize_aspect(aspect)
                if norm and norm not in aspect_deltas:
                    aspect_deltas[norm] = 0.4 * confidence

            for aspect in (llm_res.get("disliked_aspects") or []):
                norm = _normalize_aspect(aspect)
                if norm and norm not in aspect_deltas:
                    aspect_deltas[norm] = -0.4 * confidence

            for tag in (llm_res.get("extracted_tags") or []):
                norm = _normalize_aspect(tag)
                if norm and norm not in aspect_deltas:
                    aspect_deltas[norm] = sentiment * 0.3 * confidence

            for aspect, delta in aspect_deltas.items():
                profile.tag_affinity[aspect] = _clamp(profile.tag_affinity.get(aspect, 0.0) + delta)

            dir_sent = llm_res.get("director_sentiment")
            if dir_sent is not None:
                seen = set()
                for d in (movie_card.get("directors") or []):
                    nd = _normalize_aspect(d)
                    if nd and nd not in seen:
                        seen.add(nd)
                        profile.director_affinity[d] = _clamp(
                            profile.director_affinity.get(d, 0.0) + (dir_sent * profile.director_weight * confidence)
                        )

            seen_actors = set()
            for actor in (llm_res.get("standout_actors") or []):
                na = _normalize_aspect(actor)
                if na and na not in seen_actors:
                    seen_actors.add(na)
                    profile.actor_affinity[actor] = _clamp(
                        profile.actor_affinity.get(actor, 0.0) + (sentiment * profile.actor_weight * confidence)
                    )

            return {
                "mode": "gemini_llm",
                "latency_ms": latency_ms,
                "llm_output": llm_res,
                "surgical_aspects": surgical_aspects or [],
                "star_rating": star_rating,
            }

        except Exception as e:
            logger.warning(f"LLM review processing failed ({e}). Falling back to local.")
            return self.fallback.process_review(profile, movie_card, star_rating, review_text, surgical_aspects)
