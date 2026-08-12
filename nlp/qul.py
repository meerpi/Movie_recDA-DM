"""nlp/qul.py — Query understanding layer (LLM + local rules fallback)."""

from collections import defaultdict
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("cinevault.qul")


class DemonymMatch(BaseModel):
    demonym: str
    lang: str = Field(description="ISO 639-1 code")
    country: str = Field(description="lowercase country name")


class QULResult(BaseModel):
    expanded_query: str = Field(description="Descriptive prose for embedding, not a keyword dump.")
    bm25_keywords: list[str] = Field(description="3-10 concrete lexical terms for BM25.")
    matched_genres: list[str] = Field(default_factory=list)
    genre_strictness: Literal["hard_exclude", "soft_preference", "unspecified"]
    is_obscure_intent: bool = Field(description="True only for explicit obscure/cult/indie/hidden-gem signals.")
    negated_constraints: list[str] = Field(description="Explicitly excluded, not merely unmentioned.")
    detected_title: Optional[str] = Field(description="Only if query says 'like/similar to X'. Never inferred.")
    detected_demonym: Optional[DemonymMatch] = Field(description="Only if query names a national origin. Never guessed.")
    intent_type: Literal["title_reference", "general_search"]


SYSTEM_PROMPT = """You are the query understanding layer for CineVault. Your output feeds three separate retrieval lanes and a scoring stage — imprecision here has direct downstream cost, so do not guess or embellish.

- expanded_query feeds a dense embedding model: write descriptive prose, the way a critic would describe the ideal film.
- bm25_keywords feeds a lexical index: concrete literal terms (titles, cast/director names, specific nouns). Do not just repeat expanded_query here.
- matched_genres + genre_strictness drive a scoring penalty downstream. Mark "hard_exclude" only when the user would clearly reject a non-matching film. Mark "soft_preference" for tone/vibe language that merely leans toward a genre. Mark "unspecified" when no genre is implied.
- negated_constraints are things explicitly excluded. Tone adjustments ("less serious", "lighter") are NOT negations — leave them out; they belong in expanded_query as tone.
- detected_title and detected_demonym must stay null unless the query explicitly contains that signal. Never infer a title from mood alone, never guess an origin the query doesn't name.

VALID GENRES: {genres_str}

Query: "movies like Dark Knight but less serious but doesn't lose the philosophical depth"
-> detected_title: "The Dark Knight", intent_type: "title_reference", negated_constraints: [], matched_genres: [], genre_strictness: "unspecified"

Query: "psychological sci-fi thriller starring Keanu Reeves but not horror"
-> matched_genres: ["Sci-Fi", "Thriller"], genre_strictness: "hard_exclude", negated_constraints: ["horror"], detected_title: null

Query: "atmospheric slow burn Korean thriller"
-> detected_demonym: {{"demonym": "korean", "lang": "ko", "country": "south korea"}}, matched_genres: ["Thriller"], genre_strictness: "soft_preference"

Now analyze this query: "{raw_query}"
"""

GEMINI_MODELS_IN_PRIORITY_ORDER = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]


PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
TIER_A_PATH = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_PATH = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
TIER_C_PATH = PROJECT_ROOT / "dirtywork" / "tier_c_profile_cards.jsonl"

NEGATION_PATTERNS = re.compile(r"\b(?:not|no|less|without|non|except)\s+([a-zA-Z0-9\s-]+)", re.IGNORECASE)

DEMONYM_MAP = {
    "korean": {"lang": "ko", "country": "south korea"},
    "japanese": {"lang": "ja", "country": "japan"},
    "french": {"lang": "fr", "country": "france"},
    "spanish": {"lang": "es", "country": "spain"},
    "italian": {"lang": "it", "country": "italy"},
    "german": {"lang": "de", "country": "germany"},
    "british": {"lang": "en", "country": "united kingdom"},
    "chinese": {"lang": "zh", "country": "china"},
    "hong kong": {"lang": "zh", "country": "hong kong"},
    "indian": {"lang": "hi", "country": "india"},
    "mexican": {"lang": "es", "country": "mexico"},
    "swedish": {"lang": "sv", "country": "sweden"},
    "danish": {"lang": "da", "country": "denmark"},
    "russian": {"lang": "ru", "country": "russia"},
    "australian": {"lang": "en", "country": "australia"},
    "canadian": {"lang": "en", "country": "canada"},
}

CONCEPT_EXPANSIONS_PATH = PROJECT_ROOT / "model" / "concept_expansions.json"

try:
    with open(CONCEPT_EXPANSIONS_PATH, encoding="utf-8") as _f:
        _raw = json.load(_f)
    CONCEPT_EXPANSIONS = {k: v for k, v in _raw.items() if not k.startswith("_")}
except (FileNotFoundError, json.JSONDecodeError):
    CONCEPT_EXPANSIONS = {
        "dark magic": ["demons", "supernatural", "occult", "dark fantasy", "magic", "exorcism", "hell"],
        "magic": ["supernatural", "fantasy", "wizards", "sorcery", "magic"],
        "comic book": ["comic", "graphic novel", "superhero", "dc comics", "vertigo"],
        "comic": ["comic", "graphic novel", "superhero"],
        "dc": ["dc comics", "superhero", "constantine", "batman"],
        "marvel": ["marvel comics", "superhero"],
    }


class QueryUnderstandingLayer:
    """LLM + local-rules query understanding.  Accepts shared_cards to skip re-loading ~185MB JSONL."""

    def __init__(self, db_path=DB_PATH, shared_cards=None):
        t0 = time.time()
        logger.info("Initializing QUL (LLM + Local Rules)...")
        self.db_path = db_path

        self.cards = {}
        self.titles_dict = {}
        self.actor_vocab = set()
        self.director_vocab = set()
        self.tag_vocab = set()
        self.genres_vocab = {
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        }

        if shared_cards is not None:
            self.cards = shared_cards
            self._index_cards_from_shared()
        else:
            self._load_memory_assets()

        logger.info(f"QUL ready in {time.time() - t0:.2f}s — {len(self.titles_dict):,} titles, {len(self.tag_vocab):,} tags.")

    def _load_memory_assets(self):
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                rows = c.execute("""
                    SELECT m.movie_id, m.title 
                    FROM movies m
                    LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                    ORDER BY ms.num_ratings DESC
                """).fetchall()
                for mid, title in rows:
                    clean_t = title.lower().strip()
                    if clean_t not in self.titles_dict:
                        self.titles_dict[clean_t] = mid
                conn.close()
            except Exception as e:
                logger.error(f"Error loading DB titles: {e}")

        for pth in (TIER_A_PATH, TIER_B_PATH, TIER_C_PATH):
            if pth.exists():
                with open(pth, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            card = json.loads(line)
                            mid = card.get("movie_id")
                            if mid:
                                mid = int(mid)
                                self.cards[mid] = card

                                t = card.get("title")
                                if t and t.lower() not in self.titles_dict:
                                    self.titles_dict[t.lower()] = mid

                                for a in card.get("actors", []):
                                    self.actor_vocab.add(a)
                                for d in card.get("directors", []):
                                    self.director_vocab.add(d)
                                for tg in card.get("top_tags", []):
                                    tag_str = tg["tag"] if isinstance(tg, dict) else str(tg)
                                    self.tag_vocab.add(tag_str.lower())
                        except Exception:
                            continue

    def _index_cards_from_shared(self):
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                rows = c.execute("""
                    SELECT m.movie_id, m.title
                    FROM movies m
                    LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                    ORDER BY ms.num_ratings DESC
                """).fetchall()
                for mid, title in rows:
                    clean_t = title.lower().strip()
                    if clean_t not in self.titles_dict:
                        self.titles_dict[clean_t] = mid
                conn.close()
            except Exception as e:
                logger.error(f"Error loading DB titles: {e}")

        for mid, card in self.cards.items():
            t = card.get("title")
            if t and t.lower() not in self.titles_dict:
                self.titles_dict[t.lower()] = mid
            for a in card.get("actors", []):
                self.actor_vocab.add(a)
            for d in card.get("directors", []):
                self.director_vocab.add(d)
            for tg in card.get("top_tags", []):
                tag_str = tg["tag"] if isinstance(tg, dict) else str(tg)
                self.tag_vocab.add(tag_str.lower())
        logger.info(f"QUL indexed {len(self.cards):,} shared cards, {len(self.titles_dict):,} titles, {len(self.tag_vocab):,} tags.")

    def parse_query(self, raw_query):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                res = self._parse_query_llm(raw_query, api_key)
                if res and "expanded_query" in res:
                    return res
            except Exception as e:
                logger.warning(f"LLM QUL failed ({e}), falling back to local.")

        return self._parse_query_local(raw_query)

    def _parse_query_llm(self, raw_query, api_key):
        t0 = time.time()
        prompt = SYSTEM_PROMPT.format(genres_str=", ".join(sorted(self.genres_vocab)), raw_query=raw_query)

        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)

        response = None
        for model_name in GEMINI_MODELS_IN_PRIORITY_ORDER:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=QULResult,
                        temperature=0,
                    ),
                )
                if response and response.parsed:
                    break
            except Exception as e:
                logger.debug(f"Model {model_name} unavailable ({e}), trying next.")

        if not response or not response.parsed:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                import openai
                oa_client = openai.OpenAI(api_key=openai_key)
                oa_response = oa_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                data = QULResult.model_validate_json(oa_response.choices[0].message.content).model_dump()
                data.update(raw_query=raw_query, latency_ms=round((time.time() - t0) * 1000, 2), parser="llm_controlled_openai_fallback")
                return data
            raise RuntimeError("All Gemini endpoints failed and no OPENAI_API_KEY fallback set.")

        data = response.parsed.model_dump()
        data.update(raw_query=raw_query, latency_ms=round((time.time() - t0) * 1000, 2), parser="llm_controlled")
        return data

    def _parse_query_local(self, raw_query):
        t0 = time.time()
        q_lower = raw_query.lower().strip()

        matched_genres = []
        for g in sorted(self.genres_vocab):
            if re.search(r"\b" + re.escape(g.lower()) + r"\b", q_lower):
                matched_genres.append(g)

        matched_pacing = None
        for p in ("fast", "fast paced", "slow", "slow burn", "moderate", "relentless", "intense"):
            if re.search(r"\b" + re.escape(p) + r"\b", q_lower):
                matched_pacing = p
                break

        matched_tags = []
        for tag in self.tag_vocab:
            if len(tag) >= 3 and re.search(r"\b" + re.escape(tag) + r"\b", q_lower):
                matched_tags.append(tag)

        negations = [m.strip() for m in NEGATION_PATTERNS.findall(q_lower)]

        detected_demonym = None
        for demonym, meta in DEMONYM_MAP.items():
            if re.search(r"\b" + re.escape(demonym) + r"\b", q_lower):
                detected_demonym = {"demonym": demonym, "lang": meta.get("lang"), "country": meta.get("country")}
                break

        # title resolution for 'movies like X'
        detected_title = None
        matched_movie_id = None
        intent_type = "general_search"

        if q_lower in self.titles_dict:
            detected_title = q_lower.title()
            matched_movie_id = self.titles_dict[q_lower]
            intent_type = "title_reference"
        elif any(phrase in q_lower for phrase in ("movies like", "movie like", "similar to", "if i liked")):
            intent_type = "title_reference"
            raw_title_target = re.sub(r".*?(?:like|similar to|if i liked)\s+", "", q_lower, flags=re.I)
            raw_title_target = re.split(r"\b(?:but|with|and|not)\b", raw_title_target)[0].strip()

            if raw_title_target:
                target = raw_title_target.lower().strip()
                if target in self.titles_dict:
                    matched_key = target
                else:
                    matched_key = None
                    for tk in self.titles_dict:
                        if target == tk or (len(target) >= 4 and target in tk):
                            matched_key = tk
                            break

                if matched_key and matched_key in self.titles_dict:
                    detected_title = matched_key.title()
                    matched_movie_id = self.titles_dict[matched_key]

        # DNA + concept expansion
        expanded_terms = []
        if detected_demonym:
            if detected_demonym.get("country"):
                expanded_terms.append(detected_demonym["country"])
            if detected_demonym.get("demonym"):
                expanded_terms.append(detected_demonym["demonym"])

        if matched_movie_id and matched_movie_id in self.cards:
            card = self.cards[matched_movie_id]
            expanded_terms.extend(card.get("genres", []))
            expanded_terms.extend(card.get("themes", []))
            expanded_terms.extend(card.get("tone", []))
            expanded_terms.extend([t["tag"] if isinstance(t, dict) else str(t) for t in card.get("top_tags", [])[:8]])
            for d in card.get("directors", []):
                expanded_terms.append(d)

        for concept, terms in CONCEPT_EXPANSIONS.items():
            if concept in q_lower:
                expanded_terms.extend(terms)

        if expanded_terms:
            expanded = f"{raw_query} " + " ".join(expanded_terms[:12])
        else:
            expanded = raw_query

        for neg in negations:
            expanded = re.sub(r"\b" + re.escape(neg) + r"\b", "", expanded, flags=re.I)

        expanded = " ".join(expanded.split())

        is_obscure = any(
            w in q_lower
            for w in (
                "obscure", "indie", "hidden gem", "cult", "rare",
                "underrated", "under the radar", "unknown", "b-movie",
                "b movie", "b-tier", "b tier", "niche", "b-grade", "trash", "campy"
            )
        )

        return {
            "raw_query": raw_query,
            "expanded_query": expanded,
            "detected_title": detected_title,
            "matched_movie_id": matched_movie_id,
            "intent_type": intent_type,
            "is_obscure_intent": is_obscure,
            "matched_genres": matched_genres,
            "matched_pacing": matched_pacing,
            "matched_tags": matched_tags,
            "detected_demonym": detected_demonym,
            "negated_constraints": negations,
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "parser": "local_rules",
        }
