"""user_profile/schema.py — UserProfile dataclass and scoring."""

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

_CLAMP = (-3.0, 3.0)


def _clamp(val: float) -> float:
    return max(_CLAMP[0], min(_CLAMP[1], val))


@dataclass
class UserProfile:

    # identity & config
    user_id: str = "default_user"
    personalization_lambda: float = 0.7  # 0 = pure profile, 1 = pure relevance

    # signal sensitivity weights
    director_weight: float = 0.80
    actor_weight: float = 0.50
    genre_weight: float = 0.60
    tag_weight: float = 0.90
    pacing_weight: float = 0.75
    tone_weight: float = 0.40
    era_weight: float = 0.30
    language_weight: float = 0.50

    # behavioral history
    # each entry: {movie_id, title, star_rating, rated_at_ts}
    rating_log: List[Dict[str, Any]] = field(default_factory=list)

    watch_history: Set[int] = field(default_factory=set)    # all interacted
    highly_rated: Set[int] = field(default_factory=set)     # star >= 4.0
    poorly_rated: Set[int] = field(default_factory=set)     # star <= 2.0
    watchlist: Set[int] = field(default_factory=set)
    abandoned: Set[int] = field(default_factory=set)
    rewatched: Set[int] = field(default_factory=set)

    disliked_actors: List[str] = field(default_factory=list)
    disliked_directors: List[str] = field(default_factory=list)
    dealbreakers: List[str] = field(default_factory=list)  # hard-veto tags/themes

    # affinity signals
    genre_affinity: Dict[str, float] = field(default_factory=dict)
    tag_affinity: Dict[str, float] = field(default_factory=dict)
    tone_affinity: Dict[str, float] = field(default_factory=dict)
    pacing_affinity: Dict[str, float] = field(default_factory=dict)
    actor_affinity: Dict[str, float] = field(default_factory=dict)
    director_affinity: Dict[str, float] = field(default_factory=dict)
    era_affinity: Dict[str, float] = field(default_factory=dict)
    language_affinity: Dict[str, float] = field(default_factory=dict)
    country_affinity: Dict[str, float] = field(default_factory=dict)
    content_rating_affinity: Dict[str, float] = field(default_factory=dict)

    runtime_preference: str = "any"      # "short" (<90m), "standard", "epic" (>150m)
    franchise_tolerance: float = 0.5     # 0 = hates sequels, 1 = loves them

    # session & query tracking
    # each entry: {query, ts, result_count, top_result_id}
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    disabled_signals: List[str] = field(default_factory=list)

    last_session_ts: Optional[int] = None
    session_count: int = 0
    avg_queries_per_session: float = 0.0
    recency_halflife_days: int = 90

    # taste vectors (dense embeddings, recomputed periodically)
    dense_taste_vector: Optional[np.ndarray] = None     # 1024d VoyageAI centroid
    genome_taste_vector: Optional[np.ndarray] = None    # 1128d genome centroid
    taste_vector_confidence: float = 0.0
    taste_vector_updated_at: Optional[int] = None

    # meta-learning / confidence
    genre_confidence: Dict[str, int] = field(default_factory=dict)
    director_confidence: Dict[str, int] = field(default_factory=dict)
    actor_confidence: Dict[str, int] = field(default_factory=dict)

    # {ts, movie_id, title, reason} — when user says rec was wrong
    correction_log: List[Dict[str, Any]] = field(default_factory=list)

    diversity_appetite: float = 0.5
    exploration_rate: float = 0.5

    # profile screen personalization
    signal_weights: Dict[str, str] = field(default_factory=lambda: {
        "watch_history": "balanced",
        "ratings": "balanced",
        "reviews": "balanced",
    })
    memory_entries: List[str] = field(default_factory=list)

    _MIN_CONFIDENCE_INTERACTIONS: int = field(default=3, init=False, repr=False)
    _MAX_QUERY_HISTORY: int = field(default=50, init=False, repr=False)

    # optimistic locking token — set by store.load_profile()
    _db_version: int = field(default=0, init=False, repr=False)

    # --- scoring ---

    @staticmethod
    def _extract_cast(movie_item: Dict[str, Any]) -> List[str]:
        raw_actors = movie_item.get("actors")
        raw_cast = movie_item.get("cast")
        actors_list = raw_actors if isinstance(raw_actors, list) else []
        cast_list = raw_cast if isinstance(raw_cast, list) else []

        seen = set()
        combined = []
        for item in actors_list + cast_list:
            if isinstance(item, str) and item and item not in seen:
                seen.add(item)
                combined.append(item)
        return combined

    @staticmethod
    def _extract_content_terms(movie_item: Dict[str, Any]) -> Set[str]:
        terms = set()
        for g in (movie_item.get("genres") or []):
            if isinstance(g, str) and g.strip():
                terms.add(g.strip().lower())
        for t in (movie_item.get("tone") or []):
            if isinstance(t, str) and t.strip():
                terms.add(t.strip().lower())
        for th in (movie_item.get("themes") or []):
            if isinstance(th, str) and th.strip():
                terms.add(th.strip().lower())
        for tag_item in (movie_item.get("top_tags") or []):
            tag_name = tag_item if isinstance(tag_item, str) else (tag_item.get("tag", "") if isinstance(tag_item, dict) else "")
            if isinstance(tag_name, str) and tag_name.strip():
                terms.add(tag_name.strip().lower())
        pacing = movie_item.get("pacing")
        if isinstance(pacing, str) and pacing.strip():
            terms.add(pacing.strip().lower())
        return terms

    @staticmethod
    def _is_dealbreaker_match(dealbreaker: str, content_terms: Set[str]) -> bool:
        db_clean = dealbreaker.strip().lower()
        if not db_clean or not content_terms:
            return False
        if db_clean in content_terms:
            return True
        pattern = r"\b" + re.escape(db_clean) + r"\b"
        return any(bool(re.search(pattern, term)) for term in content_terms)

    def calculate_profile_boost(self, movie_item: Dict[str, Any]) -> float:
        """Personalized boost for a candidate. Returns -10.0 for hard vetoes
        (watched / dealbreaker), otherwise sums weighted affinity across all
        signal dimensions. Confidence-gated for actor/director."""
        mid = movie_item.get("movie_id")

        if mid in self.watch_history:
            return -10.0

        if self.dealbreakers:
            content_terms = self._extract_content_terms(movie_item)
            for db in self.dealbreakers:
                if self._is_dealbreaker_match(db, content_terms):
                    return -10.0

        score = 0.0

        cast = self._extract_cast(movie_item)
        for actor in cast:
            if actor in self.actor_affinity:
                gate = self._confidence_gate(self.actor_confidence.get(actor, 0))
                score += self.actor_affinity[actor] * self.actor_weight * gate

        directors = movie_item.get("directors") or []
        if not directors and movie_item.get("director"):
            directors = [movie_item["director"]]
        for d in directors:
            if d in self.director_affinity:
                gate = self._confidence_gate(self.director_confidence.get(d, 0))
                score += self.director_affinity[d] * self.director_weight * gate

        for genre in (movie_item.get("genres") or []):
            score += self.genre_affinity.get(genre, 0.0) * self.genre_weight

        for t in (movie_item.get("tone") or []):
            score += self.tone_affinity.get(t, 0.0) * self.tone_weight

        for tag_item in (movie_item.get("top_tags") or []):
            tag_name = tag_item if isinstance(tag_item, str) else (tag_item.get("tag", "") if isinstance(tag_item, dict) else "")
            if tag_name:
                score += self.tag_affinity.get(tag_name, 0.0) * self.tag_weight

        pacing = movie_item.get("pacing")
        if pacing:
            score += self.pacing_affinity.get(pacing, 0.0) * self.pacing_weight

        if self.era_affinity:
            year = movie_item.get("year")
            if year:
                try:
                    era = self.get_era_from_year(int(year))
                    score += self.era_affinity.get(era, 0.0) * self.era_weight
                except (ValueError, TypeError):
                    pass

        if self.language_affinity:
            lang = movie_item.get("original_language", "")
            if lang:
                score += self.language_affinity.get(lang, 0.0) * self.language_weight

        if self.country_affinity:
            for country in (movie_item.get("production_countries") or []):
                # piggybacks on language_weight * 0.8 — no separate country weight yet
                score += self.country_affinity.get(country, 0.0) * self.language_weight * 0.8

        if self.content_rating_affinity:
            cr = movie_item.get("content_rating", "")
            if cr:
                score += self.content_rating_affinity.get(cr, 0.0) * 0.3

        return score

    def _confidence_gate(self, count):
        return 1.0 if count >= 3 else 0.5

    # --- profile updates ---

    def apply_rating_update(self, movie_item, star_rating, review_confidence=1.0):
        """Updates all affinity signals after a rating. Weight is normalized
        to [-1, +1] via (star - 3) / 2, clamped to [-3, 3]."""
        mid = movie_item.get("movie_id")
        title = movie_item.get("title", f"Movie #{mid}")

        if mid:
            self.rating_log.append({
                "movie_id": mid,
                "title": title,
                "star_rating": star_rating,
                "rated_at_ts": int(time.time()),
            })
            self.watch_history.add(mid)
            if mid in self.watchlist:
                self.watchlist.discard(mid)
            if star_rating >= 4.0:
                self.highly_rated.add(mid)
            elif star_rating <= 2.0:
                self.poorly_rated.add(mid)

        w = (star_rating - 3.0) / 2.0 * review_confidence

        for g in (movie_item.get("genres") or []):
            self.genre_affinity[g] = _clamp(self.genre_affinity.get(g, 0.0) + w)
            self.genre_confidence[g] = self.genre_confidence.get(g, 0) + 1

        for t in (movie_item.get("tone") or []):
            self.tone_affinity[t] = _clamp(self.tone_affinity.get(t, 0.0) + w)

        for tag_item in (movie_item.get("top_tags") or []):
            tag_name = tag_item if isinstance(tag_item, str) else (tag_item.get("tag", "") if isinstance(tag_item, dict) else "")
            if tag_name:
                self.tag_affinity[tag_name] = _clamp(self.tag_affinity.get(tag_name, 0.0) + w)

        pacing = movie_item.get("pacing")
        if pacing:
            self.pacing_affinity[pacing] = _clamp(self.pacing_affinity.get(pacing, 0.0) + w)

        year = movie_item.get("year")
        if year:
            try:
                era = self.get_era_from_year(int(year))
                self.era_affinity[era] = _clamp(self.era_affinity.get(era, 0.0) + w)
            except (ValueError, TypeError):
                pass

        lang = movie_item.get("original_language", "")
        if lang:
            self.language_affinity[lang] = _clamp(self.language_affinity.get(lang, 0.0) + w)

        for country in (movie_item.get("production_countries") or []):
            if country:
                self.country_affinity[country] = _clamp(self.country_affinity.get(country, 0.0) + w)

        cr = movie_item.get("content_rating", "")
        if cr:
            self.content_rating_affinity[cr] = _clamp(self.content_rating_affinity.get(cr, 0.0) + w)

        cast = self._extract_cast(movie_item)
        for actor in cast:
            self.actor_affinity[actor] = _clamp(self.actor_affinity.get(actor, 0.0) + w)
            self.actor_confidence[actor] = self.actor_confidence.get(actor, 0) + 1

        directors = movie_item.get("directors") or []
        if not directors and movie_item.get("director"):
            directors = [movie_item["director"]]
        for d in directors:
            self.director_affinity[d] = _clamp(self.director_affinity.get(d, 0.0) + w)
            self.director_confidence[d] = self.director_confidence.get(d, 0) + 1

        if movie_item.get("collection"):
            if star_rating >= 4.0:
                self.franchise_tolerance = min(1.0, self.franchise_tolerance + 0.03)
            elif star_rating <= 2.0:
                self.franchise_tolerance = max(0.0, self.franchise_tolerance - 0.03)

        self._recompute_exploration_rate()
        self.taste_vector_confidence = self.compute_taste_confidence()

    def _recompute_exploration_rate(self):
        total = len(self.watch_history)
        if total < 5:
            return
        # rough proxy: genre spread relative to total watches
        genre_spread = len(self.genre_affinity)
        self.exploration_rate = min(1.0, genre_spread / max(1, total * 0.3))

    def add_to_watchlist(self, movie_id: int):
        self.watchlist.add(movie_id)

    def mark_abandoned(self, movie_id: int):
        self.abandoned.add(movie_id)
        self.watch_history.add(movie_id)

    def log_correction(self, movie_id, title, reason):
        self.correction_log.append({
            "ts": int(time.time()),
            "movie_id": movie_id,
            "title": title,
            "reason": reason,
        })

    def clear_query_history(self):
        self.query_history.clear()

    def record_query(self, query, result_count=0, top_result_id=None):
        if "query_history" not in self.disabled_signals:
            entry = {
                "query": query,
                "ts": int(time.time()),
                "result_count": result_count,
                "top_result_id": top_result_id,
            }
            self.query_history.append(entry)
            if len(self.query_history) > 50:
                self.query_history = self.query_history[-50:]

        self.session_count += 1
        n = self.session_count
        self.avg_queries_per_session = ((self.avg_queries_per_session * (n - 1)) + 1) / n
        self.last_session_ts = int(time.time())

    @staticmethod
    def get_era_from_year(year: int) -> str:
        decade = (year // 10) * 10
        return f"{decade}s"

    def compute_taste_confidence(self):
        return min(1.0, len(self.watch_history) / 50.0)

    def get_recent_queries(self, n=10):
        return [q["query"] for q in self.query_history[-n:]]

    def get_top_genres(self, n=5):
        return sorted(self.genre_affinity, key=self.genre_affinity.get, reverse=True)[:n]

    def get_top_directors(self, n=5):
        return sorted(self.director_affinity, key=self.director_affinity.get, reverse=True)[:n]

    # --- serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "personalization_lambda": self.personalization_lambda,
            "director_weight": self.director_weight,
            "actor_weight": self.actor_weight,
            "genre_weight": self.genre_weight,
            "tag_weight": self.tag_weight,
            "pacing_weight": self.pacing_weight,
            "tone_weight": self.tone_weight,
            "era_weight": self.era_weight,
            "language_weight": self.language_weight,
            "rating_log": self.rating_log,
            "watch_history": list(self.watch_history),
            "highly_rated": list(self.highly_rated),
            "poorly_rated": list(self.poorly_rated),
            "watchlist": list(self.watchlist),
            "abandoned": list(self.abandoned),
            "rewatched": list(self.rewatched),
            "disliked_actors": self.disliked_actors,
            "disliked_directors": self.disliked_directors,
            "dealbreakers": self.dealbreakers,
            "genre_affinity": self.genre_affinity,
            "tag_affinity": self.tag_affinity,
            "tone_affinity": self.tone_affinity,
            "pacing_affinity": self.pacing_affinity,
            "actor_affinity": self.actor_affinity,
            "director_affinity": self.director_affinity,
            "era_affinity": self.era_affinity,
            "language_affinity": self.language_affinity,
            "country_affinity": self.country_affinity,
            "content_rating_affinity": self.content_rating_affinity,
            "runtime_preference": self.runtime_preference,
            "franchise_tolerance": self.franchise_tolerance,
            "query_history": self.query_history,
            "disabled_signals": self.disabled_signals,
            "last_session_ts": self.last_session_ts,
            "session_count": self.session_count,
            "avg_queries_per_session": self.avg_queries_per_session,
            "recency_halflife_days": self.recency_halflife_days,
            # numpy arrays not serializable, skip
            "taste_vector_confidence": self.taste_vector_confidence,
            "taste_vector_updated_at": self.taste_vector_updated_at,
            "genre_confidence": self.genre_confidence,
            "director_confidence": self.director_confidence,
            "actor_confidence": self.actor_confidence,
            "correction_log": self.correction_log,
            "diversity_appetite": self.diversity_appetite,
            "exploration_rate": self.exploration_rate,
            "signal_weights": self.signal_weights,
            "memory_entries": self.memory_entries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """All new fields use safe defaults — backward-compatible with v1 profiles."""
        return cls(
            user_id=data.get("user_id", "default_user"),
            personalization_lambda=data.get("personalization_lambda", 0.7),
            director_weight=data.get("director_weight", 0.80),
            actor_weight=data.get("actor_weight", 0.50),
            genre_weight=data.get("genre_weight", 0.60),
            tag_weight=data.get("tag_weight", 0.90),
            pacing_weight=data.get("pacing_weight", 0.75),
            tone_weight=data.get("tone_weight", 0.40),
            era_weight=data.get("era_weight", 0.30),
            language_weight=data.get("language_weight", 0.50),
            rating_log=data.get("rating_log", []),
            watch_history=set(data.get("watch_history", [])),
            highly_rated=set(data.get("highly_rated", [])),
            poorly_rated=set(data.get("poorly_rated", [])),
            watchlist=set(data.get("watchlist", [])),
            abandoned=set(data.get("abandoned", [])),
            rewatched=set(data.get("rewatched", [])),
            disliked_actors=data.get("disliked_actors", []),
            disliked_directors=data.get("disliked_directors", []),
            dealbreakers=data.get("dealbreakers", []),
            genre_affinity=data.get("genre_affinity", {}),
            tag_affinity=data.get("tag_affinity", {}),
            tone_affinity=data.get("tone_affinity", {}),
            pacing_affinity=data.get("pacing_affinity", {}),
            actor_affinity=data.get("actor_affinity", {}),
            director_affinity=data.get("director_affinity", {}),
            era_affinity=data.get("era_affinity", {}),
            language_affinity=data.get("language_affinity", {}),
            country_affinity=data.get("country_affinity", {}),
            content_rating_affinity=data.get("content_rating_affinity", {}),
            runtime_preference=data.get("runtime_preference", "any"),
            franchise_tolerance=data.get("franchise_tolerance", 0.5),
            query_history=data.get("query_history", []),
            disabled_signals=data.get("disabled_signals", []),
            last_session_ts=data.get("last_session_ts"),
            session_count=data.get("session_count", 0),
            avg_queries_per_session=data.get("avg_queries_per_session", 0.0),
            recency_halflife_days=data.get("recency_halflife_days", 90),
            taste_vector_confidence=data.get("taste_vector_confidence", 0.0),
            taste_vector_updated_at=data.get("taste_vector_updated_at"),
            genre_confidence=data.get("genre_confidence", {}),
            director_confidence=data.get("director_confidence", {}),
            actor_confidence=data.get("actor_confidence", {}),
            correction_log=data.get("correction_log", []),
            diversity_appetite=data.get("diversity_appetite", 0.5),
            exploration_rate=data.get("exploration_rate", 0.5),
            signal_weights=data.get("signal_weights", {
                "watch_history": "balanced",
                "ratings": "balanced",
                "reviews": "balanced",
            }),
            memory_entries=data.get("memory_entries", []),
        )
