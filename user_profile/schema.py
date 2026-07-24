#!/usr/bin/env python3
"""
profile/schema.py — Step 10: User Profile Schema & Personalization Scorer

Defines the UserProfile data class holding multidimensional taste signals:
  • People: actor_affinity, director_affinity
  • Qualitative: genre_affinity, tag_affinity, tone_affinity, pacing_affinity
  • Semantic Taste Vectors: dense_taste_vector (1024d), genome_taste_vector (1128d)
  • History: watch_history, highly_rated, poorly_rated
  • Personalization Score: profile_boost(movie_item)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np


@dataclass
class UserProfile:
    user_id: str = "default_user"

    # People signals
    actor_affinity: Dict[str, float] = field(default_factory=dict)
    director_affinity: Dict[str, float] = field(default_factory=dict)
    disliked_actors: List[str] = field(default_factory=list)
    disliked_directors: List[str] = field(default_factory=list)

    # Content signals
    genre_affinity: Dict[str, float] = field(default_factory=dict)
    tag_affinity: Dict[str, float] = field(default_factory=dict)
    tone_affinity: Dict[str, float] = field(default_factory=dict)
    pacing_affinity: Dict[str, float] = field(default_factory=dict)

    # History sets
    watch_history: Set[int] = field(default_factory=set)
    highly_rated: Set[int] = field(default_factory=set)
    poorly_rated: Set[int] = field(default_factory=set)

    # Dense taste vectors (optional 1024d and 1128d centroids)
    dense_taste_vector: Optional[np.ndarray] = None
    genome_taste_vector: Optional[np.ndarray] = None

    # Configurable personalization dial (lambda: 1.0 = pure query, 0.0 = pure profile)
    personalization_lambda: float = 0.7

    # User-controllable sensitivity weights
    director_weight: float = 0.80
    actor_weight: float = 0.50
    genre_weight: float = 0.60
    tag_weight: float = 0.90
    pacing_weight: float = 0.75

    def calculate_profile_boost(self, movie_item: Dict[str, Any]) -> float:
        """
        Calculates a personalized boost score (0.0 to ~3.0+) for a movie item
        based on the user's affinity signals.
        """
        mid = movie_item.get("movie_id")
        if mid in self.watch_history:
            return -10.0  # Hard penalize already-seen movies

        score = 0.0

        # 1. Actor affinity boost
        cast = movie_item.get("actors", []) or movie_item.get("cast", [])
        for actor in cast:
            score += self.actor_affinity.get(actor, 0.0) * self.actor_weight

        # 2. Director affinity boost
        directors = movie_item.get("directors", [])
        if not directors and movie_item.get("director"):
            directors = [movie_item.get("director")]
        for director in directors:
            if director in self.director_affinity:
                score += self.director_affinity[director] * self.director_weight

        # 3. Genre affinity boost
        genres = movie_item.get("genres", [])
        for genre in genres:
            score += self.genre_affinity.get(genre, 0.0) * self.genre_weight

        # 4. Tone & Theme boost
        tone = movie_item.get("tone", [])
        for t in tone:
            score += self.tone_affinity.get(t, 0.0) * 0.4

        # 5. Tag affinity boost
        tags = movie_item.get("top_tags", [])
        for tag_item in tags:
            tag_name = tag_item if isinstance(tag_item, str) else tag_item.get("tag", "")
            if tag_name:
                score += self.tag_affinity.get(tag_name, 0.0) * self.tag_weight

        # 6. Pacing boost
        pacing = movie_item.get("pacing")
        if pacing:
            score += self.pacing_affinity.get(pacing, 0.0) * self.pacing_weight

        return max(0.0, score)

    def apply_rating_update(
        self,
        movie_item: Dict[str, Any],
        star_rating: float,
        review_confidence: float = 1.0
    ):
        """
        Updates user profile affinities based on a movie rating (1.0 to 5.0 stars).
        """
        mid = movie_item.get("movie_id")
        if mid:
            self.watch_history.add(mid)
            if star_rating >= 4.0:
                self.highly_rated.add(mid)
            elif star_rating <= 2.0:
                self.poorly_rated.add(mid)

        # Normalized rating weight (-1.0 for 1★ to +1.0 for 5★)
        weight = (star_rating - 3.0) / 2.0 * review_confidence

        # Update genres
        for g in movie_item.get("genres", []):
            self.genre_affinity[g] = self.genre_affinity.get(g, 0.0) + weight * self.genre_weight

        # Update tone
        for t in movie_item.get("tone", []):
            self.tone_affinity[t] = self.tone_affinity.get(t, 0.0) + weight * 0.4

        # Update tags
        for tag_item in movie_item.get("top_tags", []):
            tag_name = tag_item if isinstance(tag_item, str) else tag_item.get("tag", "")
            if tag_name:
                self.tag_affinity[tag_name] = self.tag_affinity.get(tag_name, 0.0) + weight * self.tag_weight

        # Update pacing
        pacing = movie_item.get("pacing")
        if pacing:
            self.pacing_affinity[pacing] = self.pacing_affinity.get(pacing, 0.0) + weight * self.pacing_weight

    def to_dict(self) -> Dict[str, Any]:
        """Serializes user profile to JSON-compatible dict."""
        return {
            "user_id": self.user_id,
            "actor_affinity": self.actor_affinity,
            "director_affinity": self.director_affinity,
            "disliked_actors": self.disliked_actors,
            "disliked_directors": self.disliked_directors,
            "genre_affinity": self.genre_affinity,
            "tag_affinity": self.tag_affinity,
            "tone_affinity": self.tone_affinity,
            "pacing_affinity": self.pacing_affinity,
            "watch_history": list(self.watch_history),
            "highly_rated": list(self.highly_rated),
            "poorly_rated": list(self.poorly_rated),
            "personalization_lambda": self.personalization_lambda,
            "director_weight": self.director_weight,
            "actor_weight": self.actor_weight,
            "genre_weight": self.genre_weight,
            "tag_weight": self.tag_weight,
            "pacing_weight": self.pacing_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Deserializes user profile from JSON dict."""
        profile = cls(
            user_id=data.get("user_id", "default_user"),
            actor_affinity=data.get("actor_affinity", {}),
            director_affinity=data.get("director_affinity", {}),
            disliked_actors=data.get("disliked_actors", []),
            disliked_directors=data.get("disliked_directors", []),
            genre_affinity=data.get("genre_affinity", {}),
            tag_affinity=data.get("tag_affinity", {}),
            tone_affinity=data.get("tone_affinity", {}),
            pacing_affinity=data.get("pacing_affinity", {}),
            watch_history=set(data.get("watch_history", [])),
            highly_rated=set(data.get("highly_rated", [])),
            poorly_rated=set(data.get("poorly_rated", [])),
            personalization_lambda=data.get("personalization_lambda", 0.7),
            director_weight=data.get("director_weight", 0.80),
            actor_weight=data.get("actor_weight", 0.50),
            genre_weight=data.get("genre_weight", 0.60),
            tag_weight=data.get("tag_weight", 0.90),
            pacing_weight=data.get("pacing_weight", 0.75),
        )
        return profile


if __name__ == "__main__":
    profile = UserProfile(user_id="rdj_fan")
    profile.actor_affinity["Robert Downey Jr."] = 0.95
    profile.genre_affinity["Action"] = 0.9
    profile.genre_affinity["Sci-Fi"] = 0.85

    card_ironman = {
        "movie_id": 1721,
        "title": "Iron Man",
        "cast": ["Robert Downey Jr.", "Gwyneth Paltrow"],
        "genres": ["Action", "Sci-Fi"],
    }
    card_generic = {
        "movie_id": 9999,
        "title": "Generic Drama",
        "cast": ["Unknown"],
        "genres": ["Drama"],
    }

    boost_ironman = profile.calculate_profile_boost(card_ironman)
    boost_generic = profile.calculate_profile_boost(card_generic)

    print(f"Iron Man boost score: {boost_ironman:.4f}")
    print(f"Generic Drama boost score: {boost_generic:.4f}")
