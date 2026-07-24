#!/usr/bin/env python3
"""
interface/controller.py — Central System Controller & Session Manager for CineVault

Pre-warms and coordinates all underlying engines:
  • Recommendation Pipeline (nlp/pipeline.py)
  • User Profile Store (user_profile/store.py)
  • Local Query Understanding Layer (nlp/qul.py)
  • Dual-Path LLM Review Extractor (user_profile/review_processor.py)

Acts as the single point of entry for the Terminal User Interface (TUI).
"""

import asyncio
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.pipeline import CineVaultPipeline
from user_profile.review_processor import LLMReviewProcessor
from user_profile.store import UserProfileStore
from user_profile.schema import UserProfile

logger = logging.getLogger("cinevault.controller")


class CineVaultController:
    """
    Central Orchestrator & Session Manager for CineVault.
    """

    def __init__(self, user_id: str = "default_user", auto_prewarm: bool = True):
        t0 = time.time()
        logger.info(f"Initializing CineVault Controller for user_id='{user_id}'...")
        self.user_id = user_id
        
        # 1. Profile Store & Active User Session State
        self.store = UserProfileStore()
        self.profile: UserProfile = self.store.load_profile(self.user_id)

        # 2. Interactive Session Preferences
        self.lambda_personalization: float = 0.5   # 0.0 pure profile <-> 1.0 pure query
        self.exclude_watched: bool = True           # Watch history filter toggle
        self.active_search_results: List[Dict[str, Any]] = []
        self.active_inspected_movie: Optional[Dict[str, Any]] = None
        self._query_cache: Dict[str, List[Dict[str, Any]]] = {}

        # 3. Pre-warmed Subsystems
        self.pipeline: Optional[CineVaultPipeline] = None
        self.review_processor: Optional[LLMReviewProcessor] = None

        if auto_prewarm:
            self.prewarm_engines()

        t1 = time.time()
        logger.info(f"CineVault Controller initialized in {t1 - t0:.2f}s.")

    def prewarm_engines(self):
        """Pre-loads heavy recommendation pipelines and models into memory."""
        logger.info("Pre-warming CineVault Recommendation Engine & Subsystems...")
        self.pipeline = CineVaultPipeline()
        self.review_processor = LLMReviewProcessor(
            tag_vocabulary=self.pipeline.retriever.tag_vocab if hasattr(self.pipeline.retriever, "tag_vocab") else None
        )
        logger.info("CineVault Subsystems successfully pre-warmed.")

    def search(self, query_str: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Executes a personalized query search with caching and returns top_k results.
        """
        if not self.pipeline:
            self.prewarm_engines()

        cache_key = f"{self.user_id}:{query_str.strip().lower()}:{top_k}:{self.lambda_personalization}:{self.exclude_watched}"
        if cache_key in self._query_cache:
            logger.info(f"Query cache hit for key='{cache_key}'")
            results = self._query_cache[cache_key]
            self.active_search_results = results
            return results

        logger.info(f"Controller executing search query: '{query_str}' (λ={self.lambda_personalization:.2f}, exclude_watched={self.exclude_watched})")
        raw_res = self.pipeline.recommend(
            query=query_str,
            user_id=self.user_id,
            top_k=top_k,
            personalization_lambda=self.lambda_personalization,
            include_watched=not self.exclude_watched
        )
        if isinstance(raw_res, dict):
            results = raw_res.get("results", raw_res.get("recommendations", []))
        else:
            results = raw_res

        self._query_cache[cache_key] = results
        self.active_search_results = results
        return results

    def inspect_movie(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves full hydrated metadata card for the specified movie ID.
        """
        # First check active cached search results
        for item in self.active_search_results:
            if item.get("movie_id") == movie_id:
                self.active_inspected_movie = item
                return item

        # Hydrate directly from Hydrator if not in current results
        if self.pipeline and self.pipeline.hydrator:
            hydrated = self.pipeline.hydrator.hydrate([{"movie_id": movie_id}])
            if hydrated:
                self.active_inspected_movie = hydrated[0]
                return hydrated[0]

        return None

    def submit_review(
        self,
        movie_id: int,
        star_rating: float,
        review_text: str = "",
        surgical_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Processes a user movie rating/review through Path A surgical checkboxes,
        LLM extraction (Primary), and Path B local fallback. Persists profile update to SQLite.
        """
        if not self.review_processor:
            self.prewarm_engines()

        movie_card = self.inspect_movie(movie_id) or {"movie_id": movie_id, "title": f"Movie #{movie_id}"}

        result = self.review_processor.process_review(
            profile=self.profile,
            movie_card=movie_card,
            star_rating=star_rating,
            review_text=review_text,
            surgical_aspects=surgical_aspects
        )

        # Save updated user profile to local SQLite
        self.store.save_profile(self.profile)
        logger.info(f"Persisted updated UserProfile for user_id='{self.user_id}' after review submission.")

        # Extract tags for database integration
        extracted_tags = list(result.get("extracted_fuzzy_tags", []))
        if "llm_output" in result and isinstance(result["llm_output"], dict):
            extracted_tags.extend(result["llm_output"].get("extracted_tags", []))
        if surgical_aspects:
            extracted_tags.extend(surgical_aspects)

        # Integrate review into db/cinevault.db catalog tables
        self._integrate_review_into_db(
            movie_id=movie_id,
            star_rating=star_rating,
            review_text=review_text,
            extracted_tags=list(set(extracted_tags))
        )

        return result

    def _integrate_review_into_db(
        self,
        movie_id: int,
        star_rating: float,
        review_text: str,
        extracted_tags: List[str]
    ):
        """
        Integrates submitted rating, review text, and extracted tags into db/cinevault.db
        tables (ratings, reviews, user_tags) for continuous catalog learning.
        """
        try:
            conn = sqlite3.connect(self.store.db_path)
            c = conn.cursor()
            now_ts = int(time.time())

            try:
                uid_num = int(self.user_id)
            except ValueError:
                uid_num = abs(hash(self.user_id)) % 1000000

            c.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, rated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    rating = excluded.rating,
                    rated_at = excluded.rated_at
            """, (uid_num, movie_id, star_rating, now_ts))

            if review_text.strip():
                c.execute("""
                    INSERT INTO reviews (movie_id, source, domain, review_text, score, review_date)
                    VALUES (?, 'cinevault_user', 'audience', ?, ?, CURRENT_TIMESTAMP)
                """, (movie_id, review_text.strip(), star_rating))

            for tag in extracted_tags:
                c.execute("""
                    INSERT INTO user_tags (user_id, movie_id, tag, tagged_at)
                    VALUES (?, ?, ?, ?)
                """, (uid_num, movie_id, tag.lower(), now_ts))

            conn.commit()
            conn.close()
            logger.info(f"Integrated review for movie_id={movie_id} into db/cinevault.db catalog tables.")
        except Exception as e:
            logger.error(f"Failed to integrate review into db/cinevault.db: {e}")

    def set_lambda(self, val: float):
        """Sets personalization weight λ (0.0 = pure user profile, 1.0 = pure query)."""
        self.lambda_personalization = max(0.0, min(1.0, float(val)))
        logger.info(f"Personalization λ updated to {self.lambda_personalization:.2f}")

    def set_exclude_watched(self, enabled: bool):
        """Sets watch history exclusion filter status."""
        self.exclude_watched = bool(enabled)
        logger.info(f"Watch history filter updated to exclude_watched={self.exclude_watched}")

    def update_profile_weights(self, weights: Dict[str, float]):
        """
        Updates user sensitivity weights (director, actor, genre, tag, pacing).
        """
        if "director_weight" in weights:
            self.profile.director_weight = float(weights["director_weight"])
        if "actor_weight" in weights:
            self.profile.actor_weight = float(weights["actor_weight"])
        if "genre_weight" in weights:
            self.profile.genre_weight = float(weights["genre_weight"])
        if "tag_weight" in weights:
            self.profile.tag_weight = float(weights["tag_weight"])
        if "pacing_weight" in weights:
            self.profile.pacing_weight = float(weights["pacing_weight"])

        self.store.save_profile(self.profile)
        logger.info(f"Updated profile sensitivity weights for user_id='{self.user_id}'")

    async def search_async(self, query_str: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Asynchronously executes search on a background worker thread to prevent freezing Textual UI.
        """
        return await asyncio.to_thread(self.search, query_str, top_k)

    def seed_cold_start(
        self,
        favorite_genres: List[str],
        anchor_movie_ids: List[int],
        dealbreakers: Optional[List[str]] = None
    ) -> UserProfile:
        """
        Executes 30-Second Cold-Start Onboarding seeding:
          1. Boosts favorite_genres affinities.
          2. Extracts metadata & genome tags from anchor_movie_ids to seed taste centroids.
          3. Applies negative affinities for dealbreaker tags/genres.
        """
        logger.info(f"Seeding cold-start profile for user_id='{self.user_id}'...")
        
        # 1. Favorite Genres
        for g in favorite_genres:
            g_clean = g.strip()
            self.profile.genre_affinity[g_clean] = self.profile.genre_affinity.get(g_clean, 0.0) + 0.8

        # 2. Anchor Movies Metadata Extraction
        if anchor_movie_ids:
            if not self.pipeline:
                self.prewarm_engines()
            
            cards = self.pipeline.hydrator.hydrate([{"movie_id": mid} for mid in anchor_movie_ids])
            for card in cards:
                self.profile.apply_rating_update(card, star_rating=5.0, review_confidence=1.0)

        # 3. Dealbreaker Rules
        if dealbreakers:
            for db in dealbreakers:
                db_clean = db.strip().lower()
                if "no " in db_clean:
                    db_clean = db_clean.replace("no ", "")
                self.profile.tag_affinity[db_clean] = -1.0
                self.profile.genre_affinity[db_clean.capitalize()] = -1.0

        self.store.save_profile(self.profile)
        logger.info(f"Cold-start onboarding complete for user_id='{self.user_id}'. Profile saved.")
        return self.profile

    def list_users(self) -> List[str]:
        """Returns list of all registered user IDs in local SQLite database."""
        return self.store.list_users()

    def switch_user(self, user_id: str) -> UserProfile:
        """Switches current active session user profile."""
        self.user_id = user_id
        self.profile = self.store.load_profile(user_id)
        logger.info(f"Switched active session to user_id='{self.user_id}'")
        return self.profile

    def format_inspector_markdown(self, card: Dict[str, Any]) -> str:
        """
        Formats a hydrated movie card into rich Markdown for the TUI IMDb Inspector Modal.
        """
        title = card.get("title", "Unknown Title")
        year = card.get("year", "N/A")
        rating_str = f"★ {card.get('avg_rating', 0.0):.2f}" if card.get("avg_rating") else "★ Unrated"
        count_str = f"({card.get('num_ratings', 0):,} ratings)" if card.get("num_ratings") else ""
        genres = ", ".join(card.get("genres", [])) or "N/A"
        directors = ", ".join(card.get("directors", [])) or "N/A"
        cast = ", ".join(card.get("actors", card.get("cast", []))[:6]) or "N/A"

        md_lines = [
            f"# {title} ({year})",
            f"**Rating**: {rating_str} {count_str} | **Tier**: `{card.get('tier', 'Tier C')}`",
            f"**Genres**: {genres}",
            f"**Directors**: {directors}",
            f"**Starring**: {cast}",
            "---",
        ]

        if card.get("overview"):
            md_lines.append(f"### Plot Overview\n{card['overview']}\n")
        if card.get("themes"):
            md_lines.append(f"**Themes**: {', '.join(card['themes'])}")
        if card.get("tone"):
            md_lines.append(f"**Tone**: {', '.join(card['tone'])}")
        if card.get("pacing"):
            md_lines.append(f"**Pacing**: {card['pacing']}")
        if card.get("comparable_films"):
            md_lines.append(f"**Similar Titles**: {', '.join(card['comparable_films'][:5])}")

        return "\n".join(md_lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ctrl = CineVaultController(user_id="controller_test_user")

    print("\n--- Testing Search via Controller ---")
    results = ctrl.search("atmospheric slow burn murder mystery", top_k=3)
    for r in results:
        print(f"  #{r.get('final_rank')} {r.get('title')} ({r.get('year')}) — Rating: {r.get('avg_rating')}")

    if results:
        top_movie_id = results[0]["movie_id"]
        print(f"\n--- Testing Review Submission via Controller for Movie ID {top_movie_id} ---")
        review_res = ctrl.submit_review(
            movie_id=top_movie_id,
            star_rating=5.0,
            review_text="Absolute masterpiece with incredible visual storytelling and slow burn tension.",
            surgical_aspects=["Visuals", "Plot", "Pacing"]
        )
        print("Review Submission Outcome:", json.dumps(review_res, indent=2))
        print("Updated User Profile Tag Affinities:", ctrl.profile.tag_affinity)
