"""interface/controller.py — Session controller for CineVault."""

import asyncio
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

from nlp.pipeline import CineVaultPipeline
from user_profile.review_processor import LLMReviewProcessor
from user_profile.store import ProfileConflictError, UserProfileStore
from user_profile.schema import UserPreset, UserProfile

logger = logging.getLogger("cinevault.controller")


class CineVaultController:

    def __init__(self, user_id="default_user", auto_prewarm=True):
        t0 = time.time()
        self.user_id = user_id

        self.store = UserProfileStore()
        self.profile: UserProfile = self.store.load_profile(self.user_id)

        self.lambda_personalization = 0.7
        self.exclude_watched = True
        self.active_search_results = []
        self.active_inspected_movie = None
        self._query_cache = {}

        self.pipeline = None
        self.review_processor = None

        # Preset overlay — if one was active, restore it
        self.active_preset: UserPreset | None = None
        active = self.store.load_active_preset(self.user_id)
        if active:
            self.active_preset = active
            self.lambda_personalization = active.lambda_val
            # Apply signal toggles
            self.profile.disabled_signals = [
                sig for sig, on in active.signals.items() if not on
            ]

        if not self.store.user_exists(self.user_id):
            self.store.save_profile(self.profile)

        if auto_prewarm:
            self.prewarm_engines()

        logger.info(f"Controller ready for {user_id!r} in {time.time() - t0:.2f}s.")

    def prewarm_engines(self):
        self.pipeline = CineVaultPipeline()
        self.review_processor = LLMReviewProcessor(
            tag_vocabulary=list(self.pipeline.retriever.tag_to_idx.keys()) if hasattr(self.pipeline.retriever, "tag_to_idx") else None
        )

    def _save_profile_safe(self, *, snapshot=False):
        """Save with optimistic-concurrency conflict handling."""
        try:
            self.store.save_profile(self.profile, snapshot=snapshot)
        except ProfileConflictError:
            logger.warning(f"Profile conflict for {self.user_id!r} — reloading.")
            try:
                fresh = self.store.load_profile(self.user_id)
                for attr in (
                    "genre_affinity", "tag_affinity", "tone_affinity",
                    "pacing_affinity", "actor_affinity", "director_affinity",
                    "era_affinity", "language_affinity", "country_affinity",
                    "content_rating_affinity", "genre_confidence",
                    "director_confidence", "actor_confidence",
                ):
                    merged = {**getattr(fresh, attr), **getattr(self.profile, attr)}
                    setattr(fresh, attr, merged)
                for attr in ("watch_history", "highly_rated", "poorly_rated",
                             "watchlist", "abandoned", "rewatched"):
                    setattr(fresh, attr,
                            getattr(fresh, attr) | getattr(self.profile, attr))
                existing_ids = {e["movie_id"] for e in fresh.rating_log}
                for entry in self.profile.rating_log:
                    if entry["movie_id"] not in existing_ids:
                        fresh.rating_log.append(entry)
                fresh.query_history = (
                    fresh.query_history + self.profile.query_history
                )[-50:]
                self.profile = fresh
                self.store.save_profile(self.profile, snapshot=snapshot)
            except ProfileConflictError:
                logger.warning(f"Conflict retry also failed for {self.user_id!r}. Changes lost.")

    def search(self, query_str, top_k=10):
        if not self.pipeline:
            self.prewarm_engines()

        cache_key = f"{self.user_id}:{query_str.strip().lower()}:{top_k}:{self.lambda_personalization}:{self.exclude_watched}"
        if cache_key in self._query_cache:
            self.active_search_results = self._query_cache[cache_key]
            return self._query_cache[cache_key]

        raw = self.pipeline.recommend(
            query=query_str,
            user_id=self.user_id,
            top_k=top_k,
            personalization_lambda=self.lambda_personalization,
            include_watched=not self.exclude_watched
        )
        results = raw.get("results", raw.get("recommendations", []))

        self._query_cache[cache_key] = results
        self.active_search_results = results

        top_id = results[0].get("movie_id") if results else None
        self.profile.record_query(query_str, result_count=len(results), top_result_id=top_id)
        self._save_profile_safe()

        return results

    def inspect_movie(self, movie_id):
        for item in self.active_search_results:
            if item.get("movie_id") == movie_id:
                self.active_inspected_movie = item
                return item

        if self.pipeline and self.pipeline.hydrator:
            hydrated = self.pipeline.hydrator.hydrate([{"movie_id": movie_id}])
            if hydrated:
                self.active_inspected_movie = hydrated[0]
                return hydrated[0]
        return None

    def submit_review(self, movie_id, star_rating, review_text="", surgical_aspects=None):
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

        self._save_profile_safe(snapshot=True)

        extracted_tags = list(result.get("extracted_fuzzy_tags", []))
        if "llm_output" in result and isinstance(result["llm_output"], dict):
            extracted_tags.extend(result["llm_output"].get("extracted_tags", []))
        if surgical_aspects:
            extracted_tags.extend(surgical_aspects)

        self._integrate_review_into_db(
            movie_id=movie_id,
            star_rating=star_rating,
            review_text=review_text,
            extracted_tags=list(set(extracted_tags))
        )
        return result

    def _integrate_review_into_db(self, movie_id, star_rating, review_text, extracted_tags):
        try:
            conn = sqlite3.connect(self.store.db_path)
            c = conn.cursor()
            now_ts = int(time.time())

            c.execute("""
                INSERT INTO ratings (user_id, movie_id, rating, rated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, movie_id) DO UPDATE SET
                    rating   = excluded.rating,
                    rated_at = excluded.rated_at
            """, (self.user_id, movie_id, star_rating, now_ts))

            if review_text.strip():
                c.execute("""
                    INSERT INTO reviews
                        (movie_id, source, domain, review_text, score, review_date, user_id)
                    VALUES (?, 'cinevault_user', 'audience', ?, ?, CURRENT_TIMESTAMP, ?)
                """, (movie_id, review_text.strip(), star_rating, self.user_id))

            for tag in extracted_tags:
                c.execute("""
                    INSERT INTO user_tags (user_id, movie_id, tag, tagged_at)
                    VALUES (?, ?, ?, ?)
                """, (self.user_id, movie_id, tag.lower(), now_ts))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to integrate review into db: {e}")

    def set_lambda(self, val):
        self.lambda_personalization = max(0.0, min(1.0, float(val)))

    def set_exclude_watched(self, enabled):
        self.exclude_watched = bool(enabled)

    def update_profile_weights(self, weights):
        for key in ("director_weight", "actor_weight", "genre_weight", "tag_weight", "pacing_weight"):
            if key in weights:
                setattr(self.profile, key, float(weights[key]))
        self._save_profile_safe()

    async def search_async(self, query_str, top_k=10):
        return await asyncio.to_thread(self.search, query_str, top_k)

    def seed_cold_start(self, favorite_genres, anchor_movie_ids, dealbreakers=None,
                        preferred_eras=None, preferred_languages=None, runtime_preference="any"):
        for g in favorite_genres:
            g_clean = g.strip()
            self.profile.genre_affinity[g_clean] = self.profile.genre_affinity.get(g_clean, 0.0) + 0.8
            self.profile.genre_confidence[g_clean] = self.profile.genre_confidence.get(g_clean, 0) + 3

        if anchor_movie_ids:
            if not self.pipeline:
                self.prewarm_engines()
            cards = self.pipeline.hydrator.hydrate([{"movie_id": mid} for mid in anchor_movie_ids])
            for card in cards:
                self.profile.apply_rating_update(card, star_rating=5.0, review_confidence=1.0)

                anchor_tags = []
                for t in (card.get("top_tags") or [])[:5]:
                    tag_str = t if isinstance(t, str) else (t.get("tag", "") if isinstance(t, dict) else "")
                    if tag_str.strip():
                        anchor_tags.append(tag_str.strip())
                self._integrate_review_into_db(
                    movie_id=int(card["movie_id"]),
                    star_rating=5.0,
                    review_text="",
                    extracted_tags=anchor_tags,
                )

        if dealbreakers:
            self.profile.dealbreakers = list(set(self.profile.dealbreakers + [
                re.sub(r"^no\s+", "", d.strip().lower()) for d in dealbreakers if d.strip()
            ]))

        if preferred_eras:
            for era in preferred_eras:
                era_clean = era.strip()
                self.profile.era_affinity[era_clean] = self.profile.era_affinity.get(era_clean, 0.0) + 0.8

        if preferred_languages:
            for lang in preferred_languages:
                lc = lang.strip().lower()
                self.profile.language_affinity[lc] = self.profile.language_affinity.get(lc, 0.0) + 0.8

        if runtime_preference in ("short", "standard", "epic", "any"):
            self.profile.runtime_preference = runtime_preference

        self._save_profile_safe()
        return self.profile

    # TUI calls these — they handle save-after-mutate

    def add_to_watchlist(self, movie_id):
        self.profile.add_to_watchlist(movie_id)
        self._save_profile_safe()

    def mark_abandoned(self, movie_id):
        self.profile.mark_abandoned(movie_id)
        self._save_profile_safe()

    def log_correction(self, movie_id, title, reason):
        self.profile.log_correction(movie_id, title, reason)
        self._save_profile_safe()

    def clear_query_history(self):
        self.profile.clear_query_history()
        self._save_profile_safe()

    def set_signal_enabled(self, signal_name, enabled):
        if not enabled:
            if signal_name not in self.profile.disabled_signals:
                self.profile.disabled_signals.append(signal_name)
        else:
            if signal_name in self.profile.disabled_signals:
                self.profile.disabled_signals.remove(signal_name)
        self._save_profile_safe()

    def get_signal_weights(self):
        return dict(self.profile.signal_weights)

    def set_signal_weight(self, signal_name, level):
        if level not in ("off", "light", "balanced", "strong"):
            return
        self.profile.signal_weights[signal_name] = level
        self._save_profile_safe()

    def get_memory_entries(self):
        return list(self.profile.memory_entries)

    def add_memory_entry(self, entry):
        if entry.strip() and entry.strip() not in self.profile.memory_entries:
            self.profile.memory_entries.append(entry.strip())
            self._save_profile_safe()

    def delete_memory_entry(self, index):
        if 0 <= index < len(self.profile.memory_entries):
            self.profile.memory_entries.pop(index)
            self._save_profile_safe()

    def clear_all_memory(self):
        self.profile.memory_entries.clear()
        self._save_profile_safe()

    def list_users(self):
        return self.store.list_users()

    def switch_user(self, user_id):
        self.user_id = user_id
        self.profile = self.store.load_profile(user_id)
        self._query_cache.clear()
        # Load the new user's active preset
        self.active_preset = self.store.load_active_preset(user_id)
        if self.active_preset:
            self.lambda_personalization = self.active_preset.lambda_val
            self.profile.disabled_signals = [
                sig for sig, on in self.active_preset.signals.items() if not on
            ]
        else:
            self.lambda_personalization = 0.7
            self.profile.disabled_signals = []
        return self.profile

    # ── preset management ──

    @property
    def active_preset_name(self):
        """Name of the active preset, or 'Default' if none."""
        return self.active_preset.name if self.active_preset else "Default"

    @property
    def lambda_label(self):
        """Human-readable label for the current λ value."""
        return UserPreset.lambda_to_label(self.lambda_personalization)

    def create_preset(self, name, lambda_val=0.7, signals=None):
        preset = self.store.create_preset(self.user_id, name, lambda_val, signals)
        return preset

    def list_presets(self):
        return self.store.list_presets(self.user_id)

    def activate_preset(self, name):
        """Activate a named preset — updates λ and signal toggles in-session."""
        self.store.activate_preset(self.user_id, name)
        preset = self.store.load_active_preset(self.user_id)
        if preset:
            self.active_preset = preset
            self.lambda_personalization = preset.lambda_val
            self.profile.disabled_signals = [
                sig for sig, on in preset.signals.items() if not on
            ]

    def deactivate_preset(self):
        """Clear active preset — revert to profile defaults."""
        self.store.deactivate_preset(self.user_id)
        self.active_preset = None
        self.lambda_personalization = 0.7
        self.profile.disabled_signals = []

    def delete_preset(self, name):
        was_active = self.active_preset and self.active_preset.name == name
        self.store.delete_preset(self.user_id, name)
        if was_active:
            self.active_preset = None
            self.lambda_personalization = 0.7
            self.profile.disabled_signals = []

    def update_preset(self, name, lambda_val, signals):
        self.store.update_preset(self.user_id, name, lambda_val, signals)
        # If this preset is currently active, refresh in-session values
        if self.active_preset and self.active_preset.name == name:
            self.active_preset.lambda_val = lambda_val
            self.active_preset.signals = signals
            self.lambda_personalization = lambda_val
            self.profile.disabled_signals = [
                sig for sig, on in signals.items() if not on
            ]

    def get_user_summaries(self):
        """Return list of user summary dicts for the profile switcher."""
        user_ids = self.store.list_users()
        return [self.store.get_user_summary(uid) for uid in user_ids]

    def format_inspector_markdown(self, card):
        title = card.get("title", "Unknown Title")
        year = card.get("year", "N/A")
        rating_str = f"★ {card.get('avg_rating', 0.0):.2f}" if card.get("avg_rating") else "★ Unrated"
        count_str = f"({card.get('num_ratings', 0):,} ratings)" if card.get("num_ratings") else ""
        genres = ", ".join(card.get("genres", [])) or "N/A"
        directors = ", ".join(card.get("directors", [])) or "N/A"
        cast = ", ".join(card.get("actors", card.get("cast", []))[:6]) or "N/A"

        md = [
            f"# {title} ({year})",
            f"**Rating**: {rating_str} {count_str} | **Tier**: `{card.get('tier', 'Tier C')}`",
            f"**Genres**: {genres}",
            f"**Directors**: {directors}",
            f"**Starring**: {cast}",
            "---",
        ]

        if card.get("overview"):
            md.append(f"### Plot Overview\n{card['overview']}\n")
        if card.get("themes"):
            md.append(f"**Themes**: {', '.join(card['themes'])}")
        if card.get("tone"):
            md.append(f"**Tone**: {', '.join(card['tone'])}")
        if card.get("pacing"):
            md.append(f"**Pacing**: {card['pacing']}")
        if card.get("comparable_films"):
            md.append(f"**Similar Titles**: {', '.join(card['comparable_films'][:5])}")

        return "\n".join(md)
