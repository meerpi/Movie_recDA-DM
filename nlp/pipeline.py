"""nlp/pipeline.py — End-to-end recommendation pipeline."""

import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from nlp.hydrator import ResultHydrator
from nlp.mmr import MaximalMarginalRelevance
from nlp.qul import QueryUnderstandingLayer
from nlp.reranker import CineVaultReranker
from nlp.retriever import CineVaultRetriever
from nlp.router import QueryRouter
from user_profile.schema import UserProfile
from user_profile.store import UserProfileStore

logger = logging.getLogger("cinevault.pipeline")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
load_dotenv(PROJECT_ROOT / ".env")


def normalize_profile_boost(s_prof):
    """Sigmoid norm: raw profile boost → (0, 1), centered at 0.5."""
    val = float(s_prof) / 2.0
    if val >= 0:
        return 1.0 / (1.0 + math.exp(-val))
    z = math.exp(val)
    return z / (1.0 + z)


class CineVaultPipeline:

    def __init__(self, use_qwen_4bit=False, load_dense=True, db_path=DB_PATH,
                 lazy_load_models=False):
        t0 = time.time()
        logger.info("Initializing CineVault Pipeline...")

        self.db_path = db_path
        self.router = QueryRouter(db_path=db_path)
        self.hydrator = ResultHydrator(db_path=db_path)
        self.qul = QueryUnderstandingLayer(shared_cards=self.hydrator.cards)
        self.retriever = CineVaultRetriever(load_dense=load_dense)
        self.mmr = MaximalMarginalRelevance(diversity_lambda=0.75, max_per_franchise=2)
        self.profile_store = UserProfileStore(db_path=db_path)

        self.use_qwen_4bit = use_qwen_4bit
        self._reranker: Optional[CineVaultReranker] = None
        if not lazy_load_models:
            self._reranker = CineVaultReranker(use_qwen_4bit=use_qwen_4bit)

        logger.info(f"Pipeline initialized in {time.time() - t0:.2f}s.")

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CineVaultReranker(use_qwen_4bit=self.use_qwen_4bit)
        return self._reranker

    def recommend(self, query, user_id="default_user", top_k=10,
                  personalization_lambda=0.7, include_watched=False,
                  use_qul=True, candidates_k=250, use_voyage=True):
        t0 = time.time()
        query_str = query.strip()

        # fast-path: association rules / genre lists
        fast_match = self.router.match_deterministic(query_str)
        if fast_match and fast_match.get("type") in ("association_rules", "top_movies", "genre_top", "overall_top"):
            raw_items = fast_match.get("results") or []
            to_hydrate = [{"movie_id": item["movie_id"]} for item in raw_items]
            hydrated = self.hydrator.hydrate(to_hydrate)

            raw_scores = {}
            for idx, r in enumerate(raw_items):
                score = r.get("lift") or r.get("confidence") or (1.0 / (idx + 1))
                raw_scores[r["movie_id"]] = float(score)

            min_s = min(raw_scores.values()) if raw_scores else 0.0
            max_s = max(raw_scores.values()) if raw_scores else 1.0
            score_range = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

            profile = self.profile_store.load_profile(user_id)
            raw_prof = [profile.calculate_profile_boost(c) for c in hydrated]

            ranked = []
            for c, s_prof in zip(hydrated, raw_prof):
                mid = c["movie_id"]
                cand = dict(c)

                if mid in profile.watch_history and not include_watched:
                    cand["final_score"] = -10.0
                    cand["profile_boost"] = -10.0
                else:
                    base = raw_scores.get(mid, 0.0)
                    norm_base = (base - min_s) / score_range
                    norm_prof = normalize_profile_boost(s_prof)
                    s_final = (personalization_lambda * norm_base) + ((1.0 - personalization_lambda) * norm_prof)
                    cand["final_score"] = round(s_final, 4)
                    cand["profile_boost"] = round(s_prof, 4)

                ranked.append(cand)

            ranked.sort(key=lambda x: x.get("final_score", -999.0), reverse=True)

            final = self.mmr.filter_diverse(ranked, top_k=top_k)
            for i, item in enumerate(final, 1):
                item["final_rank"] = i

            return {
                "user_id": user_id,
                "query": query_str,
                "expanded_query": query_str,
                "routing_path": f"deterministic_fast_path ({fast_match.get('type')})",
                "personalization_lambda": personalization_lambda,
                "total_candidates_retrieved": len(raw_items),
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "results": final,
            }

        # QUL query expansion
        expanded       = query_str
        intent_summary = "general_search"
        detected_demonym  = None
        matched_genres    = []
        is_obscure_intent = False
        genre_strictness  = "hard_exclude"
        bm25_keywords     = []
        if use_qul:
            try:
                qul_res = self.qul.parse_query(query_str)
                if qul_res and "expanded_query" in qul_res:
                    expanded       = qul_res["expanded_query"]
                    intent_summary = qul_res.get("intent_type", "general_search")
                    detected_demonym  = qul_res.get("detected_demonym")
                    matched_genres    = qul_res.get("matched_genres") or []
                    is_obscure_intent = bool(qul_res.get("is_obscure_intent", False))
                    genre_strictness  = qul_res.get("genre_strictness", "hard_exclude")
                    bm25_keywords     = qul_res.get("bm25_keywords") or []
            except Exception as e:
                logger.warning(f"QUL expansion failed, using raw query: {e}")

        # multi-lane RRF retrieval
        bm25_q = " ".join(bm25_keywords) if bm25_keywords else None
        search_hits = self.retriever.search(
            expanded, top_k=candidates_k, use_voyage=use_voyage, bm25_query=bm25_q
        )
        if not search_hits:
            return {
                "user_id": user_id, "query": query_str, "expanded_query": expanded,
                "routing_path": "semantic_pipeline",
                "personalization_lambda": personalization_lambda,
                "total_candidates_retrieved": 0,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "results": [],
            }

        hydrated = self.hydrator.hydrate(search_hits)

        # cross-encoder reranking — try Voyage API first, fall back to local BGE
        reranked = []
        voyage_key = os.environ.get("VOYAGE_API_KEY", "").strip()
        if use_voyage and voyage_key:
            try:
                import voyageai
                v_client = voyageai.Client(api_key=voyage_key)

                passages = []
                for item in hydrated:
                    parts = [f"{item.get('title','')} ({item.get('year','')})"]
                    if item.get("directors"): parts.append(f"Director: {', '.join(item['directors'][:2])}")
                    if item.get("actors"): parts.append(f"Starring: {', '.join(item['actors'][:4])}")
                    if item.get("genres"): parts.append(f"Genres: {', '.join(item['genres'][:3])}")
                    if item.get("themes"): parts.append(f"Themes: {', '.join(item['themes'][:3])}")
                    if item.get("overview"): parts.append(f"Plot: {item['overview'][:150]}")
                    passages.append(" | ".join(parts))

                v_res = v_client.rerank(
                    query=expanded,
                    documents=passages,
                    model="rerank-2.5",
                    top_k=len(passages),
                )
                for r in v_res.results:
                    cand = dict(hydrated[r.index])
                    cand["rerank_score"] = float(r.relevance_score)
                    reranked.append(cand)

                # anything not returned by the API gets the batch minimum
                if reranked:
                    min_rerank = min(c["rerank_score"] for c in reranked)
                    seen_mids = {c["movie_id"] for c in reranked}
                    for c in hydrated:
                        if c["movie_id"] not in seen_mids:
                            overflow = dict(c)
                            overflow["rerank_score"] = min_rerank * 0.5
                            reranked.append(overflow)

            except Exception as ve:
                logger.warning(f"Voyage rerank failed ({ve}), falling back to local BGE.")
                reranked = []

        if not reranked:
            reranked = self.reranker.rerank(
                query=expanded, candidates=hydrated, top_k=len(hydrated),
            )

        # personalization + quality score fusion
        profile = self.profile_store.load_profile(user_id)

        raw_rerank = [c.get("rerank_score", 0.0) for c in reranked]
        raw_prof = [profile.calculate_profile_boost(c) for c in reranked]

        q_lower = query_str.lower()

        is_obscure_query = is_obscure_intent or any(
            w in q_lower
            for w in (
                "obscure", "indie", "hidden gem", "cult", "rare",
                "underrated", "under the radar", "unknown", "b-movie",
                "b movie", "b-tier", "b tier", "niche", "b-grade", "trash", "campy"
            )
        )

        _ERA_TERMS = (
            "classic", "classics", "old movie", "old film", "vintage", "retro",
            "golden age", "black and white", "b&w", "film noir",
            "40s", "50s", "60s", "70s", "80s", "90s",
            "1940s", "1950s", "1960s", "1970s", "1980s", "1990s",
            "pre-2000", "twentieth century", "old school", "oldschool",
        )
        is_era_explicit = any(term in q_lower for term in _ERA_TERMS)

        for c, s_rr, s_prof in zip(reranked, raw_rerank, raw_prof):
            mid = c["movie_id"]

            if mid in profile.watch_history and not include_watched:
                c["final_score"] = -10.0
                c["profile_boost"] = -10.0
            else:
                val_rr = float(s_rr)
                if c.get("rerank_score") is not None and 0.0 <= val_rr <= 1.0:
                    norm_rr = val_rr  # Voyage: already [0,1]
                elif c.get("rerank_score") is not None:
                    # BGE/GGUF: unbounded logit → sigmoid
                    if val_rr >= 0:
                        norm_rr = 1.0 / (1.0 + math.exp(-val_rr))
                    else:
                        z = math.exp(val_rr)
                        norm_rr = z / (1.0 + z)
                else:
                    norm_rr = 0.01  # never scored, penalize hard

                norm_prof = normalize_profile_boost(s_prof)

                raw_rating = float(c.get("avg_rating", 3.0)) if c.get("avg_rating") is not None else 3.0
                num_ratings = float(c.get("num_ratings", 0))

                bayes_rating = (num_ratings * raw_rating + 100.0 * 3.2) / (num_ratings + 100.0)
                log_v_boost = min(1.0, math.log10(num_ratings + 1.0) / 2.5)
                norm_rating = 0.70 * max(0.0, min(1.0, (bayes_rating - 1.0) / 4.0)) + 0.30 * log_v_boost

                relevance = (0.88 * norm_rr) + (0.12 * norm_rating)
                base_score = (personalization_lambda * relevance) + ((1.0 - personalization_lambda) * norm_prof)

                if is_obscure_query or raw_rating >= 3.5:
                    w_rating = 1.0
                elif raw_rating >= 2.0:
                    w_rating = 0.75 + 0.25 * ((raw_rating - 2.0) / 1.5)
                else:
                    norm_low = max(0.0, (raw_rating - 0.5) / 1.5)
                    w_rating = 0.20 + 0.55 * (norm_low ** 1.5)

                s_final = base_score * w_rating

                # genre mismatch penalty
                if matched_genres and genre_strictness != "unspecified":
                    cand_genres = {g.strip().lower() for g in (c.get("genres") or []) if isinstance(g, str)}
                    target_genres = {g.strip().lower() for g in (matched_genres or []) if isinstance(g, str)}
                    if not (cand_genres & target_genres):
                        if genre_strictness == "hard_exclude":
                            s_final *= 0.3
                        elif genre_strictness == "soft_preference":
                            s_final *= 0.75

                # tone contradiction (e.g. "serious dark" query vs slapstick comedy)
                tone_list = [t.lower() for t in (c.get("tone") or []) if isinstance(t, str)]
                genre_list = [g.lower() for g in (c.get("genres") or []) if isinstance(g, str)]

                if any(w in q_lower for w in ("serious", "dark", "grim", "gritty")):
                    if any(t in tone_str for tone_str in tone_list for t in ("slapstick", "farcical", "goofy", "silly", "spoof")):
                        s_final -= 0.5
                    elif "comedy" in genre_list and not ({"drama", "thriller", "mystery", "crime"} & set(genre_list)):
                        s_final -= 0.3

                # creature/monster subject relevance
                creature_terms = (
                    "monster", "monsters", "creature", "creatures",
                    "vampire", "vampires", "zombie", "zombies",
                    "werewolf", "werewolves", "alien", "aliens",
                    "ghost", "ghosts", "demon", "demons"
                )
                if any(re.search(r"\b" + re.escape(term) + r"\b", q_lower) for term in creature_terms):
                    tags_list = [
                        (t.lower() if isinstance(t, str) else t.get("tag", "").lower())
                        for t in (c.get("top_tags") or []) if t is not None
                    ]
                    themes_list = [t.lower() for t in (c.get("themes") or []) if isinstance(t, str)]
                    all_text = " ".join(genre_list + tags_list + themes_list)
                    has_creature = any(re.search(r"\b" + re.escape(w) + r"\b", all_text) for w in creature_terms) or any(g in ("horror", "sci-fi", "fantasy") for g in genre_list)
                    if not has_creature:
                        s_final *= 0.3

                # demonym / national origin boost
                if detected_demonym:
                    target_lang = detected_demonym.get("lang")
                    target_country = detected_demonym.get("country")
                    item_lang = str(c.get("original_language", "")).lower()
                    item_countries = [str(cnt).lower() for cnt in (c.get("production_countries") or []) if cnt]
                    is_match = (
                        (target_lang and item_lang == target_lang) or
                        (target_country and any(target_country in cnt for cnt in item_countries))
                    )
                    if is_match:
                        s_final *= 1.85

                # recency multiplier — gently prefer newer. dataset ends 2019.
                # disabled for era-explicit queries ("classic", "80s", etc.)
                movie_year = c.get("year")
                if movie_year and not is_era_explicit:
                    age = max(0, 2019 - int(movie_year))
                    recency_mult = 0.75 + 0.25 / (1.0 + (age / 25.0) ** 1.5)

                    if profile.era_affinity:
                        era = profile.get_era_from_year(int(movie_year))
                        era_pref = profile.era_affinity.get(era, 0.0)
                        if era_pref > 0:
                            recency_mult = max(recency_mult, 0.95)

                    # top ~1% classics (bayes >= 4.2) get a pass
                    if bayes_rating >= 4.2:
                        recency_mult = max(recency_mult, 0.92)

                    s_final *= recency_mult

                c["final_score"] = round(s_final, 4)
                c["profile_boost"] = round(s_prof, 4)
                c["norm_rerank_score"] = round(norm_rr, 4)

        reranked.sort(key=lambda x: x.get("final_score", -999.0), reverse=True)

        final = self.mmr.filter_diverse(reranked, top_k=top_k)
        for i, item in enumerate(final, 1):
            item["final_rank"] = i

        return {
            "user_id": user_id,
            "query": query_str,
            "expanded_query": expanded,
            "intent": intent_summary,
            "routing_path": "semantic_pipeline",
            "personalization_lambda": personalization_lambda,
            "total_candidates_retrieved": len(search_hits),
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "results": final,
        }
