#!/usr/bin/env python3
"""
nlp/pipeline.py — Step 14: Unified End-to-End CineVault Pipeline

Unifies all 5 stages of CineVault into a single production entry point:
  1. Router: Fast-path deterministic association rules / genre top lists.
  2. QUL: Gemini Flash natural language query expansion & intent parsing.
  3. Retriever: Multi-lane RRF search (BM25 + Genome HNSW + Dense HNSW).
  4. Hydrator: Database ratings/metadata & Tier A/B/C card enrichment.
  5. Reranker: Cross-Encoder joint semantic scoring (BAAI / Qwen 4-bit).
  6. Personalization: User profile affinity score fusion (λ-dial) & watch history filtering.
  7. MMR: Maximal Marginal Relevance diversity filter & franchise capping (max 2).

Also handles live user feedback & reviews (`submit_review`).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp.hydrator import ResultHydrator
from nlp.mmr import MaximalMarginalRelevance
from nlp.qul import QueryUnderstandingLayer
from nlp.reranker import CineVaultReranker
from nlp.retriever import CineVaultRetriever
from nlp.router import QueryRouter
from user_profile.review_processor import NonLLMReviewProcessor
from user_profile.schema import UserProfile
from user_profile.store import UserProfileStore

logger = logging.getLogger("cinevault.pipeline")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"


class CineVaultPipeline:

    def __init__(
        self,
        use_qwen_4bit: bool = False,
        load_dense: bool = True,
        db_path: Path = DB_PATH,
        lazy_load_models: bool = False,
    ):
        """
        Initializes and holds singletons of all pipeline components.
        """
        t0 = time.time()
        logger.info("Initializing CineVault Unified Pipeline...")

        self.db_path = db_path
        self.router = QueryRouter(db_path=db_path)
        self.qul = QueryUnderstandingLayer()
        self.retriever = CineVaultRetriever(load_dense=load_dense)
        self.hydrator = ResultHydrator(db_path=db_path)
        self.mmr = MaximalMarginalRelevance(diversity_lambda=0.75, max_per_franchise=2)
        self.profile_store = UserProfileStore(db_path=db_path)

        # Review processor vocabulary from retriever's genome tags
        tag_vocab = list(self.retriever.tag_to_idx.keys()) if hasattr(self.retriever, "tag_to_idx") else []
        self.review_processor = NonLLMReviewProcessor(tag_vocabulary=tag_vocab)

        self.lazy_load_models = lazy_load_models
        self.use_qwen_4bit = use_qwen_4bit
        self._reranker: Optional[CineVaultReranker] = None

        if not lazy_load_models:
            self._reranker = CineVaultReranker(use_qwen_4bit=use_qwen_4bit)

        t1 = time.time()
        logger.info(f"CineVault Pipeline initialized in {t1 - t0:.2f}s.")

    @property
    def reranker(self) -> CineVaultReranker:
        """Lazy loads reranker model if required."""
        if self._reranker is None:
            self._reranker = CineVaultReranker(use_qwen_4bit=self.use_qwen_4bit)
        return self._reranker

    def recommend(
        self,
        query: str,
        user_id: str = "default_user",
        top_k: int = 10,
        personalization_lambda: float = 0.7,
        include_watched: bool = False,
        use_qul: bool = True,
        candidates_k: int = 35,
        use_voyage: bool = True,
    ) -> Dict[str, Any]:
        """
        Main recommendation entry point.

        Returns structured recommendation dictionary containing metadata,
        routing details, expanded query, and top-K hydrated movie items.
        """
        t0 = time.time()
        query_str = query.strip()

        # ── 1. Fast-Path Router Check ───────────────────────────────────────
        fast_match = self.router.match_deterministic(query_str)
        if fast_match and fast_match.get("type") in ("association_rules", "top_movies", "genre_top", "overall_top"):
            raw_items = fast_match.get("results", [])
            to_hydrate = [{"movie_id": item["movie_id"]} for item in raw_items]
            hydrated = self.hydrator.hydrate(to_hydrate)

            # Map raw scores (e.g. lift/confidence or rank)
            raw_scores = {}
            for idx, r in enumerate(raw_items):
                score = r.get("lift") or r.get("confidence") or (1.0 / (idx + 1))
                raw_scores[r["movie_id"]] = float(score)

            min_score = min(raw_scores.values()) if raw_scores else 0.0
            max_score = max(raw_scores.values()) if raw_scores else 1.0
            score_range = (max_score - min_score) if (max_score - min_score) > 1e-6 else 1.0

            # Load user profile for watch history filtering and personalization
            profile = self.profile_store.load_profile(user_id)
            raw_prof_scores = [profile.calculate_profile_boost(c) for c in hydrated]
            min_prof = min(s for s in raw_prof_scores if s >= 0.0) if any(s >= 0.0 for s in raw_prof_scores) else 0.0
            max_prof = max(raw_prof_scores) if raw_prof_scores else 1.0
            prof_range = (max_prof - min_prof) if (max_prof - min_prof) > 1e-6 else 1.0

            personalized_candidates = []
            for c, s_prof in zip(hydrated, raw_prof_scores):
                mid = c["movie_id"]
                cand = dict(c)

                if mid in profile.watch_history and not include_watched:
                    cand["final_score"] = -10.0
                    cand["profile_boost"] = -10.0
                else:
                    base = raw_scores.get(mid, 0.0)
                    norm_base = (base - min_score) / score_range
                    norm_prof = (s_prof - min_prof) / prof_range if s_prof >= 0.0 else 0.0

                    s_final = (personalization_lambda * norm_base) + ((1.0 - personalization_lambda) * norm_prof)
                    cand["final_score"] = round(s_final, 4)
                    cand["profile_boost"] = round(s_prof, 4)

                personalized_candidates.append(cand)

            # Sort personalized candidates by final_score descending
            personalized_candidates.sort(key=lambda x: x.get("final_score", -999.0), reverse=True)

            # Exclude watched movies and tone-contradiction items
            personalized_candidates = [c for c in personalized_candidates if c.get("final_score", -999.0) >= 0.0]

            # Pass through MMR diversity filter & franchise capping
            final_results = self.mmr.filter_diverse(personalized_candidates, top_k=top_k)

            return {
                "user_id": user_id,
                "query": query_str,
                "expanded_query": query_str,
                "routing_path": f"deterministic_fast_path ({fast_match.get('type')})",
                "personalization_lambda": personalization_lambda,
                "total_candidates_retrieved": len(raw_items),
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "results": final_results,
            }

        # ── 2. Query Understanding (QUL Expansion) ──────────────────────────
        expanded_query = query_str
        intent_summary = "search"
        if use_qul:
            try:
                qul_result = self.qul.parse_query(query_str)
                if qul_result and "expanded_query" in qul_result:
                    expanded_query = qul_result["expanded_query"]
                    intent_summary = qul_result.get("intent", "search")
            except Exception as e:
                logger.warning(f"QUL expansion failed, continuing with raw query: {e}")

        # ── 3. Multi-Lane RRF Retrieval ──────────────────────────────────────
        search_hits = self.retriever.search(expanded_query, top_k=candidates_k, use_voyage=use_voyage)
        if not search_hits:
            return {
                "user_id": user_id,
                "query": query_str,
                "expanded_query": expanded_query,
                "routing_path": "semantic_pipeline",
                "personalization_lambda": personalization_lambda,
                "total_candidates_retrieved": 0,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "results": [],
            }

        # ── 4. Candidate Hydration ─────────────────────────────────────────
        hydrated_candidates = self.hydrator.hydrate(search_hits)

        # ── 5. Two-Stage Fast GPU Cross-Encoder Reranking ──────────────────
        # Pre-sort by RRF score and take top 25 candidates for GPU cross-encoding
        to_rerank = hydrated_candidates[:25]
        remaining = hydrated_candidates[25:]

        reranked_top = self.reranker.rerank(
            query=expanded_query,
            candidates=to_rerank,
            top_k=len(to_rerank),
        )
        for r_item in remaining:
            r_item["rerank_score"] = -1.0
            r_item["rrf_rank"] = search_hits[hydrated_candidates.index(r_item)].get("rrf_rank", 99) if isinstance(search_hits, list) else 99

        reranked_candidates = reranked_top + remaining

        # ── 6. Personalization Score Fusion ─────────────────────────────────
        profile = self.profile_store.load_profile(user_id)

        # Extract reranker raw scores
        raw_rerank_scores = [c.get("rerank_score", 0.0) for c in reranked_candidates]
        min_rr = min(raw_rerank_scores) if raw_rerank_scores else 0.0
        max_rr = max(raw_rerank_scores) if raw_rerank_scores else 1.0
        rr_range = (max_rr - min_rr) if (max_rr - min_rr) > 1e-6 else 1.0

        # Calculate raw profile boosts
        raw_prof_scores = [profile.calculate_profile_boost(c) for c in reranked_candidates]
        min_prof = min(s for s in raw_prof_scores if s >= 0.0) if any(s >= 0.0 for s in raw_prof_scores) else 0.0
        max_prof = max(raw_prof_scores) if raw_prof_scores else 1.0
        prof_range = (max_prof - min_prof) if (max_prof - min_prof) > 1e-6 else 1.0

        q_lower = query_str.lower()
        for c, s_rr, s_prof in zip(reranked_candidates, raw_rerank_scores, raw_prof_scores):
            mid = c["movie_id"]

            if mid in profile.watch_history and not include_watched:
                c["final_score"] = -10.0
                c["profile_boost"] = -10.0
            else:
                # Normalize both components to [0.0, 1.0] scale
                norm_rr = (s_rr - min_rr) / rr_range
                norm_prof = (s_prof - min_prof) / prof_range if s_prof >= 0.0 else 0.0

                s_final = (personalization_lambda * norm_rr) + ((1.0 - personalization_lambda) * norm_prof)

                # Apply Tone Contradiction Penalty (e.g. serious/dark query vs slapstick comedy tone)
                tone_list = [t.lower() for t in c.get("tone", [])]
                genre_list = [g.lower() for g in c.get("genres", [])]

                if any(w in q_lower for w in ("serious", "dark", "grim", "gritty")):
                    if any(t in tone_str for tone_str in tone_list for t in ("slapstick", "farcical", "goofy", "silly", "spoof")):
                        s_final -= 0.5
                    elif "comedy" in genre_list and not ({"drama", "thriller", "mystery", "crime"} & set(genre_list)):
                        s_final -= 0.3

                c["final_score"] = round(s_final, 4)
                c["profile_boost"] = round(s_prof, 4)
                c["norm_rerank_score"] = round(norm_rr, 4)

        # Sort candidates by personalized final_score descending
        personalized_candidates = sorted(
            reranked_candidates,
            key=lambda x: x.get("final_score", -999.0),
            reverse=True,
        )

        # Filter out watched items and tone-contradiction items
        personalized_candidates = [c for c in personalized_candidates if c.get("final_score", -999.0) >= 0.0]

        # ── 7. MMR Diversity Filter & Franchise Capping ────────────────────
        final_results = self.mmr.filter_diverse(personalized_candidates, top_k=top_k)
        for rank_idx, item in enumerate(final_results, 1):
            item["final_rank"] = rank_idx

        latency_ms = round((time.time() - t0) * 1000, 2)

        return {
            "user_id": user_id,
            "query": query_str,
            "expanded_query": expanded_query,
            "intent": intent_summary,
            "routing_path": "semantic_pipeline",
            "personalization_lambda": personalization_lambda,
            "total_candidates_retrieved": len(search_hits),
            "latency_ms": latency_ms,
            "results": final_results,
        }

    def submit_review(
        self,
        user_id: str,
        movie_id: int,
        review_text: str,
        star_rating: float,
    ) -> Dict[str, Any]:
        """
        Submits a user rating and review text.
        Processes sentiment, updates user profile affinities, and persists to SQLite.
        """
        # Fetch hydrated card details for the movie
        hydrated = self.hydrator.hydrate([{"movie_id": movie_id}])
        movie_card = hydrated[0] if hydrated else {"movie_id": movie_id}

        # Load user profile
        profile = self.profile_store.load_profile(user_id)

        # Process review using NonLLMReviewProcessor
        analysis = self.review_processor.process_review(
            profile=profile,
            movie_card=movie_card,
            review_text=review_text,
            star_rating=star_rating,
        )

        # Persist updated profile back to SQLite user_profiles table
        self.profile_store.save_profile(profile)

        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "movie_title": movie_card.get("title", f"Movie #{movie_id}"),
            "analysis": analysis,
            "updated_profile_summary": profile.to_dict(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n--- Testing CineVault Pipeline Initialization ---")
    pipeline = CineVaultPipeline(load_dense=True)

    test_query = "atmospheric slow burn Korean thriller"
    print(f"\n--- Testing Semantic Search: '{test_query}' ---")
    res = pipeline.recommend(test_query, user_id="test_user", top_k=5)

    print(f"\nRouting Path: {res['routing_path']}")
    print(f"Expanded Query: {res['expanded_query']}")
    print(f"Latency: {res['latency_ms']} ms")
    print(f"Retrieved Candidates: {res['total_candidates_retrieved']}")
    print("\nTop Recommendations:")
    for i, item in enumerate(res["results"], 1):
        print(f"  {i}. {item['title']} ({item.get('year', 'N/A')}) - Tier: {item['tier']} | Score: {item.get('final_score')}")

    fast_query = "movies like Titanic"
    print(f"\n--- Testing Fast-Path Routing: '{fast_query}' ---")
    res_fast = pipeline.recommend(fast_query, user_id="test_user", top_k=5)
    print(f"Routing Path: {res_fast['routing_path']}")
    print(f"Latency: {res_fast['latency_ms']} ms")
    print("Fast-Path Top Co-Likes:")
    for i, item in enumerate(res_fast["results"], 1):
        print(f"  {i}. {item['title']} ({item.get('year', 'N/A')})")
