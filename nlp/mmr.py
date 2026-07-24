#!/usr/bin/env python3
"""
nlp/mmr.py — Step 12: Maximal Marginal Relevance (MMR) Diversity Filter

Prevents franchise domination in top recommendation results (e.g. 4 Annabelle or
Jurassic Park entries occupying top positions).

Balances relevance score with diversity penalty:
    MMR_Score(c) = λ * RerankScore(c) - (1 - λ) * MaxSimilarity(c, Selected)

Franchise deduplication priority:
    1. Primary key: TMDb `collection` field (e.g. "Jurassic Park Collection")
       — catches cross-title franchises like the Conjuring Universe.
    2. Fallback: title-stem regex matching for films without a collection field.

Usage:
    from nlp.mmr import MaximalMarginalRelevance

    mmr = MaximalMarginalRelevance(diversity_lambda=0.7)
    diversified = mmr.filter_diverse(reranked_results, top_k=10)
"""

import re
from typing import Any, Dict, List

_FRANCHISE_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "on", "at", "to",
    "movie", "part", "chapter", "creation", "origins", "collection",
}


class MaximalMarginalRelevance:

    def __init__(self, diversity_lambda: float = 0.75, max_per_franchise: int = 2):
        """
        diversity_lambda: 1.0 = pure relevance, 0.0 = max diversity.
        max_per_franchise: Hard limit on entries from the same franchise per result set.
        """
        self.diversity_lambda = diversity_lambda
        self.max_per_franchise = max_per_franchise

    def _get_franchise_key(self, item: Dict[str, Any]) -> str:
        """
        Returns the canonical franchise dedup key for an item.

        Priority order:
          1. TMDb collection name (normalized) — most reliable for cross-title franchises
             (e.g. "The Conjuring" and "Annabelle" share the same TMDb collection but no
             common title tokens, so stem matching would miss this entirely).
          2. Title stem regex — fallback for films without a collection field.
        """
        collection = item.get("collection")
        if collection:
            # Normalize: lowercase, strip "collection" suffix, strip stopwords
            normalized = collection.strip().lower()
            normalized = re.sub(r"\bcollection\b", "", normalized).strip()
            tokens = [t for t in normalized.split() if t not in _FRANCHISE_STOPWORDS]
            key = " ".join(tokens) if tokens else normalized
            return f"coll:{key}"

        # Fallback: title-stem matching
        return f"stem:{self._extract_title_stem(item.get('title', ''))}"

    def _extract_title_stem(self, title: str) -> str:
        """Extracts core franchise stem from a title string (fallback when no collection)."""
        # Strip sub-titles after colon, dash, or digits (e.g. "Annabelle: Creation" -> "annabelle")
        base = re.split(r"[:\-\d]", title)[0].strip().lower()
        tokens = [t for t in base.split() if t not in _FRANCHISE_STOPWORDS]
        return " ".join(tokens) if tokens else base

    def compute_similarity(self, item_a: Dict[str, Any], item_b: Dict[str, Any]) -> float:
        """Calculates similarity between two movie items (0.0 to 1.0)."""
        key_a = self._get_franchise_key(item_a)
        key_b = self._get_franchise_key(item_b)

        # Same TMDb collection → very high similarity
        if key_a.startswith("coll:") and key_b.startswith("coll:") and key_a == key_b:
            return 0.97

        # Same title stem → high similarity (title-stem fallback path)
        stem_a = re.sub(r"^(coll:|stem:)", "", key_a)
        stem_b = re.sub(r"^(coll:|stem:)", "", key_b)
        if stem_a and stem_b and (stem_a in stem_b or stem_b in stem_a):
            return 0.90

        # Genre overlap (low-weight secondary signal)
        genres_a = set(item_a.get("genres", []))
        genres_b = set(item_b.get("genres", []))
        if not genres_a or not genres_b:
            return 0.0
        jaccard_genre = len(genres_a & genres_b) / float(len(genres_a | genres_b))
        return jaccard_genre * 0.3

    def filter_diverse(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Selects top_k candidates from reranked list while penalizing franchise duplicates.
        Uses TMDb `collection` as the primary franchise key, falling back to title-stem.
        """
        if not candidates:
            return []

        selected: List[Dict[str, Any]] = []
        unselected = list(candidates)
        franchise_counts: Dict[str, int] = {}

        # Normalize relevance scores to [0, 1] for MMR math
        scores = [c.get("rerank_score", c.get("rrf_score", 0.0)) for c in unselected]
        min_s, max_s = min(scores), max(scores)
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        def norm_score(c):
            raw = c.get("rerank_score", c.get("rrf_score", 0.0))
            return (raw - min_s) / range_s

        while unselected and len(selected) < top_k:
            if not selected:
                # Always pick the top-ranked candidate first
                best_item = unselected.pop(0)
                selected.append(best_item)
                fkey = self._get_franchise_key(best_item)
                franchise_counts[fkey] = franchise_counts.get(fkey, 0) + 1
                continue

            best_mmr = -999.0
            best_idx = 0

            for idx, candidate in enumerate(unselected):
                fkey = self._get_franchise_key(candidate)

                # Hard franchise cap
                if franchise_counts.get(fkey, 0) >= self.max_per_franchise:
                    continue

                relevance = norm_score(candidate)
                max_sim = max(self.compute_similarity(candidate, s) for s in selected)
                mmr_val = (self.diversity_lambda * relevance) - ((1.0 - self.diversity_lambda) * max_sim)

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = idx

            if best_mmr == -999.0:
                # All remaining candidates hit the franchise cap → relax and pick top
                best_item = unselected.pop(0)
            else:
                best_item = unselected.pop(best_idx)

            selected.append(best_item)
            fkey = self._get_franchise_key(best_item)
            franchise_counts[fkey] = franchise_counts.get(fkey, 0) + 1

        return selected


if __name__ == "__main__":
    mmr = MaximalMarginalRelevance(diversity_lambda=0.75, max_per_franchise=2)

    # Test 1: Collection-based dedup — Conjuring Universe shares NO title tokens
    conjuring_candidates = [
        {"title": "The Conjuring", "collection": "The Conjuring Collection", "genres": ["Horror"], "rerank_score": 4.89},
        {"title": "Annabelle", "collection": "The Conjuring Collection", "genres": ["Horror"], "rerank_score": 4.56},
        {"title": "Annabelle: Creation", "collection": "The Conjuring Collection", "genres": ["Horror"], "rerank_score": 4.30},
        {"title": "The Boy", "collection": None, "genres": ["Horror", "Thriller"], "rerank_score": 3.80},
        {"title": "Ouija: Origin of Evil", "collection": None, "genres": ["Horror"], "rerank_score": 3.50},
    ]
    filtered = mmr.filter_diverse(conjuring_candidates, top_k=4)
    print("MMR (Collection-based) Top 4:")
    for i, item in enumerate(filtered, 1):
        coll = item.get("collection") or "None"
        print(f"  #{i} {item['title']} | Collection: {coll} | Score: {item['rerank_score']}")
    print()

    # Test 2: Title-stem fallback (no collection field)
    stem_candidates = [
        {"title": "Toy Story", "collection": None, "genres": ["Animation"], "rerank_score": 4.89},
        {"title": "Toy Story 2", "collection": None, "genres": ["Animation"], "rerank_score": 4.60},
        {"title": "Toy Story 3", "collection": None, "genres": ["Animation"], "rerank_score": 4.30},
        {"title": "Shrek", "collection": None, "genres": ["Animation"], "rerank_score": 3.80},
    ]
    filtered2 = mmr.filter_diverse(stem_candidates, top_k=3)
    print("MMR (Stem-based fallback) Top 3:")
    for i, item in enumerate(filtered2, 1):
        print(f"  #{i} {item['title']} | Score: {item['rerank_score']}")
