#!/usr/bin/env python3
"""
nlp/retriever.py — Step 5: Multi-Lane RRF Search Retriever Engine

Combines 3 search lanes via Reciprocal Rank Fusion (RRF):
  • Lane 1 (BM25)       : Keyword search over Tier A profile card text
  • Lane 2 (Genome HNSW): Tag-genome ANN search over Tier A + B movies (13,816)
  • Lane 3 (Dense HNSW) : Semantic ANN search via Voyage-4-Large over Tier A movies (9,526)

RRF Scoring:
  score(movie_id) = Σ  1.0 / (60 + rank_i + 1)
                   lanes

Usage:
    from nlp.retriever import CineVaultRetriever

    retriever = CineVaultRetriever()
    results = retriever.search("atmospheric slow burn Korean thriller", top_k=10)

CLI test:
    .venv/bin/python nlp/retriever.py "atmospheric slow burn Korean thriller"
"""

import csv
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import hnswlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT     = Path(__file__).parent.parent
BM25_INDEX_PATH  = PROJECT_ROOT / "nlp" / "bm25_index.pkl"
BM25_MAP_PATH    = PROJECT_ROOT / "nlp" / "bm25_id_map.json"
GENOME_HNSW_PATH = PROJECT_ROOT / "nlp" / "genome.hnsw"
GENOME_MAP_PATH  = PROJECT_ROOT / "nlp" / "genome_id_map.json"
GENOME_TAGS_PATH = PROJECT_ROOT / "data" / "ml-25m" / "genome-tags.csv"
DENSE_HNSW_PATH  = PROJECT_ROOT / "nlp" / "dense.hnsw"
DENSE_MAP_PATH   = PROJECT_ROOT / "nlp" / "dense_id_map.json"

VOYAGE_MODEL     = "voyage-4-large"
RRF_K            = 60
DEFAULT_LANE_K   = 100

_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


class CineVaultRetriever:
    """
    Core search engine combining BM25, Genome HNSW, and Dense Voyage HNSW.
    """

    def __init__(
        self,
        voyage_api_key: str | None = None,
        load_dense: bool = True,
    ) -> None:
        t0 = time.time()
        print("Initializing CineVault Retriever ...")

        # ── 1. Load Lane 1: BM25 ─────────────────────────────────────
        if not BM25_INDEX_PATH.exists():
            raise FileNotFoundError(f"[ERROR] {BM25_INDEX_PATH} not found. Run Step 1 first.")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25       = bm25_data["bm25"]
        self.bm25_corpus = bm25_data["corpus"]
        self.bm25_id_map = bm25_data["id_map"]
        print(f"  ✓ Lane 1 (BM25) loaded: {len(self.bm25_id_map):,} docs")

        # ── 2. Load Lane 2: Genome HNSW ──────────────────────────────
        if not GENOME_HNSW_PATH.exists():
            raise FileNotFoundError(f"[ERROR] {GENOME_HNSW_PATH} not found. Run Step 2 first.")
        with open(GENOME_MAP_PATH, encoding="utf-8") as f:
            self.genome_id_map = json.load(f)

        self.genome_dim = 1128
        self.genome_index = hnswlib.Index(space="cosine", dim=self.genome_dim)
        self.genome_index.load_index(str(GENOME_HNSW_PATH), max_elements=len(self.genome_id_map))
        self.genome_index.set_ef(100)
        print(f"  ✓ Lane 2 (Genome HNSW) loaded: {len(self.genome_id_map):,} vectors")

        # Load genome tag dictionary (tag -> index 0..1127)
        self.tag_to_idx: dict[str, int] = {}
        if GENOME_TAGS_PATH.exists():
            with open(GENOME_TAGS_PATH, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_name = row["tag"].strip().lower()
                    tag_id   = int(row["tagId"]) - 1  # 1-indexed -> 0-indexed
                    self.tag_to_idx[tag_name] = tag_id
        print(f"  ✓ Genome vocabulary loaded: {len(self.tag_to_idx):,} tags")

        # ── 3. Load Lane 3: Dense HNSW ───────────────────────────────
        self.dense_index = None
        self.dense_id_map = []
        self.voyage_client = None

        if load_dense and DENSE_HNSW_PATH.exists():
            with open(DENSE_MAP_PATH, encoding="utf-8") as f:
                self.dense_id_map = json.load(f)
            self.dense_dim = 1024
            self.dense_index = hnswlib.Index(space="cosine", dim=self.dense_dim)
            self.dense_index.load_index(str(DENSE_HNSW_PATH), max_elements=len(self.dense_id_map))
            self.dense_index.set_ef(100)

            # Check API key
            api_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY", "").strip() or "pa-EF1FgBNa-qvVdqEZstrRq4iyD2uq7KorI7EPaL8v5wn"
            if api_key:
                try:
                    import voyageai
                    self.voyage_client = voyageai.Client(api_key=api_key)
                    print(f"  ✓ Lane 3 (Dense HNSW + Voyage AI) loaded: {len(self.dense_id_map):,} vectors")
                except Exception as e:
                    print(f"  [WARN] Failed to initialize Voyage AI client ({e}). Lane 3 disabled.")
            else:
                print("  [INFO] VOYAGE_API_KEY not set. Using local BM25 + Genome HNSW (0 Cloud API Cost).")

        print(f"Retriever initialized in {time.time() - t0:.2f}s.\n")

    # -----------------------------------------------------------------------
    # Query vector building for Lane 2 (Genome HNSW)
    # -----------------------------------------------------------------------
    def build_genome_query_vector(self, query: str) -> np.ndarray:
        """
        Build a 1,128-dim sparse normalized query vector from query text by matching tags.
        Matches exact tag names as well as individual token matches.
        """
        vec = np.zeros(self.genome_dim, dtype=np.float32)
        q_lower = query.lower().strip()
        q_tokens = set(tokenize(query))

        # Check full query match
        if q_lower in self.tag_to_idx:
            vec[self.tag_to_idx[q_lower]] += 3.0

        # Check sub-phrases & single token matches
        for tag, idx in self.tag_to_idx.items():
            if tag in q_lower:
                vec[idx] += 2.0
            else:
                # token overlap
                tag_tokens = set(tokenize(tag))
                overlap = len(q_tokens & tag_tokens)
                if overlap > 0:
                    vec[idx] += 0.5 * overlap

        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        return vec

    # -----------------------------------------------------------------------
    # Multi-Lane Search
    # -----------------------------------------------------------------------
    def search_lane1_bm25(self, query: str, top_k: int = DEFAULT_LANE_K) -> list[tuple[int, float]]:
        """Run BM25 search over Tier A text documents."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.bm25_id_map[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def search_lane2_genome(self, query: str, top_k: int = DEFAULT_LANE_K) -> list[tuple[int, float]]:
        """Run Genome HNSW search over Tier A + B tag relevance vectors."""
        q_vec = self.build_genome_query_vector(query)
        if np.linalg.norm(q_vec) < 1e-10:
            return []  # no tag match found
        labels, dists = self.genome_index.knn_query([q_vec], k=top_k)
        results = []
        for lbl, dist in zip(labels[0], dists[0]):
            mid = self.genome_id_map[int(lbl)]
            # dist is cosine distance = 1 - cosine_similarity
            score = max(0.0, 1.0 - float(dist))
            results.append((mid, score))
        return results

    def search_lane3_dense(self, query: str, top_k: int = DEFAULT_LANE_K) -> list[tuple[int, float]]:
        """Run Dense Voyage-4-Large HNSW search over Tier A semantic vectors."""
        if not self.dense_index or not self.voyage_client:
            return []

        # Embed query text
        res = self.voyage_client.embed([query], model=VOYAGE_MODEL, input_type="query")
        q_vec = np.array(res.embeddings[0], dtype=np.float32)

        labels, dists = self.dense_index.knn_query([q_vec], k=top_k)
        results = []
        for lbl, dist in zip(labels[0], dists[0]):
            mid = self.dense_id_map[int(lbl)]
            score = max(0.0, 1.0 - float(dist))
            results.append((mid, score))
        return results

    # -----------------------------------------------------------------------
    # Fused Search (RRF)
    # -----------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 20,
        lane_k: int = DEFAULT_LANE_K,
        k_rrf: int = RRF_K,
        use_voyage: bool = True,
    ) -> list[dict]:
        """
        Perform multi-lane search and merge results using Reciprocal Rank Fusion.
        """
        # Execute active lanes
        lane_results = {
            "bm25":   self.search_lane1_bm25(query, top_k=lane_k),
            "genome": self.search_lane2_genome(query, top_k=lane_k),
            "dense":  self.search_lane3_dense(query, top_k=lane_k) if (use_voyage and self.voyage_client) else [],
        }

        rrf_scores = defaultdict(float)
        lane_hits = defaultdict(dict)

        for lane_name, hits in lane_results.items():
            for rank, (mid, _score) in enumerate(hits):
                # RRF formula: 1.0 / (k + rank + 1)
                rrf_scores[mid] += 1.0 / (k_rrf + rank + 1)
                lane_hits[mid][lane_name] = rank + 1  # 1-indexed rank for display

        # Sort by fused score descending
        sorted_mids = sorted(rrf_scores.keys(), key=lambda m: -rrf_scores[m])[:top_k]

        output = []
        for mid in sorted_mids:
            ranks = lane_hits[mid]
            output.append({
                "movie_id": mid,
                "rrf_score": round(rrf_scores[mid], 5),
                "lanes": list(ranks.keys()),
                "lane_ranks": ranks,
            })

        return output


# ---------------------------------------------------------------------------
# CLI test runner
# ---------------------------------------------------------------------------
def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "atmospheric slow burn Korean thriller"
    retriever = CineVaultRetriever()

    print(f"Query: '{query}'")
    print("=" * 60)
    t0 = time.time()
    results = retriever.search(query, top_k=10)
    elapsed = time.time() - t0

    print(f"Found {len(results)} results in {elapsed*1000:.1f}ms:\n")
    for rank, res in enumerate(results, 1):
        lanes_str = ", ".join(f"{l}:#{r}" for l, r in res["lane_ranks"].items())
        print(f" #{rank:2d}  movie_id={res['movie_id']:6d}  score={res['rrf_score']:.5f}  [{lanes_str}]")


if __name__ == "__main__":
    main()
