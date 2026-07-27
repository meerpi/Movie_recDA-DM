"""
nlp/retrieval_augment.py — Two supplementary retrieval lanes that run AFTER the
main RRF search and augment the candidate pool before reranking.

Lane A — Profile-Entity SQL Backstop  (Fix 2)
----------------------------------------------
Queries the DB directly for movies whose directors or lead actors match the
user's top affinities. Catches well-known director/actor titles that don't rank
in the ANN top-k for a given topic query (e.g. "Alien" under "sci-fi adventure").

Lane B — Local Fallback Dense HNSW  (Fix 3)
---------------------------------------------
A sentence-transformers (all-MiniLM-L6-v2, 384-dim) index built over the ~48 K
movies NOT in the Voyage dense index. Provides semantic recall for long-tail
titles whose Voyage embeddings don't exist.

Both lanes return lists of raw dicts compatible with the retriever's output
format so they can be merged with the main search_hits before hydration.

Usage (in eval scripts):
    from nlp.retrieval_augment import entity_backstop, FallbackDenseRetriever

    # After main retrieval:
    retrieved_ids = {h['movie_id'] for h in search_hits}

    extra = entity_backstop(profile, db_path, retrieved_ids, max_per_entity=20)
    search_hits.extend(extra)                    # merge before hydration

    fallback = FallbackDenseRetriever()
    extra2 = fallback.search(query, retrieved_ids, top_k=50)
    search_hits.extend(extra2)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np

logger = logging.getLogger("cinevault.retrieval_augment")

PROJECT_ROOT = Path(__file__).parent.parent
FALLBACK_INDEX_PATH  = PROJECT_ROOT / "dirtywork" / "fallback_dense.hnsw"
FALLBACK_MAP_PATH    = PROJECT_ROOT / "dirtywork" / "fallback_dense_id_map.json"
FALLBACK_DIM         = 384
FALLBACK_MODEL_NAME  = "all-MiniLM-L6-v2"


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_stub(movie_id: int, source: str) -> Dict[str, Any]:
    """Minimal retrieval hit dict that hydrator can process."""
    return {
        "movie_id":   movie_id,
        "rrf_rank":   9999,
        "rrf_score":  0.001,          # low but non-zero so it isn't dropped
        "lanes":      [source],
        "lane_ranks": {source: 9999},
    }


# ── Fix 2: Profile-Entity SQL Backstop ───────────────────────────────────────

def entity_backstop(
    profile,
    db_path: Path,
    already_retrieved: Set[int],
    n_directors: int = 3,
    n_actors: int = 5,
    max_per_entity: int = 20,
) -> List[Dict[str, Any]]:
    """
    Returns stub retrieval hits for movies by the user's top directors/actors
    that were NOT already in the main retrieval pool.

    Parameters
    ----------
    profile          : UserProfile with director_affinity / actor_affinity dicts
    db_path          : Path to cinevault SQLite database
    already_retrieved: Set of movie_ids already returned by the main retriever
    n_directors      : Number of top directors to query (sorted by affinity weight)
    n_actors         : Number of top lead actors to query
    max_per_entity   : Max DB rows fetched per director / actor

    Returns
    -------
    List of minimal retrieval-hit dicts (same shape as retriever.search() output)
    ready to be merged into search_hits and passed to hydrator.hydrate().
    """
    # Only query entities with meaningful positive affinity
    MIN_AFFINITY = 0.05

    top_directors = [
        d for d in profile.get_top_directors(n_directors)
        if profile.director_affinity.get(d, 0.0) >= MIN_AFFINITY
    ]
    top_actors = [
        a for a in getattr(profile, "get_top_actors", lambda n: [])(n_actors)
        if profile.actor_affinity.get(a, 0.0) >= MIN_AFFINITY
    ]

    # Fallback: read actors sorted by affinity weight if no get_top_actors method
    if not hasattr(profile, "get_top_actors"):
        top_actors = sorted(
            profile.actor_affinity, key=profile.actor_affinity.get, reverse=True
        )[:n_actors]
        top_actors = [a for a in top_actors if profile.actor_affinity.get(a, 0.0) >= MIN_AFFINITY]

    if not top_directors and not top_actors:
        return []

    conn = sqlite3.connect(str(db_path))
    seen: Set[int] = set(already_retrieved)
    results: List[Dict[str, Any]] = []

    try:
        for director in top_directors:
            # directors column is plain text: "Ridley Scott" or "A, B"
            rows = conn.execute(
                "SELECT movie_id FROM movies WHERE directors LIKE ? LIMIT ?",
                (f"%{director}%", max_per_entity),
            ).fetchall()
            for (mid,) in rows:
                if mid not in seen:
                    seen.add(mid)
                    results.append(_make_stub(mid, "entity_director"))
            logger.debug(f"  entity_backstop director={director!r}: {len(rows)} rows")

        for actor in top_actors:
            rows = conn.execute(
                "SELECT movie_id FROM movies WHERE actors LIKE ? LIMIT ?",
                (f"%{actor}%", max_per_entity),
            ).fetchall()
            for (mid,) in rows:
                if mid not in seen:
                    seen.add(mid)
                    results.append(_make_stub(mid, "entity_actor"))
            logger.debug(f"  entity_backstop actor={actor!r}: {len(rows)} rows")

    finally:
        conn.close()

    logger.info(
        f"entity_backstop: +{len(results)} candidates "
        f"(directors={top_directors}, actors={top_actors[:3]})"
    )
    return results


# ── Fix 3: Local Fallback Dense HNSW ─────────────────────────────────────────

class FallbackDenseRetriever:
    """
    Lightweight semantic search over movies NOT in the Voyage dense index.

    Uses all-MiniLM-L6-v2 (384-dim, locally cached) to embed the query and
    searches a pre-built hnswlib index covering ~48 K long-tail titles.

    Call build_fallback_index.py once to create the index file, then use this
    class in eval scripts for augmented recall.
    """

    def __init__(self) -> None:
        if not FALLBACK_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"Fallback index not found: {FALLBACK_INDEX_PATH}\n"
                "Run:  .venv/bin/python scripts/build_fallback_index.py"
            )

        with open(FALLBACK_MAP_PATH, encoding="utf-8") as f:
            self._id_map: List[int] = json.load(f)

        import hnswlib
        self._index = hnswlib.Index(space="cosine", dim=FALLBACK_DIM)
        self._index.load_index(str(FALLBACK_INDEX_PATH), max_elements=len(self._id_map))
        self._index.set_ef(100)

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(FALLBACK_MODEL_NAME)

        logger.info(
            f"FallbackDenseRetriever loaded: {len(self._id_map):,} vectors "
            f"({FALLBACK_MODEL_NAME}, {FALLBACK_DIM}-dim)"
        )

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._model.encode([query], normalize_embeddings=True)[0]
        return vec.astype(np.float32)

    def search(
        self,
        query: str,
        already_retrieved: Set[int],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Returns up to top_k stub retrieval hits for movies NOT in
        already_retrieved whose embedding is closest to the query.
        """
        q_vec = self.embed_query(query)
        # Fetch 3x requested so we have headroom after filtering out
        # already-retrieved items
        fetch_k = min(top_k * 3, len(self._id_map))
        labels, dists = self._index.knn_query([q_vec], k=fetch_k)

        results: List[Dict[str, Any]] = []
        for lbl, dist in zip(labels[0], dists[0]):
            mid = self._id_map[int(lbl)]
            if mid in already_retrieved:
                continue
            stub = _make_stub(mid, "fallback_dense")
            stub["rrf_score"] = max(0.0, round(1.0 - float(dist), 5))
            results.append(stub)
            if len(results) >= top_k:
                break

        logger.info(f"FallbackDenseRetriever.search: +{len(results)} candidates")
        return results
