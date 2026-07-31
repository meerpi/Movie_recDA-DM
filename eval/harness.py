"""
eval/harness.py — Offline evaluation metrics for CineVault.

Provides four metric functions and one end-to-end per-user evaluator:

  recall_at_k(retrieved_ids, relevant_ids, k)
      Set-based Recall@K.

  ndcg_at_k(ranked_items, relevant_ids, k)
      Binary-relevance NDCG@K.  Assumes the list is already in the correct
      ranked order (deterministic tiebreak must be applied by the caller).

  intra_list_distance(top_k_items, embedding_fn, k)
      Average pairwise cosine distance across the top-K items.

  load_embedding_from_db(db_path, column)
      Returns a closure suitable for ``intra_list_distance`` that lazily
      reads float32 BLOBs from ``movie_embeddings``.

  evaluate_user(user_id, test_interactions, ranked_results, retrieved_ids, ...)
      Computes all metrics for a single user and returns them in a dict.

Metric formulae
---------------
Recall@K  = |retrieved[:K] ∩ relevant| / |relevant|

DCG@K     = Σ_{i=1}^{K}  rel_i / log2(i + 1)       [binary rel_i ∈ {0, 1}]
IDCG@K    = Σ_{i=1}^{min(|relevant|, K)} 1 / log2(i + 1)
NDCG@K    = DCG@K / IDCG@K

ILD       = (2 / (K·(K−1))) · Σ_{i<j} (1 − cosine_similarity(e_i, e_j))
"""

import math
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[int],
    relevant_ids: Set[int],
    k: int = 100,
) -> float:
    """
    Computes Recall@K: the fraction of relevant items captured in the top-K
    retrieved set.

    Formally:  |retrieved[:K] ∩ relevant| / |relevant|

    Returns 0.0 when ``relevant_ids`` is empty (no ground truth to recall).

    Parameters
    ----------
    retrieved_ids : list of int
        Ordered list of movie_ids as returned by the retriever, before any
        reranking.  Only the first K entries are considered.
    relevant_ids : set of int
        Ground-truth relevant movie_ids for this user (e.g. ``rating >= 4.0``
        in the held-out test set).
    k : int
        Retrieval depth.

    Returns
    -------
    float
        Recall in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return len(top_k_set & relevant_ids) / len(relevant_ids)


# ---------------------------------------------------------------------------
# NDCG@K
# ---------------------------------------------------------------------------

def ndcg_at_k(
    ranked_items: List[Dict[str, Any]],
    relevant_ids: Set[int],
    k: int = 10,
) -> float:
    """
    Computes NDCG@K over a final ranked list using binary relevance labels.

    Relevant items (``movie_id in relevant_ids``) receive relevance = 1;
    all others receive 0.

    The list is assumed to already carry the correct ranking order.  The
    deterministic tiebreak — ``(final_score DESC, movie_id ASC)`` — must be
    applied by the caller (e.g. via ``nlp.pipeline.score_candidates``) before
    passing the list here; this function does not re-sort.

    DCG@K  = Σ_{i=1}^{K} rel_i / log2(i + 1)
    IDCG@K = Σ_{i=1}^{min(|relevant|, K)} 1 / log2(i + 1)
    NDCG@K = DCG@K / IDCG@K

    Returns 0.0 when ``relevant_ids`` is empty or IDCG is effectively zero.

    Parameters
    ----------
    ranked_items : list of dict
        Ordered candidate dicts, each with at minimum a ``"movie_id"`` key.
    relevant_ids : set of int
        Ground-truth relevant movie_ids.
    k : int
        Ranking depth.

    Returns
    -------
    float
        NDCG in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0

    dcg = 0.0
    for i, item in enumerate(ranked_items[:k], start=1):
        if item.get("movie_id") in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)

    n_ideal = min(len(relevant_ids), k)
    idcg    = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    if idcg < 1e-12:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Intra-List Distance
# ---------------------------------------------------------------------------

def intra_list_distance(
    top_k_items: List[Dict[str, Any]],
    embedding_fn: Callable[[int], Optional[np.ndarray]],
    k: int = 10,
) -> float:
    """
    Computes Intra-List Distance (ILD): average pairwise cosine distance
    between the embedding vectors of the top-K items.

    ILD = (2 / (K·(K−1))) · Σ_{i<j} (1 − cosine_similarity(e_i, e_j))

    Cosine distance lies in [0.0, 2.0] for arbitrary unit vectors; for
    non-negative embedding spaces (e.g. genome relevance scores) it is in
    [0.0, 1.0].  Items whose embedding cannot be retrieved, or whose vector has
    zero norm, are silently excluded from the pairwise computation.  If fewer
    than two embeddings are available, returns 0.0.

    Parameters
    ----------
    top_k_items : list of dict
        Ordered candidate dicts, each with a ``"movie_id"`` key.
    embedding_fn : callable
        ``(movie_id: int) → np.ndarray or None``  The array may be any
        fixed-dimensionality dense float vector; it does not need to be
        pre-normalised.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Average pairwise cosine distance.
    """
    unit_vecs: List[np.ndarray] = []
    for item in top_k_items[:k]:
        mid = item.get("movie_id")
        if mid is None:
            continue
        raw = embedding_fn(mid)
        if raw is None:
            continue
        vec  = np.asarray(raw, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        unit_vecs.append(vec / norm)

    n = len(unit_vecs)
    if n < 2:
        return 0.0

    total_dist, n_pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim    = float(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_sim    = max(-1.0, min(1.0, cos_sim))   # guard fp drift
            total_dist += 1.0 - cos_sim
            n_pairs    += 1

    return total_dist / n_pairs if n_pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# DB-backed embedding loader
# ---------------------------------------------------------------------------

def load_embedding_from_db(
    db_path: Any = DB_PATH,
    column: str = "v_genome",
) -> Callable[[int], Optional[np.ndarray]]:
    """
    Returns a closure that reads one movie's embedding from cinevault.db.

    Embeddings are persisted as raw float32 BLOBs in ``movie_embeddings``.
    Opening a fresh connection per call keeps the closure thread-safe and
    avoids holding a connection open across the full evaluation run.

    Parameters
    ----------
    db_path : Path or str
        Path to cinevault.db.
    column : str
        Column to read: ``'v_genome'`` (1128-d), ``'v_critic'``, or
        ``'v_audience'``.

    Returns
    -------
    callable
        ``(movie_id: int) → np.ndarray or None``
    """
    def _load(movie_id: int) -> Optional[np.ndarray]:
        try:
            conn = sqlite3.connect(str(db_path))
            row  = conn.execute(
                f"SELECT {column} FROM movie_embeddings WHERE movie_id = ?",
                (movie_id,),
            ).fetchone()
            conn.close()
            if row and row[0]:
                return np.frombuffer(bytes(row[0]), dtype=np.float32).copy()
        except Exception:
            pass
        return None

    return _load


# ---------------------------------------------------------------------------
# End-to-end per-user evaluator
# ---------------------------------------------------------------------------

def evaluate_user(
    user_id: str,
    test_interactions: List[Dict[str, Any]],
    ranked_results: List[Dict[str, Any]],
    retrieved_ids: List[int],
    embedding_fn: Optional[Callable[[int], Optional[np.ndarray]]] = None,
    k_recall: int = 100,
    k_ndcg: int = 10,
    k_ild: int = 10,
    min_star_relevant: float = 4.0,
) -> Dict[str, Any]:
    """
    Computes all offline metrics for a single user.

    The caller is responsible for:

    1. Calling ``eval.replay.time_split_user()`` to obtain ``test_interactions``.
    2. Calling ``eval.replay.replay_profile()`` with the cutoff timestamp from the
       last train interaction.
    3. Running the retriever to get ``retrieved_ids`` (for Recall@K).
    4. Calling ``nlp.pipeline.score_candidates()`` to get ``ranked_results``
       (already sorted with the deterministic tiebreak applied).

    This function only computes the metrics; it performs no I/O.

    Parameters
    ----------
    user_id : str
        User identifier, included in the return dict for logging.
    test_interactions : list of dict
        Held-out interactions.  Each dict must have ``"movie_id"`` and
        ``"rating"`` keys.
    ranked_results : list of dict
        Final ranked candidate list from ``score_candidates()``, ordered by
        ``(final_score DESC, movie_id ASC)``.
    retrieved_ids : list of int
        All movie_ids returned by the retriever stage (before reranking).
    embedding_fn : callable or None
        ``(movie_id: int) → np.ndarray or None``.  If None, ILD is skipped and
        returned as None.
    k_recall : int
        Retrieval depth for Recall (default 100).
    k_ndcg : int
        Ranking depth for NDCG (default 10).
    k_ild : int
        Number of top items for ILD (default 10).
    min_star_relevant : float
        Minimum star rating to label an interaction "relevant" (default 4.0).

    Returns
    -------
    dict
        Keys: ``user_id``, ``n_test``, ``n_relevant``, ``recall_at_k``,
        ``ndcg_at_k``, ``ild`` (float or None), ``k_recall``, ``k_ndcg``.
    """
    relevant_ids: Set[int] = {
        int(r["movie_id"])
        for r in test_interactions
        if float(r.get("rating", 0.0)) >= min_star_relevant
    }

    rec  = recall_at_k(retrieved_ids, relevant_ids, k=k_recall)
    ndcg = ndcg_at_k(ranked_results,  relevant_ids, k=k_ndcg)
    ild: Optional[float] = (
        intra_list_distance(ranked_results, embedding_fn, k=k_ild)
        if embedding_fn is not None
        else None
    )

    return {
        "user_id":    user_id,
        "n_test":     len(test_interactions),
        "n_relevant": len(relevant_ids),
        "recall_at_k": rec,
        "ndcg_at_k":   ndcg,
        "ild":         ild,
        "k_recall":    k_recall,
        "k_ndcg":      k_ndcg,
    }
