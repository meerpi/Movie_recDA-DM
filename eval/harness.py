"""
eval/harness.py — Offline evaluation metrics for CineVault.

Recall@K  = |retrieved[:K] ∩ relevant| / |relevant|
DCG@K     = Σ_{i=1}^{K}  rel_i / log2(i + 1)       [binary rel_i ∈ {0, 1}]
IDCG@K    = Σ_{i=1}^{min(|relevant|, K)} 1 / log2(i + 1)
NDCG@K    = DCG@K / IDCG@K
ILD       = (2 / (K·(K−1))) · Σ_{i<j} (1 − cosine_similarity(e_i, e_j))
"""

import math
import sqlite3
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"


# ── Recall@K ─────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids, relevant_ids, k=100):
    """|retrieved[:K] ∩ relevant| / |relevant|.  Returns 0.0 if relevant is empty."""
    if not relevant_ids:
        return 0.0
    top_k_set = set(retrieved_ids[:k])
    return len(top_k_set & relevant_ids) / len(relevant_ids)


# ── NDCG@K ───────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_items, relevant_ids, k=10):
    """Binary-relevance NDCG@K.  List must already be in final ranked order."""
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


# ── Intra-List Distance ──────────────────────────────────────────────────────

def intra_list_distance(top_k_items, embedding_fn, k=10):
    """Average pairwise cosine distance across top-K embeddings (ILD).  Returns 0.0 if < 2 vectors available."""
    unit_vecs = []
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


# ── DB-backed embedding loader ───────────────────────────────────────────────

def load_embedding_from_db(db_path=DB_PATH, column="v_genome"):
    """Returns a closure (movie_id → np.ndarray | None) that reads float32 BLOBs from movie_embeddings."""
    def _load(movie_id):
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


# ── End-to-end per-user evaluator ────────────────────────────────────────────

def evaluate_user(user_id, test_interactions, ranked_results, retrieved_ids,
                  embedding_fn=None, k_recall=100, k_ndcg=10, k_ild=10,
                  min_star_relevant=4.0):
    """Computes Recall@K, NDCG@K, and ILD for one user.  Pure metric math, no I/O."""
    relevant_ids = {
        int(r["movie_id"])
        for r in test_interactions
        if float(r.get("rating", 0.0)) >= min_star_relevant
    }

    rec  = recall_at_k(retrieved_ids, relevant_ids, k=k_recall)
    ndcg = ndcg_at_k(ranked_results,  relevant_ids, k=k_ndcg)
    ild  = (
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
