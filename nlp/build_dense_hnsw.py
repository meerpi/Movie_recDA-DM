#!/usr/bin/env python3
"""
nlp/build_dense_hnsw.py — Step 4: Build Dense HNSW Index (Lane 3)

Loads the 1024-dimensional Voyage AI dense embeddings generated in Step 3 from:
  • nlp/tier_a_voyage_1024d.npy    (shape: [9526, 1024], float32)
  • nlp/tier_a_voyage_id_map.json  (list of movie_ids)

Builds an hnswlib cosine-space HNSW index covering all 9,526 Tier A movies.

Output:
  nlp/dense.hnsw          — binary HNSW index file
  nlp/dense_id_map.json   — maps internal row index → movie_id (list of ints)

Usage:
    cd /home/meerpi/curr_project/movie_rec
    .venv/bin/python nlp/build_dense_hnsw.py

Runtime: ~5-10 seconds on CPU.
"""

import json
import time
from pathlib import Path

import hnswlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths (run from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.parent
NPY_PATH       = PROJECT_ROOT / "nlp" / "tier_a_voyage_1024d.npy"
NPY_MAP_PATH   = PROJECT_ROOT / "nlp" / "tier_a_voyage_id_map.json"
INDEX_PATH     = PROJECT_ROOT / "nlp" / "dense.hnsw"
ID_MAP_PATH    = PROJECT_ROOT / "nlp" / "dense_id_map.json"

# ---------------------------------------------------------------------------
# HNSW Hyperparameters
# ---------------------------------------------------------------------------
DIM             = 1024
SPACE           = "cosine"     # Cosine similarity
M               = 32           # Bidirectional links per node
EF_CONSTRUCTION = 200          # Search depth during index build
EF_SEARCH       = 100          # Runtime search depth for queries


def main() -> None:
    print("=" * 60)
    print("CineVault — Build Dense HNSW Index (Step 4)")
    print("=" * 60)

    # ── 1. Load Step 3 artifacts ──────────────────────────────────
    if not NPY_PATH.exists():
        raise FileNotFoundError(f"[ERROR] {NPY_PATH} not found. Complete Step 3 first.")
    if not NPY_MAP_PATH.exists():
        raise FileNotFoundError(f"[ERROR] {NPY_MAP_PATH} not found. Complete Step 3 first.")

    print(f"  Loading embeddings from {NPY_PATH.name} ...")
    t0 = time.time()
    embeddings = np.load(str(NPY_PATH))

    with open(NPY_MAP_PATH, encoding="utf-8") as f:
        movie_ids = json.load(f)

    N, dim = embeddings.shape
    print(f"  Loaded matrix shape [{N:,}, {dim}] in {time.time() - t0:.2f}s")
    print(f"  ID map contains {len(movie_ids):,} entries")

    if N != len(movie_ids):
        raise ValueError(f"Mismatch: matrix has {N} rows but ID map has {len(movie_ids)} entries.")
    if dim != DIM:
        raise ValueError(f"Mismatch: expected {DIM}-dim vectors, got {dim}-dim.")

    # ── 2. Build HNSW index ──────────────────────────────────────
    print(f"\n  Building HNSW index (dim={DIM}, space={SPACE!r}, M={M}, ef_construction={EF_CONSTRUCTION}) ...")
    t_build = time.time()

    index = hnswlib.Index(space=SPACE, dim=DIM)
    index.init_index(max_elements=N, M=M, ef_construction=EF_CONSTRUCTION, random_seed=42)
    index.set_ef(EF_SEARCH)

    labels = list(range(N))
    index.add_items(embeddings, labels)

    elapsed_build = time.time() - t_build
    print(f"    → Index built in {elapsed_build:.2f}s ({N / elapsed_build:.0f} vectors/sec)")

    # ── 3. Save index & ID map ───────────────────────────────────
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(INDEX_PATH))
    index_mb = INDEX_PATH.stat().st_size / 1_048_576
    print(f"\n  Saved index → {INDEX_PATH}  ({index_mb:.1f} MB)")

    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(movie_ids, f)
    print(f"  Saved id map → {ID_MAP_PATH}  ({N} entries)")

    # ── 4. Smoke test ────────────────────────────────────────────
    print("\n  Smoke test — searching for nearest neighbours of movie_id "
          f"{movie_ids[0]} (row 0) ...")
    labels_out, distances = index.knn_query([embeddings[0]], k=6)
    hits = list(zip(labels_out[0], distances[0]))
    print("  Results (label → movie_id | distance):")
    for lbl, dist in hits:
        mid_hit = movie_ids[int(lbl)]
        self_flag = " ← self" if mid_hit == movie_ids[0] else ""
        print(f"    label={lbl:5d}  movie_id={mid_hit:6d}  dist={dist:.4f}{self_flag}")

    # ── 5. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("=" * 60)
    print(f"  dense.hnsw       : {INDEX_PATH}")
    print(f"  dense_id_map.json: {ID_MAP_PATH}")
    print(f"  Total movies indexed: {N:,}")
    print(f"  Build time: {elapsed_build:.2f}s")
    print()
    print("  Next: STEP 5 — Build RRF Retriever (nlp/retriever.py)")


if __name__ == "__main__":
    main()
