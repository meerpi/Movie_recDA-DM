#!/usr/bin/env python3
"""
nlp/build_genome_hnsw.py — Step 2: Build Genome HNSW Index (Lane 2)

Loads the 1128-dimensional tag-genome vectors from:
  • tier_a_profile_cards_v3.jsonl  (~9,526 movies, LLM cards + genome)
  • tier_b_profile_cards.jsonl     (~4,290 movies, genome only)

Builds an hnswlib cosine-space HNSW index covering all 13,816 Tier A + B movies.

Output:
  nlp/genome.hnsw          — binary HNSW index file
  nlp/genome_id_map.json   — maps internal row index → movie_id (list of ints)

Usage:
    cd /home/meerpi/curr_project/movie_rec
    .venv/bin/python nlp/build_genome_hnsw.py

Runtime: ~5-15s for 13,816 vectors at 1128 dims on CPU.
No API calls needed — pure local computation.
"""

import json
import math
import struct
import sys
import time
from pathlib import Path

import hnswlib

# ---------------------------------------------------------------------------
# Paths (run from project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.parent
TIER_A_CARDS   = PROJECT_ROOT / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS   = PROJECT_ROOT / "tier_b_profile_cards.jsonl"
INDEX_PATH     = PROJECT_ROOT / "nlp" / "genome.hnsw"
ID_MAP_PATH    = PROJECT_ROOT / "nlp" / "genome_id_map.json"

# ---------------------------------------------------------------------------
# HNSW hyperparameters (from architecture.md)
# ---------------------------------------------------------------------------
DIM             = 1128
SPACE           = "cosine"     # hnswlib normalises internally for cosine
M               = 32           # bidirectional links per node — higher → better recall, more RAM
EF_CONSTRUCTION = 200          # search depth during build — higher → better recall, slower build
EF_SEARCH       = 100          # runtime search depth (set on index after load for queries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_norm(vec: list[float]) -> list[float]:
    """L2-normalise a vector in place (returns new list)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-10:
        return vec   # zero vector — leave as-is
    inv = 1.0 / norm
    return [x * inv for x in vec]


def load_cards(path: Path, label: str) -> tuple[list[int], list[list[float]]]:
    """
    Stream a JSONL profile-card file and extract (movie_id, genome_vector) pairs.
    Skips any record without a 'genome_vector' field or with wrong dimensions.
    Returns (movie_ids, vectors).
    """
    movie_ids: list[int] = []
    vectors:   list[list[float]] = []
    skipped = 0

    if not path.exists():
        print(f"  [WARN] {path.name} not found — skipping {label}.", file=sys.stderr)
        return movie_ids, vectors

    print(f"  Loading {label} from {path.name} ...")
    t0 = time.time()

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON error on line {lineno}: {e}", file=sys.stderr)
                skipped += 1
                continue

            mid = rec.get("movie_id")
            gv  = rec.get("genome_vector")

            if mid is None or gv is None:
                skipped += 1
                continue

            if len(gv) != DIM:
                print(
                    f"  [WARN] movie_id={mid} has genome_vector dim={len(gv)}, expected {DIM} — skipping.",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            movie_ids.append(int(mid))
            vectors.append(l2_norm(gv))   # normalise now; hnswlib cosine also normalises but
                                           # doing it here makes the vectors portable for other uses

    elapsed = time.time() - t0
    print(f"    → {len(movie_ids)} vectors loaded in {elapsed:.1f}s"
          + (f" ({skipped} skipped)" if skipped else ""))
    return movie_ids, vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CineVault — Build Genome HNSW Index (Step 2)")
    print("=" * 60)

    # ── 1. Load Tier A ───────────────────────────────────────────
    a_ids, a_vecs = load_cards(TIER_A_CARDS, "Tier A (v3)")

    # ── 2. Load Tier B ───────────────────────────────────────────
    b_ids, b_vecs = load_cards(TIER_B_CARDS, "Tier B")

    # ── 3. Deduplicate (Tier A takes priority) ───────────────────
    seen: set[int] = set(a_ids)
    b_ids_dedup, b_vecs_dedup = [], []
    for mid, vec in zip(b_ids, b_vecs):
        if mid not in seen:
            b_ids_dedup.append(mid)
            b_vecs_dedup.append(vec)
            seen.add(mid)

    if len(b_ids) != len(b_ids_dedup):
        print(f"  [INFO] Tier B dedup removed {len(b_ids) - len(b_ids_dedup)} IDs "
              f"that already appear in Tier A.")

    all_ids  = a_ids  + b_ids_dedup
    all_vecs = a_vecs + b_vecs_dedup
    total    = len(all_ids)

    print(f"\n  Tier A: {len(a_ids):,} movies")
    print(f"  Tier B: {len(b_ids_dedup):,} movies (net new after dedup)")
    print(f"  Total : {total:,} movies → index will cover {total:,} vectors")

    if total == 0:
        sys.exit("[ERROR] No vectors to index. Check input files.")

    # ── 4. Build HNSW index ──────────────────────────────────────
    print(f"\n  Building HNSW index  (dim={DIM}, space={SPACE!r}, "
          f"M={M}, ef_construction={EF_CONSTRUCTION}) ...")
    t_build = time.time()

    index = hnswlib.Index(space=SPACE, dim=DIM)
    index.init_index(max_elements=total, M=M, ef_construction=EF_CONSTRUCTION, random_seed=42)
    index.set_ef(EF_SEARCH)

    # Add vectors in one shot — row i in all_vecs maps to internal label i
    # We add them with sequential integer labels 0..total-1
    # genome_id_map.json then maps label → movie_id
    labels = list(range(total))
    index.add_items(all_vecs, labels)

    elapsed_build = time.time() - t_build
    print(f"    → Index built in {elapsed_build:.1f}s  "
          f"({total / elapsed_build:.0f} vectors/sec)")

    # ── 5. Save index ────────────────────────────────────────────
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(INDEX_PATH))
    index_mb = INDEX_PATH.stat().st_size / 1_048_576
    print(f"\n  Saved index → {INDEX_PATH}  ({index_mb:.1f} MB)")

    # ── 6. Save id map ───────────────────────────────────────────
    # List where position i = movie_id for internal label i
    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)
    print(f"  Saved id map → {ID_MAP_PATH}  ({total} entries)")

    # ── 7. Smoke test ────────────────────────────────────────────
    print("\n  Smoke test — searching for the 5 nearest neighbours of movie_id "
          f"{all_ids[0]} (vector[0]) ...")
    labels_out, distances = index.knn_query([all_vecs[0]], k=6)  # k=6 because [0] returns itself
    hits = list(zip(labels_out[0], distances[0]))
    print("  Results (label → movie_id | distance):")
    for lbl, dist in hits:
        mid_hit = all_ids[int(lbl)]
        self_flag = " ← self" if mid_hit == all_ids[0] else ""
        print(f"    label={lbl:5d}  movie_id={mid_hit:6d}  dist={dist:.4f}{self_flag}")

    # ── 8. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"  genome.hnsw      : {INDEX_PATH}")
    print(f"  genome_id_map.json: {ID_MAP_PATH}")
    print(f"  Total movies indexed: {total:,}")
    print(f"  Tier A: {len(a_ids):,}  |  Tier B (net): {len(b_ids_dedup):,}")
    print(f"  Build time: {elapsed_build:.1f}s")
    print()
    print("  Next step: STEP 3 — Generate Dense Embeddings (Kaggle/Voyage AI)")
    print("  Or:        STEP 5 — Build retriever.py using BM25 + Genome HNSW lanes")


if __name__ == "__main__":
    main()
