#!/usr/bin/env python3
"""
scripts/build_fallback_index.py — Build a local 384-dim HNSW index covering
the ~48 K movies that are NOT in the Voyage dense index (dirtywork/dense.hnsw).

Embedding model : all-MiniLM-L6-v2  (locally cached, no API key needed)
Dimension       : 384
Text strategy   : "{title} {year} {directors} {actors} {top_tags}"
                  Falls back to title-only for movies with no profile card data.

Output:
    dirtywork/fallback_dense.hnsw         — binary hnswlib index
    dirtywork/fallback_dense_id_map.json  — list[int] mapping row → movie_id

Runtime: ~2–4 minutes on CPU for 48 K movies.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import hnswlib
import numpy as np

PROJECT_ROOT   = Path(__file__).parent.parent
DENSE_MAP_PATH = PROJECT_ROOT / "dirtywork" / "dense_id_map.json"
OUT_INDEX_PATH = PROJECT_ROOT / "dirtywork" / "fallback_dense.hnsw"
OUT_MAP_PATH   = PROJECT_ROOT / "dirtywork" / "fallback_dense_id_map.json"
DB_PATH        = PROJECT_ROOT / "db" / "cinevault.db"

TIER_A_CARDS   = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS   = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
TIER_C_CARDS   = PROJECT_ROOT / "dirtywork" / "tier_c_profile_cards.jsonl"

MODEL_NAME  = "all-MiniLM-L6-v2"
DIM         = 384
M           = 16          # fewer bidirectional links vs main index (faster build)
EF_CONST    = 100
EF_SEARCH   = 100
BATCH_SIZE  = 128         # encode() batch size — kept small for CPU inference


def load_profile_cards() -> Dict[int, dict]:
    """Load all three tier card files into a movie_id → card dict."""
    cards: Dict[int, dict] = {}
    for path in (TIER_A_CARDS, TIER_B_CARDS, TIER_C_CARDS):
        if not path.exists():
            print(f"  [warn] {path.name} not found, skipping.")
            continue
        t0 = time.time()
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    mid = int(card.get("movie_id", 0))
                    if mid:
                        cards[mid] = card
                        n += 1
                except json.JSONDecodeError:
                    continue
        print(f"  Loaded {n:,} cards from {path.name}  ({time.time()-t0:.1f}s)")
    return cards


def build_text(movie_id: int, db_row: tuple, card: Optional[dict]) -> str:
    """
    Construct embedding text from whatever data is available.

    Priority: profile card fields > DB fields > title only.
    """
    parts: List[str] = []

    # Title (DB)
    title = db_row[1] or ""
    year  = str(db_row[2]) if db_row[2] else ""
    db_directors = db_row[3] or ""
    db_actors    = db_row[4] or ""
    db_overview  = db_row[5] or ""

    parts.append(f"{title} {year}".strip())

    if card:
        directors = card.get("directors") or db_directors
        actors    = card.get("actors") or db_actors
        top_tags  = card.get("top_tags") or []
        synopsis  = card.get("synopsis") or card.get("overview") or db_overview

        if directors:
            parts.append(f"directed by {directors}")
        if actors:
            # actors may be comma-separated string or list
            if isinstance(actors, list):
                actors = ", ".join(actors[:5])
            parts.append(f"starring {actors[:200]}")
        if top_tags:
            # top_tags may be list of str or list of dicts
            tag_strs = []
            for t in top_tags[:15]:
                if isinstance(t, str):
                    tag_strs.append(t)
                elif isinstance(t, dict):
                    tag_strs.append(t.get("tag", ""))
            parts.append(" ".join(tag_strs))
        if synopsis:
            parts.append(synopsis[:300])
    else:
        # DB-only fallback
        if db_directors:
            parts.append(f"directed by {db_directors}")
        if db_actors:
            parts.append(f"starring {db_actors[:200]}")
        if db_overview:
            parts.append(db_overview[:300])

    return " ".join(p for p in parts if p).strip()


def main() -> None:
    print("=" * 64)
    print("Build Fallback Dense HNSW Index (all-MiniLM-L6-v2, 384-dim)")
    print("=" * 64)

    # ── 1. Load existing index map to find gap ────────────────────
    with open(DENSE_MAP_PATH, encoding="utf-8") as f:
        already_indexed: set = set(json.load(f))
    print(f"\nVoyage index covers {len(already_indexed):,} movies.")

    # ── 2. Get all movie_ids from DB ──────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    all_db_rows = conn.execute(
        "SELECT movie_id, title, year, directors, actors, overview FROM movies"
    ).fetchall()
    conn.close()

    db_map = {row[0]: row for row in all_db_rows}
    print(f"Total movies in DB: {len(db_map):,}")

    # Movies to index: in DB but NOT in Voyage index
    to_index_ids = [mid for mid in db_map if mid not in already_indexed]
    print(f"Movies to add to fallback index: {len(to_index_ids):,}")

    # ── 3. Load profile cards for richer text ─────────────────────
    print("\nLoading profile cards...")
    cards = load_profile_cards()

    # ── 4. Build text corpus ──────────────────────────────────────
    print("\nBuilding text corpus...")
    texts:      List[str] = []
    movie_ids:  List[int] = []
    t0 = time.time()

    for mid in to_index_ids:
        db_row = db_map[mid]
        card   = cards.get(mid)
        text   = build_text(mid, db_row, card)
        texts.append(text if text else (db_row[1] or f"movie {mid}"))
        movie_ids.append(mid)

    print(f"  {len(texts):,} texts ready in {time.time()-t0:.1f}s")

    # ── 5. Embed with sentence-transformers ──────────────────────
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()

    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"\nEmbedding with {MODEL_NAME} on device={device} (batch_size={BATCH_SIZE})...")
    t0    = time.time()

    all_embeddings: List[np.ndarray] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch  = texts[i : i + BATCH_SIZE]
        vecs   = model.encode(
            batch,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(vecs.astype(np.float32))
        if (i // BATCH_SIZE) % 20 == 0:
            pct = (i + len(batch)) / len(texts) * 100
            elapsed = time.time() - t0
            print(f"  [{i + len(batch):,}/{len(texts):,}] {pct:.1f}%  elapsed: {elapsed:.0f}s")

    embeddings = np.vstack(all_embeddings)
    print(f"  Embedding complete: shape={embeddings.shape}  ({time.time()-t0:.1f}s)")

    # ── 6. Build hnswlib index ────────────────────────────────────
    print(f"\nBuilding HNSW index (M={M}, ef_construction={EF_CONST})...")
    t0    = time.time()
    N     = len(movie_ids)
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=N, M=M, ef_construction=EF_CONST, random_seed=42)
    index.set_ef(EF_SEARCH)
    index.add_items(embeddings, list(range(N)))
    elapsed = time.time() - t0
    print(f"  Index built in {elapsed:.1f}s ({N/elapsed:.0f} vecs/sec)")

    # ── 7. Save ───────────────────────────────────────────────────
    OUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(OUT_INDEX_PATH))
    with open(OUT_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(movie_ids, f)

    size_mb = OUT_INDEX_PATH.stat().st_size / 1_048_576
    print(f"\nSaved: {OUT_INDEX_PATH.name}  ({size_mb:.1f} MB)")
    print(f"Saved: {OUT_MAP_PATH.name}  ({N:,} entries)")

    # ── 8. Smoke test ─────────────────────────────────────────────
    print("\nSmoke test — 'indie drama hidden gem':")
    q_vec  = model.encode(["indie drama hidden gem"], normalize_embeddings=True)[0].astype(np.float32)
    labels, dists = index.knn_query([q_vec], k=5)
    for lbl, dist in zip(labels[0], dists[0]):
        mid   = movie_ids[int(lbl)]
        title = db_map[mid][1]
        print(f"  movie_id={mid:7d}  sim={1-dist:.4f}  {title}")

    print("\n" + "=" * 64)
    print("DONE — fallback index ready.")
    print("=" * 64)


if __name__ == "__main__":
    main()
