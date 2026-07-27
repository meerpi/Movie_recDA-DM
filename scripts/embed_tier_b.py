#!/usr/bin/env python3
"""
scripts/embed_tier_b.py — Embed Tier B cards with Voyage 4 and update dense.hnsw

1. Reads dirtywork/tier_b_profile_cards.jsonl (4,290 cards).
2. Embeds them using voyage-4-large into 1024-dim vectors.
3. Merges with Tier A embeddings (9,526 cards) → 13,816 total vectors.
4. Saves combined dense_id_map.json and builds combined dense.hnsw index.

Runtime: ~15-30 seconds via Voyage AI API.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np
import voyageai

PROJECT_ROOT = Path(__file__).parent.parent
TIER_B_CARDS = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"

TIER_A_NPY = PROJECT_ROOT / "dirtywork" / "tier_a_voyage_1024d.npy"
TIER_A_MAP = PROJECT_ROOT / "dirtywork" / "tier_a_voyage_id_map.json"

OUT_HNSW = PROJECT_ROOT / "dirtywork" / "dense.hnsw"
OUT_MAP  = PROJECT_ROOT / "dirtywork" / "dense_id_map.json"

MODEL = "voyage-4-large"
DIM = 1024
BATCH_SIZE = 128


def serialize_card(card: dict) -> str:
    parts = []
    title = card.get("title", "?")
    directors = card.get("directors", "")
    actors = card.get("actors", "")
    top_tags = card.get("top_tags", [])

    parts.append(title)
    if directors:
        parts.append(f"directed by {directors}")
    if actors:
        if isinstance(actors, list):
            actors = ", ".join(actors[:5])
        parts.append(f"starring {actors[:200]}")

    if top_tags:
        tag_strs = []
        for t in top_tags[:15]:
            if isinstance(t, str):
                tag_strs.append(t)
            elif isinstance(t, dict):
                tag_strs.append(t.get("tag", ""))
        if tag_strs:
            parts.append("Tags: " + ", ".join(tag_strs))

    return " ".join(parts)


def main():
    print("=" * 60)
    print("Embed Tier B Cards with Voyage 4 & Build Combined HNSW")
    print("=" * 60)

    # 1. Load Tier A embeddings
    if not TIER_A_NPY.exists() or not TIER_A_MAP.exists():
        sys.exit("ERROR: Tier A embeddings not found in dirtywork/")

    tier_a_vecs = np.load(str(TIER_A_NPY))
    with open(TIER_A_MAP) as f:
        tier_a_ids = json.load(f)

    print(f"Loaded Tier A: {len(tier_a_ids):,} vectors, shape {tier_a_vecs.shape}")

    # 2. Load Tier B cards (filtering out any already in Tier A)
    tier_a_set = set(tier_a_ids)
    tier_b_ids = []
    tier_b_texts = []

    with open(TIER_B_CARDS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            card = json.loads(line)
            mid = int(card["movie_id"])
            if mid not in tier_a_set:
                tier_b_ids.append(mid)
                tier_b_texts.append(serialize_card(card))

    print(f"Tier B movies to embed: {len(tier_b_ids):,}")

    # 3. Embed Tier B via Voyage API
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        # try loading from .env
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.environ.get("VOYAGE_API_KEY", "").strip()

    if not api_key:
        sys.exit("ERROR: VOYAGE_API_KEY not found in environment or .env")

    vo = voyageai.Client(api_key=api_key)
    print(f"Calling Voyage API ({MODEL}, batch_size={BATCH_SIZE})...")

    t0 = time.time()
    tier_b_vec_list = []
    for i in range(0, len(tier_b_texts), BATCH_SIZE):
        batch = tier_b_texts[i : i + BATCH_SIZE]
        res = vo.embed(batch, model=MODEL, input_type="document")
        tier_b_vec_list.extend(res.embeddings)
        print(f"  Embedded [{i+len(batch):,}/{len(tier_b_texts):,}]  ({time.time()-t0:.1f}s)")

    tier_b_vecs = np.array(tier_b_vec_list, dtype=np.float32)
    print(f"Tier B embedding complete: shape {tier_b_vecs.shape}")

    # 4. Combine Tier A + Tier B
    all_vecs = np.vstack([tier_a_vecs, tier_b_vecs])
    all_ids = tier_a_ids + tier_b_ids

    N, dim = all_vecs.shape
    print(f"\nCombined matrix shape: [{N:,}, {dim}]")

    # 5. Build combined HNSW index
    print(f"Building combined HNSW index (space='cosine', M=32, ef=200)...")
    index = hnswlib.Index(space="cosine", dim=DIM)
    index.init_index(max_elements=N, M=32, ef_construction=200, random_seed=42)
    index.set_ef(100)
    index.add_items(all_vecs, list(range(N)))

    # Save
    index.save_index(str(OUT_HNSW))
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    # Also copy to root/nlp for compatibility if retriever looks there
    nlp_hnsw = PROJECT_ROOT / "nlp" / "dense.hnsw"
    nlp_map = PROJECT_ROOT / "nlp" / "dense_id_map.json"
    index.save_index(str(nlp_hnsw))
    with open(nlp_map, "w", encoding="utf-8") as f:
        json.dump(all_ids, f)

    size_mb = OUT_HNSW.stat().st_size / 1_048_576
    print(f"\nSaved combined index → {OUT_HNSW.name} ({size_mb:.1f} MB)")
    print(f"Total indexed movies: {N:,} (Tier A + Tier B)")
    print("=" * 60)


if __name__ == "__main__":
    main()
