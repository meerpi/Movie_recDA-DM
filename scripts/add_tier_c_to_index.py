#!/usr/bin/env python3
"""
scripts/add_tier_c_to_index.py — Add enriched Tier C to HNSW semantic index

Adds the 6,410 Tier C movies (those with Wikipedia plots, enriched via Voyage k-NN)
to dense_v2.hnsw alongside the existing 13,775 Tier A + Tier B vectors.

Output: dense_v2.hnsw updated to 20,185 total vectors
Token cost: ~1.0M (total spend: ~6.05M / 200M free = 3.0%)
"""

from __future__ import annotations
import json, os, time
from pathlib import Path
import hnswlib
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
TIER_C_FILE  = PROJECT_ROOT / "dirtywork" / "tier_c_voyage_cards.jsonl"
V2_INDEX     = PROJECT_ROOT / "dirtywork" / "dense_v2.hnsw"
V2_IDS_FILE  = PROJECT_ROOT / "dirtywork" / "dense_v2_id_map.json"

VOYAGE_MODEL = "voyage-4-large"
BATCH_SIZE   = 128
SLEEP        = 1.05
DIM          = 1024


def build_enriched_text(card: dict) -> str:
    parts = [card.get("title", "")]
    if card.get("themes"):               parts.append("Themes: " + ", ".join(card["themes"]) + ".")
    if card.get("tone"):                 parts.append("Tone: " + ", ".join(card["tone"]) + ".")
    if card.get("moral_complexity"):     parts.append("Moral complexity: " + card["moral_complexity"][:300])
    if card.get("directorial_style_notes"): parts.append("Directorial style: " + card["directorial_style_notes"][:300])
    if card.get("comparable_films"):     parts.append("Comparable films: " + ", ".join(card["comparable_films"][:3]) + ".")
    if card.get("tagline"):              parts.append("Tagline: " + card["tagline"])
    return " ".join(parts)


def main():
    print("=" * 64)
    print("Adding Tier C (wiki-enriched) to dense_v2.hnsw")
    print("=" * 64)

    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # ── Step 1: Load existing index and extract all vectors ──
    print("\n[Step 1] Loading existing dense_v2.hnsw (Tier A + Tier B)...")
    old_idx = hnswlib.Index(space="cosine", dim=DIM)
    old_idx.load_index(str(V2_INDEX))
    existing_ids = json.load(open(V2_IDS_FILE))
    n_existing = old_idx.get_current_count()

    # Extract existing vectors
    existing_vecs = old_idx.get_items(list(range(n_existing)))
    print(f"  Extracted {n_existing:,} existing vectors (Tier A + Tier B)")

    # ── Step 2: Load and embed Tier C enriched cards ──
    print("\n[Step 2] Embedding Tier C enriched profile cards...")
    tier_c_cards = [json.loads(l) for l in open(TIER_C_FILE) if l.strip()]
    print(f"  Tier C cards to embed: {len(tier_c_cards):,}")

    tier_c_mids = []
    tier_c_vecs = []

    batches = [tier_c_cards[i:i+BATCH_SIZE] for i in range(0, len(tier_c_cards), BATCH_SIZE)]
    t_start = time.time()

    for b_idx, batch in enumerate(batches):
        mids  = [int(c["movie_id"]) for c in batch]
        texts = [build_enriched_text(c)[:2000] for c in batch]

        result = client.embed(texts, model=VOYAGE_MODEL)
        vecs   = result.embeddings

        tier_c_mids.extend(mids)
        tier_c_vecs.extend(vecs)

        if (b_idx + 1) % 5 == 0 or b_idx == len(batches) - 1:
            elapsed = time.time() - t_start
            rate = len(tier_c_vecs) / elapsed
            print(
                f"  Batch {b_idx+1}/{len(batches)} | "
                f"Embedded: {len(tier_c_vecs):,}/{len(tier_c_cards):,} | "
                f"{rate*60:.0f}/min",
                flush=True,
            )

        if b_idx < len(batches) - 1:
            time.sleep(SLEEP)

    # ── Step 3: Build new combined index ──
    print("\n[Step 3] Building new combined index (A + B + C)...")
    all_ids  = existing_ids + tier_c_mids
    all_vecs = list(existing_vecs) + tier_c_vecs

    all_vecs_np = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(all_vecs_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_vecs_np = all_vecs_np / norms

    new_idx = hnswlib.Index(space="cosine", dim=DIM)
    new_idx.init_index(max_elements=len(all_vecs_np) + 500, ef_construction=400, M=32)
    new_idx.add_items(all_vecs_np, list(range(len(all_vecs_np))))
    new_idx.set_ef(200)

    new_idx.save_index(str(V2_INDEX))
    with open(V2_IDS_FILE, "w") as f:
        json.dump(all_ids, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"dense_v2.hnsw UPDATED — {new_idx.get_current_count():,} total vectors")
    print(f"  Tier A: 9,526 | Tier B: 4,249 | Tier C: {len(tier_c_vecs):,}")
    print(f"Completed in {elapsed:.0f}s | Tokens used: ~{sum(len(build_enriched_text(c))//4 for c in tier_c_cards):,}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
