#!/usr/bin/env python3
"""
scripts/rebuild_tier_b_index.py — Re-embed Tier B with Enriched Profile Card Text

What this does:
  BEFORE: Tier B movies in dense.hnsw were embedded using sparse synthetic text
          (title + director + raw genome tags) → weak thematic clustering
  AFTER:  Tier B movies re-embedded using ENRICHED profile card text inherited
          from Tier A via Voyage k-NN propagation (themes, tone, moral_complexity,
          directorial_style_notes, comparable_films) → rich thematic clustering

Result: New index `dirtywork/dense_v2.hnsw` where ALL movies (Tier A + B) are
embedded from natural language profile card text with consistent Gemini vocabulary.

Token cost: ~0.63M tokens (total spend after: ~4.09M / 200M free quota = 2.0%)
"""

from __future__ import annotations
import json
import os
import time
from pathlib import Path

import hnswlib
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).parent.parent
TIER_A_CARDS  = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS  = PROJECT_ROOT / "dirtywork" / "tier_b_voyage_cards.jsonl"
OLD_INDEX     = PROJECT_ROOT / "dirtywork" / "dense.hnsw"
OLD_IDS_FILE  = PROJECT_ROOT / "dirtywork" / "dense_id_map.json"
NEW_INDEX     = PROJECT_ROOT / "dirtywork" / "dense_v2.hnsw"
NEW_IDS_FILE  = PROJECT_ROOT / "dirtywork" / "dense_v2_id_map.json"

VOYAGE_MODEL = "voyage-4-large"
BATCH_SIZE   = 128
SLEEP        = 1.05
DIM          = 1024


def build_enriched_text(card: dict) -> str:
    """
    Construct a rich natural language string from an enriched profile card.
    This is the text that gets embedded — uses Gemini-quality vocabulary
    inherited via Voyage k-NN label propagation.
    """
    parts = [card.get("title", "")]
    themes = card.get("themes", [])
    tone   = card.get("tone", [])
    moral  = card.get("moral_complexity") or ""
    style  = card.get("directorial_style_notes") or ""
    comps  = card.get("comparable_films", [])
    tagline = card.get("tagline") or ""

    if themes:  parts.append("Themes: " + ", ".join(themes) + ".")
    if tone:    parts.append("Tone: " + ", ".join(tone) + ".")
    if moral:   parts.append("Moral complexity: " + moral[:300])
    if style:   parts.append("Directorial style: " + style[:300])
    if comps:   parts.append("Comparable films: " + ", ".join(comps[:3]) + ".")
    if tagline: parts.append("Tagline: " + tagline)
    return " ".join(parts)


def load_jsonl(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                mid = rec.get("movie_id")
                if mid is not None:
                    records[int(mid)] = rec
            except Exception:
                continue
    return records


def main():
    print("=" * 64)
    print("Tier B Re-Embedding: Synthetic → Enriched Profile Card Text")
    print("=" * 64)

    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    # ── Step 1: Extract Tier A vectors from existing index (unchanged) ──
    print("\n[Step 1] Extracting Tier A vectors from dense.hnsw...")
    old_idx = hnswlib.Index(space="cosine", dim=DIM)
    old_idx.load_index(str(OLD_INDEX))
    old_ids = json.load(open(OLD_IDS_FILE))

    tier_a_mids = {int(m) for m in load_jsonl(TIER_A_CARDS).keys()}

    tier_a_labels   = []
    tier_a_vecs     = []
    tier_a_movie_ids = []

    for pos, mid in enumerate(old_ids):
        mid = int(mid)
        if mid in tier_a_mids:
            vec = old_idx.get_items([pos])[0]
            tier_a_labels.append(pos)
            tier_a_vecs.append(vec)
            tier_a_movie_ids.append(mid)

    print(f"  Tier A vectors extracted: {len(tier_a_vecs):,}")

    # ── Step 2: Embed Tier B enriched cards ──
    print("\n[Step 2] Embedding Tier B enriched profile cards via Voyage API...")
    tier_b_cards = load_jsonl(TIER_B_CARDS)
    tier_b_items = list(tier_b_cards.items())   # [(mid, card), ...]

    tier_b_movie_ids = []
    tier_b_vecs      = []

    batches = [tier_b_items[i:i+BATCH_SIZE] for i in range(0, len(tier_b_items), BATCH_SIZE)]
    t_start = time.time()

    for b_idx, batch in enumerate(batches):
        mids  = [int(item[0]) for item in batch]
        texts = [build_enriched_text(item[1]) for item in batch]
        texts = [t[:2000] for t in texts]  # cap at 2000 chars

        result = client.embed(texts, model=VOYAGE_MODEL)
        vecs   = result.embeddings

        tier_b_movie_ids.extend(mids)
        tier_b_vecs.extend(vecs)

        if (b_idx + 1) % 5 == 0 or b_idx == len(batches) - 1:
            elapsed = time.time() - t_start
            rate = len(tier_b_vecs) / elapsed
            print(
                f"  Batch {b_idx+1}/{len(batches)} | "
                f"Tier B embedded: {len(tier_b_vecs):,}/{len(tier_b_items):,} | "
                f"{rate*60:.0f}/min",
                flush=True,
            )

        if b_idx < len(batches) - 1:
            time.sleep(SLEEP)

    print(f"  Total Tier B vectors: {len(tier_b_vecs):,}")
    print(f"  Tokens used: ~{sum(len(build_enriched_text(c))//4 for c in tier_b_cards.values()):,}")

    # ── Step 3: Build new combined HNSW index ──
    print("\n[Step 3] Building new HNSW index (Tier A original + Tier B enriched)...")
    all_movie_ids = tier_a_movie_ids + tier_b_movie_ids
    all_vecs      = tier_a_vecs + tier_b_vecs

    all_vecs_np = np.array(all_vecs, dtype=np.float32)
    # Normalize for cosine
    norms = np.linalg.norm(all_vecs_np, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_vecs_np = all_vecs_np / norms

    new_idx = hnswlib.Index(space="cosine", dim=DIM)
    new_idx.init_index(max_elements=len(all_vecs_np) + 500, ef_construction=400, M=32)
    new_idx.add_items(all_vecs_np, list(range(len(all_vecs_np))))
    new_idx.set_ef(200)

    new_idx.save_index(str(NEW_INDEX))

    with open(NEW_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_movie_ids, f)

    print(f"\n  New index saved: {NEW_INDEX}")
    print(f"  ID map saved:   {NEW_IDS_FILE}")
    print(f"  Total vectors:  {new_idx.get_current_count():,}")
    print(f"    - Tier A: {len(tier_a_vecs):,} (vectors unchanged, same Gemini profile card embeddings)")
    print(f"    - Tier B: {len(tier_b_vecs):,} (re-embedded from enriched profile card text)")

    elapsed = time.time() - t_start
    print(f"\n{'='*64}")
    print(f"DONE in {elapsed:.0f}s!")
    print(f"New index ready: {NEW_INDEX}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
