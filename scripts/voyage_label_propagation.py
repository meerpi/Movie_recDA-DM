#!/usr/bin/env python3
"""
scripts/voyage_label_propagation.py — Voyage k-NN Label Propagation

For each Tier B / Tier C movie with a Wikipedia plot:
  1. Embed the plot via voyage-4-large API (batch=128, 1s sleep between batches)
  2. Search existing Tier A HNSW index (dense.hnsw) for top-K nearest neighbors
  3. Aggregate Gemini-generated labels from those neighbors:
       themes, tone, pacing, moral_complexity, directorial_style_notes,
       comparable_films, standout_performances
  4. Write enriched card to output JSONL (with resume support)

Rate limits observed:
  - Basic TPM: 3,000,000 tokens/min  → 1s sleep per batch of 128
  - Basic RPM: 2,000 req/min         → well within limit

Token budget: ~7.1M / 200M free tokens (3.5%)

Output:
  dirtywork/tier_b_voyage_cards.jsonl
  dirtywork/tier_c_voyage_cards.jsonl
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import hnswlib
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

TIER_A_CARDS  = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS  = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
TIER_B_WIKI   = PROJECT_ROOT / "dirtywork" / "tier_b_wikipedia.jsonl"
TIER_C_WIKI   = PROJECT_ROOT / "dirtywork" / "tier_c_wikipedia.jsonl"

# The existing Tier A HNSW index built by embed_tier_b.py (voyage-4-large, 1024d)
HNSW_INDEX    = PROJECT_ROOT / "dirtywork" / "dense.hnsw"
HNSW_IDS_FILE = PROJECT_ROOT / "dirtywork/dense_id_map.json"

OUT_TIER_B    = PROJECT_ROOT / "dirtywork" / "tier_b_voyage_cards.jsonl"
OUT_TIER_C    = PROJECT_ROOT / "dirtywork" / "tier_c_voyage_cards.jsonl"

VOYAGE_MODEL  = "voyage-4-large"
BATCH_SIZE    = 128      # max Voyage batch size
SLEEP_BETWEEN_BATCHES = 1.05  # seconds — keeps us safely under 3M TPM
TOP_K         = 5        # nearest Tier A neighbors to aggregate from
MIN_COSINE    = 0.60     # minimum similarity to propagate labels (flag low confidence below)

# ---------------------------------------------------------------------------
# Load data helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> dict:
    """Load JSONL into dict keyed by movie_id (int)."""
    records = {}
    if not path.exists():
        return records
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


def load_completed_ids(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["movie_id"]))
            except Exception:
                continue
    return completed


# ---------------------------------------------------------------------------
# HNSW index loader
# ---------------------------------------------------------------------------
def load_hnsw_index(index_path: Path, ids_path: Path):
    """Load existing HNSW index + id mapping. Returns (index, id_list)."""
    if not index_path.exists():
        raise FileNotFoundError(f"HNSW index not found: {index_path}")

    idx = hnswlib.Index(space="cosine", dim=1024)
    idx.load_index(str(index_path))
    idx.set_ef(200)

    if ids_path.exists():
        with open(ids_path, encoding="utf-8") as f:
            id_list = json.load(f)   # list of movie_ids in index order
    else:
        # fallback: use internal labels
        id_list = list(range(idx.get_current_count()))

    print(f"  HNSW index loaded: {idx.get_current_count():,} vectors, dim=1024")
    return idx, id_list


# ---------------------------------------------------------------------------
# Label aggregation from k-NN neighbors
# ---------------------------------------------------------------------------
def aggregate_labels(neighbors: list[dict], similarities: list[float]) -> dict:
    """
    Given k Tier A neighbors and their cosine similarities,
    return aggregated label fields (weighted by similarity).
    """
    themes_counter  = Counter()
    tones_counter   = Counter()
    pacing_counter  = Counter()
    moral_list      = []
    style_list      = []
    comparables     = []
    performances    = []

    for neighbor, sim in zip(neighbors, similarities):
        w = float(sim)
        for t in neighbor.get("themes", []):
            themes_counter[t.lower().strip()] += w
        for t in neighbor.get("tone", []):
            tones_counter[t.lower().strip()] += w
        p = neighbor.get("pacing", "")
        if p:
            pacing_counter[p.strip()] += w
        m = neighbor.get("moral_complexity", "")
        if m:
            moral_list.append((m.strip(), w))
        s = neighbor.get("directorial_style_notes", "")
        if s:
            style_list.append((s.strip(), w))
        title = neighbor.get("title", "")
        if title:
            comparables.append(title)
        for perf in neighbor.get("standout_performances", []):
            performances.append(perf)

    top_themes = [t for t, _ in themes_counter.most_common(5)]
    top_tones  = [t for t, _ in tones_counter.most_common(4)]
    top_pacing = pacing_counter.most_common(1)[0][0] if pacing_counter else None
    top_moral  = max(moral_list, key=lambda x: x[1])[0] if moral_list else None
    top_style  = max(style_list, key=lambda x: x[1])[0] if style_list else None
    # Deduplicate while preserving order
    seen = set()
    comp_dedup = []
    for c in comparables:
        if c not in seen:
            seen.add(c)
            comp_dedup.append(c)
    perf_dedup = list(dict.fromkeys(performances))[:6]

    return {
        "themes":                  top_themes,
        "tone":                    top_tones,
        "pacing":                  top_pacing,
        "moral_complexity":        top_moral,
        "directorial_style_notes": top_style,
        "comparable_films":        comp_dedup[:5],
        "standout_performances":   perf_dedup,
    }


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------
def enrich_tier(
    label: str,
    wiki_index: dict,
    base_cards: dict,
    output_path: Path,
    voyage_client,
    hnsw_idx,
    id_list: list,
    tier_a_cards: dict,
):
    completed = load_completed_ids(output_path)
    print(f"\n--- {label}: {len(completed):,} already enriched ---")

    # Build work queue
    queue = []
    for mid, wiki in wiki_index.items():
        if mid in completed:
            continue
        text = wiki.get("plot") or wiki.get("intro") or ""
        if not text.strip():
            continue
        queue.append((mid, wiki.get("title", ""), text[:2000]))  # cap at 2000 chars

    print(f"Queued: {len(queue):,} movies with plots for {label}")
    if not queue:
        print(f"All {label} movies already enriched!")
        return

    # Split into batches
    batches = [queue[i:i+BATCH_SIZE] for i in range(0, len(queue), BATCH_SIZE)]
    print(f"Total batches: {len(batches)} (batch_size={BATCH_SIZE})")
    print(f"Estimated tokens: ~{len(queue) * 400:,} | ETA: ~{len(batches) * SLEEP_BETWEEN_BATCHES / 60:.1f}m")

    t_start = time.time()
    n_done  = 0
    n_low   = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for batch_idx, batch in enumerate(batches):
            mids  = [item[0] for item in batch]
            titles = [item[1] for item in batch]
            texts = [item[2] for item in batch]

            # Embed batch via Voyage API
            try:
                result = voyage_client.embed(texts, model=VOYAGE_MODEL)
                vecs = np.array(result.embeddings, dtype=np.float32)
            except Exception as e:
                print(f"  [warn] Batch {batch_idx} embed failed: {e}", flush=True)
                time.sleep(5)
                continue

            # HNSW search for each vector
            labels_arr, distances_arr = hnsw_idx.knn_query(vecs, k=min(TOP_K, hnsw_idx.get_current_count()))
            # hnswlib returns cosine distances (1 - cosine similarity)
            similarities_arr = 1.0 - distances_arr

            for i, (mid, title) in enumerate(zip(mids, titles)):
                neighbor_ids   = labels_arr[i]
                neighbor_sims  = similarities_arr[i].tolist()
                max_sim        = max(neighbor_sims)

                # Resolve neighbor movie_ids → Tier A cards
                neighbor_cards = []
                for nid in neighbor_ids:
                    if nid < len(id_list):
                        a_mid = int(id_list[nid])
                        if a_mid in tier_a_cards:
                            neighbor_cards.append(tier_a_cards[a_mid])

                if not neighbor_cards:
                    continue

                agg = aggregate_labels(neighbor_cards, neighbor_sims[:len(neighbor_cards)])

                base = base_cards.get(mid, {"movie_id": mid, "title": title})
                record = {
                    **base,
                    **agg,
                    "voyage_enriched":       True,
                    "nearest_tier_a_max_sim": round(max_sim, 4),
                    "low_confidence":        max_sim < MIN_COSINE,
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_done += 1
                if max_sim < MIN_COSINE:
                    n_low += 1

            out_f.flush()

            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0
            eta_m = (len(queue) - n_done) / rate / 60 if rate > 0 else 0

            print(
                f"  [{label}] batch {batch_idx+1}/{len(batches)} | "
                f"{n_done:,}/{len(queue):,} done | "
                f"{rate*60:.0f}/min | ETA={eta_m:.1f}m | low_conf={n_low}",
                flush=True,
            )

            # Rate limit: sleep between batches to stay under 3M TPM
            if batch_idx < len(batches) - 1:
                time.sleep(SLEEP_BETWEEN_BATCHES)

    elapsed = time.time() - t_start
    print(
        f"\nDONE — {label}: {n_done:,} enriched in {elapsed:.0f}s "
        f"({n_done/elapsed*60:.0f} movies/min) | low_confidence={n_low}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Voyage k-NN Label Propagation Pipeline")
    print(f"Model : {VOYAGE_MODEL}  |  Batch: {BATCH_SIZE}  |  k={TOP_K}")
    print("=" * 64)

    # Init Voyage client
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY not set in .env")
    voyage_client = voyageai.Client(api_key=api_key)
    print(f"Voyage client ready (key: {api_key[:15]}...)")

    # Load HNSW index
    print("\n[Step 1] Loading Tier A HNSW index...")
    hnsw_idx, id_list = load_hnsw_index(HNSW_INDEX, HNSW_IDS_FILE)

    # Load Tier A cards (source of labels)
    print("[Step 2] Loading Tier A profile cards (label source)...")
    tier_a_cards = load_jsonl(TIER_A_CARDS)
    # Only keep Tier A cards (not Tier B in the same file if any)
    print(f"  Tier A cards loaded: {len(tier_a_cards):,}")

    # Load Tier B
    print("\n[Step 3] Loading Tier B data...")
    tier_b_wiki  = load_jsonl(TIER_B_WIKI)
    tier_b_cards = load_jsonl(TIER_B_CARDS)
    print(f"  Wiki plots: {len(tier_b_wiki):,}  |  Base cards: {len(tier_b_cards):,}")

    # Enrich Tier B
    print("\n[Step 4] Enriching Tier B via Voyage k-NN propagation...")
    enrich_tier(
        label="Tier B", wiki_index=tier_b_wiki, base_cards=tier_b_cards,
        output_path=OUT_TIER_B, voyage_client=voyage_client,
        hnsw_idx=hnsw_idx, id_list=id_list, tier_a_cards=tier_a_cards,
    )

    # Load Tier C
    print("\n[Step 5] Loading Tier C Wikipedia plots...")
    tier_c_wiki = load_jsonl(TIER_C_WIKI)
    print(f"  Wiki plots: {len(tier_c_wiki):,}")

    # Enrich Tier C
    print("\n[Step 6] Enriching Tier C via Voyage k-NN propagation...")
    enrich_tier(
        label="Tier C", wiki_index=tier_c_wiki, base_cards={},
        output_path=OUT_TIER_C, voyage_client=voyage_client,
        hnsw_idx=hnsw_idx, id_list=id_list, tier_a_cards=tier_a_cards,
    )

    print("\n" + "=" * 64)
    print("Voyage Label Propagation Complete!")
    print(f"  Tier B → {OUT_TIER_B}")
    print(f"  Tier C → {OUT_TIER_C}")
    print("=" * 64)


if __name__ == "__main__":
    main()
