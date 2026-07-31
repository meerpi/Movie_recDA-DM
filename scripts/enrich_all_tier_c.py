#!/usr/bin/env python3
"""
scripts/enrich_all_tier_c.py — Universal 3-Level Fallback Engine for Tier C

Enriches all ~48,600 Tier C movies into full Gemini-quality profile cards
using Voyage-4-Large k-NN label propagation.

Universal 3-Level Text Fallback Hierarchy:
  Level 1: Wikipedia Plot / Intro (if available)
  Level 2: IMDb User Review Narrative (if Wikipedia plot missing)
  Level 3: Synthetic Metadata Descriptor (title, year, directors, actors, tags)

Speed: ~2,000 movies / minute (ETA: ~6-8 minutes for all 48k titles)
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

PROJECT_ROOT = Path(__file__).parent.parent
TIER_A_CARDS  = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_C_WIKI   = PROJECT_ROOT / "dirtywork" / "tier_c_wikipedia.jsonl"
IMDB_REVIEWS  = PROJECT_ROOT / "dirtywork" / "imdb_user_reviews.jsonl"
HNSW_INDEX    = PROJECT_ROOT / "dirtywork" / "dense.hnsw"
HNSW_IDS_FILE = PROJECT_ROOT / "dirtywork" / "dense_id_map.json"
OUT_TIER_C    = PROJECT_ROOT / "dirtywork" / "tier_c_voyage_cards.jsonl"

VOYAGE_MODEL  = "voyage-4-large"
BATCH_SIZE    = 128
SLEEP_BETWEEN_BATCHES = 1.05
TOP_K         = 15
MIN_COSINE    = 0.40


def load_jsonl(path: Path) -> dict:
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


def load_imdb_reviews(path: Path) -> dict:
    imdb_map = {}
    if not path.exists():
        return imdb_map
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                mid = rec.get("movie_id")
                revs = rec.get("reviews") or rec.get("review") or []
                if mid and revs:
                    if isinstance(revs, list):
                        text = " ".join([r if isinstance(r, str) else r.get("text", "") for r in revs[:4]])
                    elif isinstance(revs, str):
                        text = revs
                    else:
                        text = ""
                    if text.strip():
                        imdb_map[int(mid)] = text[:1500]
            except Exception:
                continue
    return imdb_map


def build_synthetic_descriptor(card: dict) -> str:
    title = card.get("title", "")
    year  = card.get("year", "")
    directors = ", ".join(card.get("directors", [])) if card.get("directors") else ""
    actors    = ", ".join(card.get("actors", [])[:5]) if card.get("actors") else ""
    raw_tags  = card.get("top_tags", [])
    if raw_tags and isinstance(raw_tags[0], dict):
        tags_str = ", ".join([t["tag"] for t in raw_tags[:8]])
    elif raw_tags and isinstance(raw_tags[0], str):
        tags_str = ", ".join(raw_tags[:8])
    else:
        tags_str = ""

    parts = [f"{title} ({year})"]
    if directors:
        parts.append(f"directed by {directors}")
    if actors:
        parts.append(f"starring {actors}")
    if tags_str:
        parts.append(f"Key themes and tags: {tags_str}")
    return ". ".join(parts) + "."


def aggregate_labels(neighbors: list[dict], similarities: list[float]) -> dict:
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


def main():
    print("=" * 64)
    print("Tier C Universal Coverage Voyage Label Propagation Engine")
    print("=" * 64)

    api_key = os.getenv("VOYAGE_API_KEY")
    client = voyageai.Client(api_key=api_key)

    idx = hnswlib.Index(space="cosine", dim=1024)
    idx.load_index(str(HNSW_INDEX))
    idx.set_ef(200)
    with open(HNSW_IDS_FILE) as f:
        id_list = json.load(f)

    tier_a_cards = load_jsonl(TIER_A_CARDS)
    tier_c_wiki  = load_jsonl(TIER_C_WIKI)
    imdb_reviews = load_imdb_reviews(IMDB_REVIEWS)

    # Load already completed Tier C items if any
    completed_mids = set()
    if OUT_TIER_C.exists():
        with open(OUT_TIER_C, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        completed_mids.add(int(c["movie_id"]))
                    except Exception:
                        continue

    print(f"Tier C Total Wiki Records: {len(tier_c_wiki):,}")
    print(f"Already enriched Tier C cards: {len(completed_mids):,}")

    queue = []
    source_stats = Counter()

    for mid, item in tier_c_wiki.items():
        if mid in completed_mids:
            continue
        text = item.get("plot") or item.get("intro") or ""
        source = "wikipedia"

        if not text.strip():
            text = imdb_reviews.get(mid, "")
            source = "imdb_user_review"

        if not text.strip():
            text = build_synthetic_descriptor(item)
            source = "synthetic_metadata"

        source_stats[source] += 1
        queue.append((mid, item.get("title", ""), text[:2000], source, item))

    print(f"\nRemaining Queue to Enrich: {len(queue):,}")
    for k, v in source_stats.items():
        print(f"  Source [{k}]: {v:,} movies")

    if not queue:
        print("All Tier C movies are already 100% enriched!")
        return

    batches = [queue[i:i+BATCH_SIZE] for i in range(0, len(queue), BATCH_SIZE)]
    print(f"\nProcessing {len(batches)} batches (~{len(batches)*SLEEP_BETWEEN_BATCHES/60:.1f} minutes)...")

    t_start = time.time()
    n_done = len(completed_mids)

    with open(OUT_TIER_C, "a", encoding="utf-8") as out_f:
        for b_idx, batch in enumerate(batches):
            texts = [b_item[2] for b_item in batch]
            try:
                res = client.embed(texts, model=VOYAGE_MODEL)
                vecs = np.array(res.embeddings, dtype=np.float32)
            except Exception as e:
                print(f"  [warn] Batch {b_idx} failed: {e}", flush=True)
                time.sleep(5)
                continue

            labels_arr, dists_arr = idx.knn_query(vecs, k=TOP_K)
            sims_arr = 1.0 - dists_arr

            for i, (mid, title, _, source, item) in enumerate(batch):
                n_ids = labels_arr[i]
                n_sims = sims_arr[i].tolist()
                max_sim = max(n_sims)

                n_cards = []
                for nid in n_ids:
                    if nid < len(id_list):
                        a_mid = int(id_list[nid])
                        if a_mid in tier_a_cards:
                            n_cards.append(tier_a_cards[a_mid])

                if not n_cards:
                    continue

                agg = aggregate_labels(n_cards, n_sims[:len(n_cards)])
                record = {
                    **item,
                    **agg,
                    "voyage_enriched": True,
                    "enrichment_text_source": source,
                    "nearest_tier_a_max_sim": round(max_sim, 4),
                    "low_confidence": max_sim < MIN_COSINE,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_done += 1

            out_f.flush()
            if (b_idx + 1) % 20 == 0 or b_idx == len(batches) - 1:
                elapsed = time.time() - t_start
                rate = (n_done - len(completed_mids)) / elapsed if elapsed > 0 else 0
                eta_m = (len(queue) - (n_done - len(completed_mids))) / rate / 60 if rate > 0 else 0
                print(
                    f"  Batch {b_idx+1}/{len(batches)} | Total Tier C Enriched: {n_done:,}/{len(tier_c_wiki):,} "
                    f"| Rate: {rate*60:.0f}/min | ETA: {eta_m:.1f}m",
                    flush=True,
                )

            if b_idx < len(batches) - 1:
                time.sleep(SLEEP_BETWEEN_BATCHES)

    elapsed = time.time() - t_start
    print(f"\nSUCCESS! Tier C Universal Enrichment Complete: {n_done:,} enriched in {elapsed/60:.1f} minutes!")
    print(f"Output: {OUT_TIER_C}")


if __name__ == "__main__":
    main()
