#!/usr/bin/env python3
"""
scripts/extract_tier_c_nli.py — Fast GPU Zero-Shot Categorical Extraction for Tier C Movies

Uses MoritzLaurer/ModernBERT-large-zeroshot-v2.0 on CUDA GPU to extract:
  - pacing
  - tone
  - themes

Reads scraped Wikipedia plot summaries from dirtywork/tier_c_wikipedia.jsonl
and outputs enriched records to dirtywork/tier_c_profile_cards_v2.jsonl.

Speed: ~100-150 movies / minute on GPU.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH   = PROJECT_ROOT / "dirtywork" / "tier_c_wikipedia.jsonl"
CARDS_PATH   = PROJECT_ROOT / "dirtywork" / "tier_c_profile_cards.jsonl"
OUTPUT_PATH  = PROJECT_ROOT / "dirtywork" / "tier_c_profile_cards_v2.jsonl"

MODEL_ID     = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"

PACING_LABELS = ["slow burn", "fast-paced", "moderate pacing"]

TONE_LABELS = [
    "atmospheric and spooky",
    "dark and gritty",
    "lighthearted comedy",
    "intense thriller",
    "somber drama",
    "whimsical and surreal",
]

THEME_LABELS = [
    "grief and trauma",
    "existential dread",
    "paranoia and conspiracy",
    "coming of age",
    "revenge and justice",
    "survival and isolation",
    "love and obsession",
]


def load_completed_ids(out_path: Path) -> set:
    completed = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed.add(json.loads(line)["movie_id"])
                    except Exception:
                        pass
    return completed


def load_base_cards(cards_path: Path) -> dict:
    cards = {}
    if cards_path.exists():
        with open(cards_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        cards[c["movie_id"]] = c
                    except Exception:
                        pass
    return cards


def main():
    print("=" * 64)
    print(f"Tier C NLI Categorical Extraction ({MODEL_ID})")
    print("=" * 64)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading Zero-Shot NLI Pipeline on device={'CUDA (GPU 0)' if device == 0 else 'CPU'}...")
    t0 = time.time()
    classifier = pipeline("zero-shot-classification", model=MODEL_ID, device=device)
    print(f"Pipeline ready in {time.time()-t0:.1f}s")

    completed_ids = load_completed_ids(OUTPUT_PATH)
    base_cards    = load_base_cards(CARDS_PATH)
    print(f"Loaded {len(completed_ids):,} completed cards.")

    records_to_process = []
    if INPUT_PATH.exists():
        with open(INPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                mid = rec["movie_id"]
                plot = rec.get("plot") or rec.get("intro") or ""
                if mid not in completed_ids and plot:
                    records_to_process.append({"movie_id": mid, "title": rec["title"], "plot": plot[:1000]})

    print(f"Total Tier C movies with plot summaries to process: {len(records_to_process):,}")

    t_start = time.time()
    n_done  = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_file:
        for i, item in enumerate(records_to_process, 1):
            mid   = item["movie_id"]
            title = item["title"]
            text  = item["plot"]

            try:
                # 1. Pacing (single label classification)
                res_p = classifier(text, candidate_labels=PACING_LABELS, multi_label=False)
                top_pacing = res_p["labels"][0]

                # 2. Tone (multi-label)
                res_t = classifier(text, candidate_labels=TONE_LABELS, multi_label=True)
                top_tones = [lbl for lbl, score in zip(res_t["labels"], res_t["scores"]) if score >= 0.4][:3]

                # 3. Themes (multi-label)
                res_th = classifier(text, candidate_labels=THEME_LABELS, multi_label=True)
                top_themes = [lbl for lbl, score in zip(res_th["labels"], res_th["scores"]) if score >= 0.4][:3]

                base_card = base_cards.get(mid, {"movie_id": mid, "title": title, "tier": "C"})
                enriched = {
                    **base_card,
                    "pacing": top_pacing,
                    "tone":   top_tones,
                    "themes": top_themes,
                }

                out_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                out_file.flush()
                n_done += 1

                if i % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = i / elapsed
                    eta_m = (len(records_to_process) - i) / rate / 60
                    print(
                        f"  [{i:,}/{len(records_to_process):,}] "
                        f"rate={rate:.1f} movies/sec ({rate*60:.0f}/min) "
                        f"ETA={eta_m:.1f}m",
                        flush=True,
                    )

            except Exception as e:
                print(f"  [warn] Failed for movie_id={mid}: {e}", flush=True)

    print("\n" + "=" * 64)
    print(f"DONE — Processed {n_done:,} Tier C movies in {time.time()-t_start:.1f}s")
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
