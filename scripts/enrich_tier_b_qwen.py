#!/usr/bin/env python3
"""
scripts/enrich_tier_b_qwen.py — Local Tier B Card Enrichment using Qwen 2.5 7B GGUF

Reads Tier B cards from dirtywork/tier_b_profile_cards.jsonl, extracts:
  - themes: list[str]
  - tone: list[str]
  - pacing: str
  - directorial_style_notes: str
  - moral_complexity: str
  - comparable_films: list[str]
  - standout_performances: list[str]

Saves enriched records to dirtywork/tier_b_profile_cards_v2.jsonl.
Supports resuming from partial runs seamlessly.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_PATH   = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
OUTPUT_PATH  = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards_v2.jsonl"
MODEL_PATH   = PROJECT_ROOT / "model" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"

SYSTEM_PROMPT = """You are an expert film analyst. Given a movie's title, year, directors, actors, and crowdsourced tags, extract structured JSON metadata.

Return ONLY a JSON object with these exact keys:
{
  "themes": ["theme 1", "theme 2", "theme 3"],
  "tone": ["tone 1", "tone 2"],
  "pacing": "slow burn" or "fast-paced" or "moderate",
  "directorial_style_notes": "A brief 1-2 sentence note on visual and narrative style.",
  "moral_complexity": "A brief 1-2 sentence note on moral/ethical ambiguity.",
  "comparable_films": ["Film Title 1", "Film Title 2"],
  "standout_performances": ["Actor Name 1", "Actor Name 2"]
}"""


def load_wikipedia_plots(wiki_path: Path) -> Dict[int, str]:
    plots: Dict[int, str] = {}
    if wiki_path.exists():
        with open(wiki_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        mid = rec.get("movie_id")
                        plot = rec.get("plot") or rec.get("intro") or ""
                        if mid and plot:
                            plots[mid] = plot[:1500]  # truncate to 1500 chars max
                    except Exception:
                        continue
    return plots


def build_user_prompt(card: dict, wiki_plots: Dict[int, str]) -> str:
    title     = card.get("title", "?")
    directors = card.get("directors", [])
    actors    = card.get("actors", [])
    top_tags  = card.get("top_tags", [])
    mid       = card.get("movie_id")

    dir_str = ", ".join(directors) if isinstance(directors, list) else str(directors)
    act_str = ", ".join(actors[:5]) if isinstance(actors, list) else str(actors)

    tag_strs = []
    for t in top_tags[:15]:
        if isinstance(t, str):
            tag_strs.append(t)
        elif isinstance(t, dict):
            tag_strs.append(t.get("tag", ""))

    plot_text = wiki_plots.get(mid, "")

    parts = [
        f"Movie: {title}",
        f"Directors: {dir_str}",
        f"Key Actors: {act_str}",
        f"Crowdsourced Tags: {', '.join(tag_strs)}",
    ]
    if plot_text:
        parts.append(f"Plot Summary: {plot_text}")

    parts.append("\nExtract structured JSON metadata:")
    return "\n".join(parts)


def load_completed_ids(output_path: Path) -> set:
    completed = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        completed.add(c["movie_id"])
                    except Exception:
                        continue
    return completed


def main() -> None:
    print("=" * 64)
    print("Enrich Tier B Profile Cards via Qwen 2.5 7B GGUF")
    print("=" * 64)

    if not MODEL_PATH.exists():
        sys.exit(f"ERROR: Model file not found at {MODEL_PATH}.\nWait for background download task to finish.")

    if not INPUT_PATH.exists():
        sys.exit(f"ERROR: Input file not found at {INPUT_PATH}")

    completed_ids = load_completed_ids(OUTPUT_PATH)
    print(f"Loaded {len(completed_ids):,} already enriched cards from {OUTPUT_PATH.name}")

    # Load llama-cpp model on GPU
    from llama_cpp import Llama

    print(f"\nLoading Qwen 2.5 7B GGUF on GPU ({MODEL_PATH.name})...")
    t0 = time.time()
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,      # Offload all layers to CUDA GPU
        n_ctx=2048,
        verbose=False,
    )
    print(f"Model loaded on GPU in {time.time() - t0:.2f}s")

    # Process cards
    cards_to_process = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            card = json.loads(line)
            mid  = int(card["movie_id"])
            if mid not in completed_ids:
                cards_to_process.append(card)

    print(f"Remaining cards to process: {len(cards_to_process):,}")

    wiki_path = PROJECT_ROOT / "dirtywork" / "tier_b_wikipedia.jsonl"
    wiki_plots = load_wikipedia_plots(wiki_path)
    print(f"Loaded {len(wiki_plots):,} Wikipedia plot summaries for enrichment context.")

    t_start = time.time()
    n_done  = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        for idx, card in enumerate(cards_to_process, 1):
            mid = card["movie_id"]
            user_prompt = build_user_prompt(card, wiki_plots)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            try:
                res = llm.create_chat_completion(
                    messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=320,
                )
                raw_json = res["choices"][0]["message"]["content"]
                extracted = json.loads(raw_json)

                # Merge extracted fields into original card
                enriched = {
                    **card,
                    "themes":                  extracted.get("themes", []),
                    "tone":                    extracted.get("tone", []),
                    "pacing":                  extracted.get("pacing", "moderate"),
                    "directorial_style_notes": extracted.get("directorial_style_notes", ""),
                    "moral_complexity":        extracted.get("moral_complexity", ""),
                    "comparable_films":        extracted.get("comparable_films", []),
                    "standout_performances":    extracted.get("standout_performances", []),
                }

                out_f.write(json.dumps(enriched) + "\n")
                out_f.flush()
                n_done += 1

                if idx % 5 == 0:
                    elapsed = time.time() - t_start
                    rate    = idx / elapsed
                    eta_m   = (len(cards_to_process) - idx) / rate / 60
                    print(
                        f"  [Qwen 7B] [{idx:,}/{len(cards_to_process):,}] "
                        f"rate={rate:.2f} cards/sec ({rate*60:.1f}/min)  ETA={eta_m:.1f}m",
                        flush=True,
                    )

            except Exception as e:
                print(f"  [warn] Failed for movie_id={mid}: {e}", flush=True)

    print("\n" + "=" * 64)
    print(f"COMPLETE — Enriched {n_done:,} Tier B cards in {time.time() - t_start:.1f}s")
    print(f"Saved: {OUTPUT_PATH}")
    print("=" * 64)


if __name__ == "__main__":
    main()
