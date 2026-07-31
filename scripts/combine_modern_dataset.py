#!/usr/bin/env python3
"""
scripts/combine_modern_dataset.py

Combines all 6 modern movie data sources (2019-2026) into a single, clean,
unified dataset file:
  dirtywork/modern_movies_combined_2019_2026.jsonl

Inputs:
  1. dirtywork/modern_movies_2019_2026.jsonl                 (Core Metadata)
  2. dirtywork/modern_wikipedia_2019_2026.jsonl              (Wikipedia Plot)
  3. dirtywork/modern_imdb_reviews_2019_2026.jsonl           (IMDb Top User Reviews)
  4. dirtywork/modern_guardian_critic_reviews_2019_2026.jsonl (Guardian Critic Reviews)
  5. dirtywork/modern_empire_critic_reviews_2019_2026.jsonl   (Empire Critic Reviews)
  6. dirtywork/modern_tmdb_keywords_2019_2026.jsonl          (TMDB Keywords & Tropes)
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIRTYWORK    = PROJECT_ROOT / "dirtywork"

FILE_MOVIES   = DIRTYWORK / "modern_movies_2019_2026.jsonl"
FILE_WIKI     = DIRTYWORK / "modern_wikipedia_2019_2026.jsonl"
FILE_IMDB     = DIRTYWORK / "modern_imdb_reviews_2019_2026.jsonl"
FILE_GUARDIAN = DIRTYWORK / "modern_guardian_critic_reviews_2019_2026.jsonl"
FILE_EMPIRE   = DIRTYWORK / "modern_empire_critic_reviews_2019_2026.jsonl"
FILE_KEYWORDS = DIRTYWORK / "modern_tmdb_keywords_2019_2026.jsonl"

OUTPUT_COMBINED = DIRTYWORK / "modern_movies_combined_2019_2026.jsonl"


def load_jsonl_by_tmdb_id(path: Path) -> dict:
    data = {}
    if not path.exists():
        print(f"⚠️ Warning: File not found {path.name}")
        return data

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    tid = obj.get("tmdb_id")
                    if tid is not None:
                        data[tid] = obj
                except Exception:
                    pass
    return data


def main():
    print("=" * 70)
    print("📦 CineVault Modern Dataset Consolidation Pipeline (2019-2026)")
    print("=" * 70)

    # 1. Load core metadata
    print("Loading 1/6: Core metadata (modern_movies_2019_2026.jsonl)...")
    base_movies = load_jsonl_by_tmdb_id(FILE_MOVIES)
    print(f"  -> Loaded {len(base_movies):,} base movies.")

    # 2. Load Wikipedia plots
    print("Loading 2/6: Wikipedia plots (modern_wikipedia_2019_2026.jsonl)...")
    wiki_data = load_jsonl_by_tmdb_id(FILE_WIKI)
    print(f"  -> Loaded Wikipedia plots for {len(wiki_data):,} movies.")

    # 3. Load IMDb user reviews
    print("Loading 3/6: IMDb user reviews (modern_imdb_reviews_2019_2026.jsonl)...")
    imdb_data = load_jsonl_by_tmdb_id(FILE_IMDB)
    print(f"  -> Loaded IMDb reviews for {len(imdb_data):,} movies.")

    # 4. Load Guardian critic reviews
    print("Loading 4/6: Guardian critic reviews (modern_guardian_critic_reviews_2019_2026.jsonl)...")
    guardian_data = load_jsonl_by_tmdb_id(FILE_GUARDIAN)
    print(f"  -> Loaded Guardian reviews for {len(guardian_data):,} movies.")

    # 5. Load Empire critic reviews
    print("Loading 5/6: Empire critic reviews (modern_empire_critic_reviews_2019_2026.jsonl)...")
    empire_data = load_jsonl_by_tmdb_id(FILE_EMPIRE)
    print(f"  -> Loaded Empire reviews for {len(empire_data):,} movies.")

    # 6. Load TMDB keywords
    print("Loading 6/6: TMDB keywords (modern_tmdb_keywords_2019_2026.jsonl)...")
    keywords_data = load_jsonl_by_tmdb_id(FILE_KEYWORDS)
    print(f"  -> Loaded TMDB keywords for {len(keywords_data):,} movies.")

    print("\nConsolidating into unified records...")
    combined_records = []
    
    n_wiki = 0
    n_imdb = 0
    n_guardian = 0
    n_empire = 0
    n_keywords = 0

    for tid, base in base_movies.items():
        rec = dict(base)  # copy base metadata

        # Attach Wikipedia
        w = wiki_data.get(tid, {})
        intro_text = w.get("intro") or w.get("intro_summary") or ""
        plot_text = w.get("plot") or w.get("plot_summary") or ""
        rec["wikipedia"] = {
            "wiki_url": w.get("wikipedia_page") or w.get("wiki_url") or "",
            "intro_summary": intro_text,
            "plot_summary": plot_text
        }
        if intro_text or plot_text:
            n_wiki += 1

        # Attach IMDb top-voted user reviews
        im = imdb_data.get(tid, {})
        rec["imdb_user_reviews"] = im.get("reviews", [])
        if rec["imdb_user_reviews"]:
            n_imdb += 1

        # Attach Guardian critic reviews
        g = guardian_data.get(tid, {})
        rec["guardian_critic_reviews"] = g.get("reviews", [])
        if rec["guardian_critic_reviews"]:
            n_guardian += 1

        # Attach Empire critic reviews
        emp = empire_data.get(tid, {})
        rec["empire_critic_reviews"] = emp.get("reviews", [])
        if rec["empire_critic_reviews"]:
            n_empire += 1

        # Attach TMDB keywords
        kw = keywords_data.get(tid, {})
        rec["tmdb_keywords"] = kw.get("keywords", [])
        if rec["tmdb_keywords"]:
            n_keywords += 1

        combined_records.append(rec)

    print(f"Writing {len(combined_records):,} unified movie records to {OUTPUT_COMBINED.name}...")
    with open(OUTPUT_COMBINED, "w", encoding="utf-8") as f:
        for rec in combined_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_size_mb = OUTPUT_COMBINED.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"🎉 CONSOLIDATION COMPLETE! Created {OUTPUT_COMBINED.name} ({out_size_mb:.2f} MB)")
    print(f"  • Total Movies:                   {len(combined_records):,}")
    print(f"  • Movies with Wikipedia Plots:     {n_wiki:,}")
    print(f"  • Movies with IMDb User Reviews:   {n_imdb:,}")
    print(f"  • Movies with Guardian Reviews:    {n_guardian:,}")
    print(f"  • Movies with Empire Reviews:      {n_empire:,}")
    print(f"  • Movies with TMDB Keywords:       {n_keywords:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
