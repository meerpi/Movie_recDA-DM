#!/usr/bin/env python3
"""
scripts/fetch_tmdb_keywords.py

Fetches official TMDB Keywords (themes, tropes, concepts, sub-genres) for modern movies (2019-2026)
using TMDB API's /movie/{id}/keywords endpoint with 32 parallel worker threads.

Outputs:
  dirtywork/modern_tmdb_keywords_2019_2026.jsonl
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIRTYWORK    = PROJECT_ROOT / "dirtywork"
INPUT_MOVIES = DIRTYWORK / "modern_movies_2019_2026.jsonl"
OUTPUT_JSONL = DIRTYWORK / "modern_tmdb_keywords_2019_2026.jsonl"

TMDB_API_KEY = "2cc1fd2a583e2e14f6b634fb124f9ced"
MAX_WORKERS  = 32


def fetch_tmdb_keywords_for_movie(movie: dict) -> dict:
    tmdb_id = movie.get("tmdb_id")
    title   = movie.get("title", "")
    year    = movie.get("year")

    if not tmdb_id:
        return {"tmdb_id": tmdb_id, "title": title, "keywords": []}

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/keywords?api_key={TMDB_API_KEY}"
    req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})

    keywords = []
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("keywords", [])
                keywords = [k.get("name") for k in results if k.get("name")]
    except Exception:
        pass

    return {
        "tmdb_id": tmdb_id,
        "imdb_id": movie.get("imdb_id"),
        "title": title,
        "year": year,
        "keywords": keywords,
        "keyword_count": len(keywords),
    }


def main():
    print("=" * 70)
    print("🏷️ CineVault TMDB Keywords Extraction Pipeline (2019-2026)")
    print("=" * 70)

    if not INPUT_MOVIES.exists():
        print(f"❌ Input file not found: {INPUT_MOVIES}")
        return

    movies = []
    with open(INPUT_MOVIES, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    movies.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(movies):,} modern movies to process.")

    processed_ids = set()
    if OUTPUT_JSONL.exists():
        with open(OUTPUT_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["tmdb_id"])
                    except Exception:
                        pass
    print(f"Already processed: {len(processed_ids):,} movies.")

    to_process = [m for m in movies if m.get("tmdb_id") not in processed_ids]
    print(f"Remaining movies to fetch TMDB keywords for: {len(to_process):,}")

    if not to_process:
        print("✅ All movies already have TMDB keywords processed!")
        return

    n_done = 0
    n_has_kw = 0
    total_kw_count = 0
    t0 = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_tmdb_keywords_for_movie, m): m for m in to_process}
            for future in as_completed(futures):
                res = future.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                kws = res.get("keywords", [])
                if kws:
                    n_has_kw += 1
                    total_kw_count += len(kws)

                if n_done % 500 == 0 or n_done == len(to_process):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(
                        f"  Keywords Progress [{n_done:,}/{len(to_process):,}] ({rate:.1f} movies/sec) | "
                        f"Movies with TMDB keywords: {n_has_kw:,} ({total_kw_count:,} total keywords)",
                        flush=True
                    )

    print(f"\n🎉 DONE! Extracted {total_kw_count:,} TMDB keywords for {n_has_kw:,} movies in {time.time()-t0:.1f}s.")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
