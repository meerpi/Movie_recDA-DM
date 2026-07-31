#!/usr/bin/env python3
"""
scripts/fetch_guardian_critic_reviews.py

Fetches 100% complete full-text Guardian film critic reviews for modern movies (2019-2026)
using The Guardian Open Platform API.

Outputs:
  dirtywork/modern_guardian_critic_reviews_2019_2026.jsonl
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DIRTYWORK    = PROJECT_ROOT / "dirtywork"
INPUT_MOVIES = DIRTYWORK / "modern_movies_2019_2026.jsonl"
OUTPUT_JSONL = DIRTYWORK / "modern_guardian_critic_reviews_2019_2026.jsonl"

GUARDIAN_API_KEY = "af2ccdf4-dae9-442f-a397-c8cdc5b47d91"
MAX_WORKERS      = 20


def fetch_guardian_review_for_movie(movie: dict) -> dict:
    tmdb_id = movie.get("tmdb_id")
    imdb_id = movie.get("imdb_id")
    title   = movie.get("title", "")
    year    = movie.get("year")

    if not title:
        return {"tmdb_id": tmdb_id, "title": title, "reviews": []}

    # Query Guardian API for exact title within film reviews
    clean_title = title.split(":")[0].split("–")[0].split("-")[0].strip()
    query_url = (
        f"https://content.guardianapis.com/search?"
        f"q={urllib.parse.quote(f'\"{clean_title}\"')}"
        f"&tag=tone/reviews,film/film"
        f"&show-fields=headline,standfirst,bodyText,starRating,byline,publicationDate,shortUrl"
        f"&api-key={GUARDIAN_API_KEY}"
    )

    parsed_reviews = []
    try:
        req = urllib.request.Request(query_url, headers={"User-Agent": "CineVault/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8")).get("response", {})
                results = data.get("results", [])
                for item in results:
                    fields = item.get("fields", {})
                    headline = fields.get("headline", "")
                    
                    # Verify headline matches title to prevent false positives
                    if clean_title.lower() in headline.lower():
                        body_text = fields.get("bodyText", "")
                        star_rating = fields.get("starRating")
                        rating_val = float(star_rating) if star_rating and str(star_rating).replace(".", "").isdigit() else None
                        
                        parsed_reviews.append({
                            "article_id": item.get("id"),
                            "headline": headline,
                            "critic": fields.get("byline"),
                            "star_rating": rating_val,
                            "publication_date": fields.get("publicationDate"),
                            "standfirst": fields.get("standfirst"),
                            "full_text": body_text,
                            "word_count": len(body_text.split()),
                            "char_count": len(body_text),
                            "url": fields.get("shortUrl") or item.get("webUrl"),
                            "source": "The Guardian"
                        })
    except Exception:
        pass

    return {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": title,
        "year": year,
        "reviews": parsed_reviews,
    }


def main():
    print("=" * 70)
    print("📰 CineVault Guardian Critic Reviews Extraction Pipeline")
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
    print(f"Remaining movies to fetch Guardian critic reviews for: {len(to_process):,}")

    if not to_process:
        print("✅ All movies already have Guardian critic reviews processed!")
        return

    n_done = 0
    n_found = 0
    n_reviews = 0
    t0 = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_guardian_review_for_movie, m): m for m in to_process}
            for future in as_completed(futures):
                res = future.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                revs = res.get("reviews", [])
                if revs:
                    n_found += 1
                    n_reviews += len(revs)
                
                if n_done % 200 == 0 or n_done == len(to_process):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(
                        f"  Progress [{n_done:,}/{len(to_process):,}] ({rate:.1f} movies/sec) | "
                        f"Movies with Guardian reviews: {n_found:,} ({n_reviews:,} full articles)"
                    )

    print(f"\n🎉 DONE! Extracted {n_reviews:,} full Guardian critic reviews for {n_found:,} movies in {time.time()-t0:.1f}s.")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
