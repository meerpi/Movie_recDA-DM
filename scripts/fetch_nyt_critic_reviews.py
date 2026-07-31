#!/usr/bin/env python3
"""
scripts/fetch_nyt_critic_reviews.py

Fetches official New York Times film critic reviews for modern movies (2019-2026)
using NYT Article Search API Key: D7hUho2c3QUxUOypLPOLxNHLKgqfqeeM0M960DrGhTrJveIS

Outputs:
  dirtywork/modern_nyt_critic_reviews_2019_2026.jsonl
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
OUTPUT_JSONL = DIRTYWORK / "modern_nyt_critic_reviews_2019_2026.jsonl"

NYT_API_KEY = "D7hUho2c3QUxUOypLPOLxNHLKgqfqeeM0M960DrGhTrJveIS"
MAX_WORKERS = 3  # Rate-limited compliance for NYT API


def fetch_nyt_review_for_movie(movie: dict) -> dict:
    tmdb_id = movie.get("tmdb_id")
    imdb_id = movie.get("imdb_id")
    title   = movie.get("title", "")
    year    = movie.get("year")

    if not title:
        return {"tmdb_id": tmdb_id, "title": title, "reviews": []}

    clean_title = title.split(":")[0].split("–")[0].split("-")[0].strip()
    query_url = (
        f"https://api.nytimes.com/svc/search/v2/articlesearch.json?"
        f"q={urllib.parse.quote(f'\"{clean_title}\"')}"
        f"&fq=section_name:(\"Movies\")"
        f"&api-key={NYT_API_KEY}"
    )

    parsed_reviews = []
    for attempt in range(4):
        try:
            req = urllib.request.Request(query_url, headers={"User-Agent": "CineVault/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "OK" and data.get("response"):
                    docs = data.get("response", {}).get("docs", [])
                    for doc in docs:
                        headline_obj = doc.get("headline", {})
                        main_headline = headline_obj.get("main", "") or headline_obj.get("print_headline", "")
                        
                        # Verify headline contains title or review keyword
                        if clean_title.lower() in main_headline.lower() or clean_title.lower() in doc.get("snippet", "").lower():
                            lead_para = doc.get("lead_paragraph", "")
                            snippet = doc.get("snippet", "")
                            byline = doc.get("byline", {}).get("original", "")
                            
                            parsed_reviews.append({
                                "article_id": doc.get("_id"),
                                "headline": main_headline,
                                "critic": byline,
                                "publication_date": doc.get("pub_date"),
                                "snippet": snippet,
                                "lead_paragraph": lead_para,
                                "abstract": doc.get("abstract"),
                                "url": doc.get("web_url"),
                                "source": "The New York Times"
                            })
                    break
        except Exception as e:
            if "429" in str(e):
                time.sleep(3.0 * (attempt + 1))
            else:
                break

    return {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": title,
        "year": year,
        "reviews": parsed_reviews,
    }


def main():
    print("=" * 70)
    print("📰 CineVault NYT Film Critic Reviews Extraction Pipeline")
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
    print(f"Remaining movies to fetch NYT critic reviews for: {len(to_process):,}")

    if not to_process:
        print("✅ All movies already have NYT critic reviews processed!")
        return

    n_done = 0
    n_found = 0
    n_reviews = 0
    t0 = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_nyt_review_for_movie, m): m for m in to_process}
            for future in as_completed(futures):
                res = future.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                revs = res.get("reviews", [])
                if revs:
                    n_found += 1
                    n_reviews += len(revs)

                if n_done % 100 == 0 or n_done == len(to_process):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(
                        f"  NYT Progress [{n_done:,}/{len(to_process):,}] ({rate:.1f} movies/sec) | "
                        f"Movies with NYT reviews: {n_found:,} ({n_reviews:,} articles)",
                        flush=True
                    )

    print(f"\n🎉 DONE! Extracted {n_reviews:,} NYT critic reviews for {n_found:,} movies in {time.time()-t0:.1f}s.")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
