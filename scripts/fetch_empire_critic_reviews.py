#!/usr/bin/env python3
"""
scripts/fetch_empire_critic_reviews.py

Extracts 100% complete full-text Empire Magazine film critic reviews (no paywall)
for modern movies (2019-2026).

Outputs:
  dirtywork/modern_empire_critic_reviews_2019_2026.jsonl
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from curl_cffi import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
DIRTYWORK    = PROJECT_ROOT / "dirtywork"
INPUT_MOVIES = DIRTYWORK / "modern_movies_2019_2026.jsonl"
OUTPUT_JSONL = DIRTYWORK / "modern_empire_critic_reviews_2019_2026.jsonl"

MAX_WORKERS  = 64


def slugify_title(title: str) -> str:
    title = title.split(":")[0].split("–")[0].split("-")[0].strip().lower()
    title = re.sub(r"[^\w\s-]", "", title)
    title = re.sub(r"[\s_-]+", "-", title).strip("-")
    return title


def fetch_empire_review_for_movie(movie: dict) -> dict:
    tmdb_id = movie.get("tmdb_id")
    imdb_id = movie.get("imdb_id")
    title   = movie.get("title", "")
    year    = movie.get("year")

    if not title:
        return {"tmdb_id": tmdb_id, "title": title, "reviews": []}

    slug = slugify_title(title)
    candidate_urls = [
        f"https://www.empireonline.com/movies/reviews/{slug}/",
        f"https://www.empireonline.com/movies/reviews/{slug}-{year}/",
    ]

    parsed_reviews = []
    for url in candidate_urls:
        try:
            resp = requests.get(url, impersonate="chrome120", timeout=6)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                h1 = soup.find("h1")
                headline = h1.text.strip() if h1 else f"{title} Review"
                
                # Check review body paragraphs
                paras = []
                for p in soup.find_all("p"):
                    t = p.text.strip()
                    if len(t) > 60 and "H Bauer Publishing" not in t and "Cookie Policy" not in t:
                        paras.append(t)
                
                full_text = "\n\n".join(paras)
                if len(full_text) > 300:
                    parsed_reviews.append({
                        "headline": headline,
                        "url": url,
                        "full_text": full_text,
                        "word_count": len(full_text.split()),
                        "char_count": len(full_text),
                        "source": "Empire Magazine"
                    })
                    break
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
    print("🎬 CineVault Empire Magazine 100% Full-Text Critic Reviews Pipeline")
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
    print(f"Remaining movies to fetch Empire full critic reviews for: {len(to_process):,}")

    if not to_process:
        print("✅ All movies already have Empire critic reviews processed!")
        return

    n_done = 0
    n_found = 0
    n_reviews = 0
    t0 = time.time()

    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(fetch_empire_review_for_movie, m): m for m in to_process}
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
                        f"Movies with Empire 100% Full Reviews: {n_found:,} ({n_reviews:,} articles)",
                        flush=True
                    )

    print(f"\n🎉 DONE! Extracted {n_reviews:,} full Empire critic reviews for {n_found:,} movies in {time.time()-t0:.1f}s.")
    print(f"Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
