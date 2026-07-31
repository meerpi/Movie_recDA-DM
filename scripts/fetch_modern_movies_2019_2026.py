#!/usr/bin/env python3
"""
scripts/fetch_modern_movies_2019_2026.py

Fetches 1,000 most popular movies per year for 2019-2026 from TMDB API (8,000 movies),
scrapes top 5 voted IMDb user reviews, fetches Wikipedia plot summaries, and updates
CineVault database.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"
DIRTYWORK    = PROJECT_ROOT / "dirtywork"

TMDB_API_KEY = "2cc1fd2a583e2e14f6b634fb124f9ced"
YEARS        = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
PAGES_PER_YEAR = 50  # 50 pages * 20 movies = 1,000 movies per year (8,000 total)

OUT_MOVIES_JSONL = DIRTYWORK / "modern_movies_2019_2026.jsonl"
OUT_REVIEWS_JSONL = DIRTYWORK / "modern_imdb_reviews_2019_2026.jsonl"
OUT_WIKI_JSONL    = DIRTYWORK / "modern_wikipedia_2019_2026.jsonl"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


# ── STAGE 1: Fetch Top 1,000 Movies per Year from TMDB ─────────────────────

def fetch_tmdb_page(year: int, page: int) -> list[dict]:
    url = (
        f"https://api.themoviedb.org/3/discover/movie?"
        f"api_key={TMDB_API_KEY}&sort_by=popularity.desc&primary_release_year={year}&page={page}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception as e:
        print(f"    ⚠️ Failed page {page} for year {year}: {e}")
        return []


def fetch_tmdb_details(tmdb_id: int) -> dict | None:
    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}?"
        f"api_key={TMDB_API_KEY}&append_to_response=credits,release_dates"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
            directors = [
                c.get("name") for c in data.get("credits", {}).get("crew", [])
                if c.get("job") == "Director" and c.get("name")
            ]
            cast = [
                c.get("name") for c in data.get("credits", {}).get("cast", [])[:5]
                if c.get("name")
            ]
            genres = [g.get("name") for g in data.get("genres", []) if g.get("name")]
            
            mpaa = ""
            for res in data.get("release_dates", {}).get("results", []):
                if res.get("iso_3166_1") == "US":
                    for rd in res.get("release_dates", []):
                        cert = rd.get("certification")
                        if cert:
                            mpaa = cert
                            break
            
            rel_date = data.get("release_date") or ""
            year = int(rel_date[:4]) if len(rel_date) >= 4 and rel_date[:4].isdigit() else None
            
            imdb_id = data.get("imdb_id") or ""
            if imdb_id.startswith("tt"):
                imdb_id = imdb_id[2:]
            
            return {
                "tmdb_id": tmdb_id,
                "imdb_id": imdb_id.zfill(7) if imdb_id else "",
                "title": data.get("title", ""),
                "year": year,
                "overview": data.get("overview", ""),
                "genres": genres,
                "directors": directors,
                "actors": cast,
                "content_rating": mpaa,
                "original_language": data.get("original_language", ""),
                "production_countries": [c.get("name") for c in data.get("production_countries", [])],
                "popularity": data.get("popularity", 0.0),
                "vote_average": data.get("vote_average", 0.0),
                "vote_count": data.get("vote_count", 0),
                "poster_path": data.get("poster_path"),
                "backdrop_path": data.get("backdrop_path"),
            }
    except Exception:
        return None


def run_stage_1_tmdb():
    print("=" * 65)
    print("🎬 STAGE 1: Fetching 1,000 Most Popular Movies per Year (2019-2026)")
    print("=" * 65)

    existing_tmdb_ids = set()
    if OUT_MOVIES_JSONL.exists():
        with open(OUT_MOVIES_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_tmdb_ids.add(json.loads(line)["tmdb_id"])
                    except Exception:
                        pass
    print(f"Existing movies in JSONL: {len(existing_tmdb_ids):,}")

    all_movies = []
    for year in YEARS:
        print(f"\n🗓️ Year {year}: Fetching top 1,000 movies from TMDB API...")
        year_tmdb_ids = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_tmdb_page, year, page) for page in range(1, PAGES_PER_YEAR + 1)]
            for future in as_completed(futures):
                results = future.result()
                for item in results:
                    t_id = item.get("id")
                    if t_id and t_id not in existing_tmdb_ids:
                        existing_tmdb_ids.add(t_id)
                        year_tmdb_ids.append(t_id)

        print(f"  -> Discovered {len(year_tmdb_ids):,} new movie IDs for {year}. Fetching details...")

        n_done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(fetch_tmdb_details, tid) for tid in year_tmdb_ids]
            for future in as_completed(futures):
                res = future.result()
                if res and res.get("title"):
                    all_movies.append(res)
                    with open(OUT_MOVIES_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 200 == 0 or n_done == len(year_tmdb_ids):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(f"    Progress [{n_done:,}/{len(year_tmdb_ids):,}] ({rate:.1f} movies/sec)")

    print(f"\n✅ STAGE 1 Complete. Total movies saved to {OUT_MOVIES_JSONL.name}: {len(all_movies):,}")
    return all_movies


# ── STAGE 2: Scrape Top 5 IMDb Reviews ─────────────────────────────────────

IMDB_GRAPHQL_QUERY = """
query GetTitleReviews($id: ID!) {
  title(id: $id) {
    id
    titleText { text }
    reviews(first: 5, sort: { by: HELPFULNESS_SCORE, order: DESC }) {
      edges {
        node {
          id
          author { nickName }
          authorRating
          summary { originalText }
          text { originalText { plainText } }
          submissionDate
          helpfulness {
            upVotes
            downVotes
          }
        }
      }
    }
  }
}
"""

def fetch_imdb_graphql_reviews(movie: dict) -> dict:
    tmdb_id = movie.get("tmdb_id")
    raw_imdb = str(movie.get("imdb_id", "")).replace("tt", "").strip()
    imdb_id = f"tt{raw_imdb.zfill(7)}" if raw_imdb else ""
    title   = movie.get("title")

    parsed_reviews = []
    if imdb_id:
        try:
            req = urllib.request.Request(
                "https://graphql.imdb.com/",
                data=json.dumps({"query": IMDB_GRAPHQL_QUERY, "variables": {"id": imdb_id}}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                }
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    edges = data.get("data", {}).get("title", {}).get("reviews", {}).get("edges", [])
                    for edge in edges:
                        n = edge.get("node", {})
                        if not n:
                            continue
                        author = n.get("author", {}).get("nickName") if n.get("author") else ""
                        text_val = n.get("text", {}).get("originalText", {}).get("plainText", "") if n.get("text") else ""
                        summary_val = n.get("summary", {}).get("originalText", "") if n.get("summary") else ""
                        helpfulness = n.get("helpfulness", {})
                        parsed_reviews.append({
                            "review_id": n.get("id"),
                            "rating": n.get("authorRating"),
                            "author": author,
                            "summary": summary_val,
                            "text": text_val,
                            "date": n.get("submissionDate"),
                            "up_votes": helpfulness.get("upVotes"),
                            "down_votes": helpfulness.get("downVotes"),
                            "source": "imdb_graphql_top_helpful"
                        })
        except Exception:
            pass

    # Fallback to TMDB user reviews if IMDb GraphQL didn't return reviews
    if not parsed_reviews and tmdb_id:
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={TMDB_API_KEY}"
            req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = data.get("results", [])
                for r in results[:5]:
                    content = r.get("content", "").replace("\r\n", "\n").strip()
                    first_line = content.split("\n")[0] if content else ""
                    summary = first_line[:120] if len(first_line) > 5 else content[:120]
                    parsed_reviews.append({
                        "review_id": r.get("id"),
                        "rating": r.get("author_details", {}).get("rating"),
                        "author": r.get("author"),
                        "summary": summary,
                        "text": content,
                        "date": r.get("created_at"),
                        "source": "tmdb_api_fallback"
                    })
        except Exception:
            pass

    return {
        "tmdb_id": tmdb_id,
        "imdb_id": imdb_id,
        "title": title,
        "reviews": parsed_reviews,
    }


def run_stage_2_imdb(movies: list[dict]):
    print("\n" + "=" * 65)
    print("⭐ STAGE 2: Scraping Top 5 IMDb User Reviews for Modern Movies")
    print("=" * 65)

    processed_tmdb_ids = set()
    if OUT_REVIEWS_JSONL.exists():
        with open(OUT_REVIEWS_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_tmdb_ids.add(json.loads(line)["tmdb_id"])
                    except Exception:
                        pass

    to_scrape = [m for m in movies if m.get("tmdb_id") not in processed_tmdb_ids]
    print(f"Movies remaining for user review fetching: {len(to_scrape):,}")

    if not to_scrape:
        print("✅ All movies already have user reviews fetched!")
        return

    n_done = 0
    n_reviews = 0
    t0 = time.time()

    with open(OUT_REVIEWS_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(fetch_imdb_graphql_reviews, m): m for m in to_scrape}
            for future in as_completed(futures):
                res = future.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                n_reviews += len(res.get("reviews", []))
                if n_done % 200 == 0 or n_done == len(to_scrape):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(f"  IMDb Progress: [{n_done:,}/{len(to_scrape):,}] {rate:.1f} movies/sec | Total reviews: {n_reviews:,}")

    print(f"✅ STAGE 2 Complete. Scraped {n_reviews:,} reviews for {n_done:,} movies.")


# ── STAGE 3: Fetch Wikipedia Plot Summaries ───────────────────────────────

def fetch_wikipedia_plot(movie: dict) -> dict:
    title = movie["title"]
    year = movie.get("year")
    tmdb_id = movie.get("tmdb_id")

    normalized = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
    query = f"{normalized} ({year} film)" if year else f"{normalized} film"

    session = requests.Session()
    session.headers.update({"User-Agent": "CineVaultSystemBot/1.0 (contact: meerpi@example.com)"})
    search_url = "https://en.wikipedia.org/w/api.php"

    page_title = None
    intro = None
    plot = None

    try:
        resp = session.get(search_url, params={"action": "query", "list": "search", "srsearch": query, "format": "json", "utf8": 1}, timeout=2.0)
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                page_title = results[0]["title"]
    except Exception:
        pass

    if page_title:
        try:
            resp = session.get(search_url, params={"action": "parse", "page": page_title, "prop": "wikitext", "format": "json", "utf8": 1}, timeout=2.0)
            if resp.status_code == 200:
                wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
                intro, plot = extract_intro_and_plot(wikitext)
        except Exception:
            pass

    return {
        "tmdb_id": tmdb_id,
        "title": title,
        "year": year,
        "wikipedia_page": page_title,
        "intro": intro,
        "plot": plot,
    }


def extract_intro_and_plot(wikitext: str) -> tuple[str | None, str | None]:
    clean_wt = re.sub(r"<!--.*?-->", "", wikitext, flags=re.DOTALL)
    sections = re.split(r"^(==+\s*[^=]+\s*==+)", clean_wt, flags=re.MULTILINE)

    intro_raw = sections[0] if sections else ""
    plot_raw  = ""

    for i in range(1, len(sections), 2):
        header = sections[i].lower()
        if "plot" in header or "synopsis" in header or "summary" in header:
            if i + 1 < len(sections):
                plot_raw = sections[i + 1]
                break

    intro = clean_wikitext(intro_raw) if intro_raw else None
    plot  = clean_wikitext(plot_raw) if plot_raw else None
    return intro, plot


def clean_wikitext(text: str) -> str:
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[Category:.*?\]\]", "", text)
    text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[Image:.*?\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"'''+", "", text)
    text = re.sub(r"''", "", text)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>", "", text)
    text = re.sub(r"==+.*?==+", "", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def run_stage_3_wikipedia(movies: list[dict]):
    print("\n" + "=" * 65)
    print("📖 STAGE 3: Fetching Wikipedia Plot Summaries for Modern Movies")
    print("=" * 65)

    processed_tmdb_ids = set()
    if OUT_WIKI_JSONL.exists():
        with open(OUT_WIKI_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_tmdb_ids.add(json.loads(line)["tmdb_id"])
                    except Exception:
                        pass

    to_scrape = [m for m in movies if m.get("tmdb_id") not in processed_tmdb_ids]
    print(f"Movies remaining for Wikipedia plot fetching: {len(to_scrape):,}")

    if not to_scrape:
        print("✅ All movies already have Wikipedia plots fetched!")
        return

    n_done = 0
    n_plots = 0
    t0 = time.time()

    with open(OUT_WIKI_JSONL, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(fetch_wikipedia_plot, m): m for m in to_scrape}
            for future in as_completed(futures):
                res = future.result()
                out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if res.get("plot"):
                    n_plots += 1
                if n_done % 300 == 0 or n_done == len(to_scrape):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    print(f"  Wiki Progress: [{n_done:,}/{len(to_scrape):,}] {rate:.1f} movies/sec | Plots found: {n_plots:,}")

    print(f"✅ STAGE 3 Complete. Fetched {n_plots:,} Wikipedia plots for {n_done:,} movies.")


# ── MAIN PIPELINE ──────────────────────────────────────────────────────────

def main():
    print("🚀 CineVault Modern Movie Ingestion Pipeline (2019-2026)")
    
    # Load all existing movies from JSONL if already present
    movies = []
    if OUT_MOVIES_JSONL.exists():
        with open(OUT_MOVIES_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        movies.append(json.loads(line))
                    except Exception:
                        pass

    if len(movies) < 6500:
        run_stage_1_tmdb()

    # Always reload complete list of movies from JSONL
    movies = []
    with open(OUT_MOVIES_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    movies.append(json.loads(line))
                except Exception:
                    pass

    print(f"\nTotal Modern Movies Available: {len(movies):,}")

    # Stage 2: Scrape Top 5 IMDb Reviews
    run_stage_2_imdb(movies)

    # Stage 3: Fetch Wikipedia Plots
    run_stage_3_wikipedia(movies)

    print("\n🎉 ALL STAGES COMPLETE!")
    print(f"  - Movies metadata:  {OUT_MOVIES_JSONL}")
    print(f"  - IMDb top reviews: {OUT_REVIEWS_JSONL}")
    print(f"  - Wikipedia plots:  {OUT_WIKI_JSONL}")


if __name__ == "__main__":
    main()
