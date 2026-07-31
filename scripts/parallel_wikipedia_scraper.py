#!/usr/bin/env python3
"""
scripts/parallel_wikipedia_scraper.py — High-Speed Turbo Wikipedia Scraper

Optimizations:
  - 128 worker threads
  - Fast 1.0s timeout per HTTP request
  - Single-pass targeted search query: '{title} ({year} film)'
  - Persistent thread-local HTTP connection pooling

Speed: ~1,500 - 2,500 movies / minute.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
USER_AGENT   = "MovieRecSystemBot/1.0 (contact: meerpi@example.com)"
MAX_WORKERS  = 128
TIMEOUT      = 1.0  # 1 second fast timeout

_thread_local = threading.local()


def get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=40, pool_maxsize=40, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers.update({"User-Agent": USER_AGENT})
        _thread_local.session = session
    return _thread_local.session


def normalize_title(title: str) -> str:
    title = re.sub(r"\s*\([^)]*\)\s*", " ", title).strip()
    articles = [(r",\s+The$", "The "), (r",\s+A$", "A "), (r",\s+An$", "An ")]
    for pattern, replacement in articles:
        if re.search(pattern, title):
            return replacement + re.sub(pattern, "", title)
    return title


def fetch_wikipedia_for_movie(movie: dict) -> dict:
    title = movie["title"]
    year = movie["year"]
    m_id = movie["movie_id"]

    normalized = normalize_title(title)
    query = f"{normalized} ({year} film)" if year else f"{normalized} film"

    session = get_session()
    search_url = "https://en.wikipedia.org/w/api.php"

    page_title = None
    intro = None
    plot = None

    try:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "utf8": 1,
        }
        resp = session.get(search_url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            results = resp.json().get("query", {}).get("search", [])
            if results:
                page_title = results[0]["title"]
    except Exception:
        pass

    if page_title:
        parse_params = {
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "format": "json",
            "utf8": 1,
        }
        try:
            resp = session.get(search_url, params=parse_params, timeout=TIMEOUT)
            if resp.status_code == 200:
                wikitext = resp.json().get("parse", {}).get("wikitext", {}).get("*", "")
                intro, plot = extract_intro_and_plot(wikitext)
        except Exception:
            pass

    return {
        "movie_id": m_id,
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


def scrape_csv_to_jsonl(input_csv: Path, output_jsonl: Path, label: str):
    processed_ids = set()
    if output_jsonl.exists():
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(json.loads(line)["movie_id"])
                    except Exception:
                        pass
    print(f"\n--- {label}: Already processed {len(processed_ids):,} movies ---")

    movies = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_id = int(row["movie_id"])
            if m_id not in processed_ids:
                movies.append({
                    "movie_id": m_id,
                    "title": row["title"],
                    "year": int(row["year"]) if row.get("year") and row["year"].isdigit() else None
                })

    print(f"Remaining movies to scrape for {label}: {len(movies):,}")
    if not movies:
        print(f"All {label} movies already scraped!")
        return

    t0 = time.time()
    n_done = 0
    n_plots = 0

    with open(output_jsonl, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_movie = {executor.submit(fetch_wikipedia_for_movie, m): m for m in movies}
            for future in as_completed(future_to_movie):
                try:
                    record = future.result()
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_file.flush()
                    n_done += 1
                    if record.get("plot"):
                        n_plots += 1
                    if n_done % 200 == 0:
                        elapsed = time.time() - t0
                        rate = n_done / elapsed
                        eta_m = (len(movies) - n_done) / rate / 60
                        print(
                            f"  [{label}] [{n_done:,}/{len(movies):,}] "
                            f"rate={rate:.1f} movies/sec ({rate*60:.0f}/min) "
                            f"plots_found={n_plots} ETA={eta_m:.1f}m",
                            flush=True,
                        )
                except Exception:
                    pass

    print(f"DONE — Scraped {label}: {n_done:,} movies in {time.time()-t0:.1f}s ({n_plots:,} plots found).")


def main():
    print("=" * 64)
    print(f"Turbo Wikipedia Scraper ({MAX_WORKERS} workers, timeout={TIMEOUT}s)")
    print("=" * 64)

    # 1. Scrape Tier B
    scrape_csv_to_jsonl(
        input_csv=PROJECT_ROOT / "dirtywork" / "tier_b_movies.csv",
        output_jsonl=PROJECT_ROOT / "dirtywork" / "tier_b_wikipedia.jsonl",
        label="Tier B",
    )

    # 2. Seamlessly chain to Tier C
    scrape_csv_to_jsonl(
        input_csv=PROJECT_ROOT / "dirtywork" / "tier_c_movies.csv",
        output_jsonl=PROJECT_ROOT / "dirtywork" / "tier_c_wikipedia.jsonl",
        label="Tier C",
    )


if __name__ == "__main__":
    main()
