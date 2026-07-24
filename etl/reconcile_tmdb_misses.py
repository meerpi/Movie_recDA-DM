#!/usr/bin/env python3
"""
etl/reconcile_tmdb_misses.py — TMDb Reconciliation Pass for Unenriched Movies

Targets Tier A and Tier B cards that have no actors/directors from the enrichment
pass, typically due to title/year mismatches in the stored tmdb_id.

Strategy:
  1. Identify unenriched movie_ids (no actors field in profile cards).
  2. For each, query TMDb /search/movie?query=<title>&year=<year> to find the
     correct tmdb_id.
  3. Store the corrected tmdb_id back to SQLite.
  4. Re-run the standard TMDb detail fetch (same as enrich_tier_a_b_tmdb.py).

Usage:
    python etl/reconcile_tmdb_misses.py
    python etl/reconcile_tmdb_misses.py 50    # limit to 50 movies for testing
"""

import json
import sqlite3
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
TIER_A_CARDS = PROJECT_ROOT / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS = PROJECT_ROOT / "tier_b_profile_cards.jsonl"

TMDB_API_KEY = "2cc1fd2a583e2e14f6b634fb124f9ced"
MAX_WORKERS = 20  # Lower than normal — search endpoint has stricter rate limits


def find_unenriched_ids() -> dict:
    """Returns {movie_id: (title, year)} for all Tier A/B cards missing actors."""
    missing = {}
    for path in [TIER_A_CARDS, TIER_B_CARDS]:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    if not card.get("actors") and not card.get("directors"):
                        mid = card.get("movie_id")
                        title = card.get("title", "")
                        year = card.get("year")
                        if mid:
                            missing[int(mid)] = (title, year)
                except Exception:
                    pass
    return missing


def search_tmdb_id(movie_id: int, title: str, year):
    """Search TMDb by title+year to find the correct tmdb_id."""
    # Strip alternate-title suffixes like "(a.k.a. Ghost Busters)"
    clean_title = title.split("(a.k.a.")[0].split("(aka")[0].strip()
    # Strip trailing year patterns like "(1984)"
    import re
    clean_title = re.sub(r"\s*\(\d{4}\)\s*$", "", clean_title).strip()

    params = {"api_key": TMDB_API_KEY, "query": clean_title}
    if year:
        params["year"] = str(year)

    url = "https://api.themoviedb.org/3/search/movie?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                return {"movie_id": movie_id, "tmdb_id": results[0]["id"], "success": True}
            # Try without year if year-scoped search returned nothing
            if year:
                params2 = {"api_key": TMDB_API_KEY, "query": clean_title}
                url2 = "https://api.themoviedb.org/3/search/movie?" + urllib.parse.urlencode(params2)
                req2 = urllib.request.Request(url2, headers={"User-Agent": "CineVault/1.0"})
                with urllib.request.urlopen(req2, timeout=10) as resp2:
                    data2 = json.loads(resp2.read().decode("utf-8"))
                    results2 = data2.get("results", [])
                    if results2:
                        return {"movie_id": movie_id, "tmdb_id": results2[0]["id"], "success": True}
            return {"movie_id": movie_id, "success": False, "reason": "no_results"}
    except Exception as e:
        return {"movie_id": movie_id, "success": False, "reason": str(e)}


def fetch_tmdb_detail(movie_id: int, tmdb_id: int):
    """Fetch full TMDb detail (same as enrich_tier_a_b_tmdb.py)."""
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,keywords,release_dates"
    req = urllib.request.Request(url, headers={"User-Agent": "CineVault/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

            directors = [
                c.get("name")
                for c in data.get("credits", {}).get("crew", [])
                if c.get("job") == "Director" and c.get("name")
            ]
            cast = [
                c.get("name")
                for c in data.get("credits", {}).get("cast", [])[:5]
                if c.get("name")
            ]
            mpaa = ""
            for res in data.get("release_dates", {}).get("results", []):
                if res.get("iso_3166_1") == "US":
                    for rd in res.get("release_dates", []):
                        cert = rd.get("certification")
                        if cert:
                            mpaa = cert
                            break
            keywords = [
                k.get("name")
                for k in data.get("keywords", {}).get("keywords", [])
                if k.get("name")
            ]
            collection_info = data.get("belongs_to_collection")
            collection_name = (
                collection_info.get("name") if isinstance(collection_info, dict) else None
            )
            return {
                "movie_id": movie_id,
                "tmdb_id": tmdb_id,
                "directors": directors,
                "actors": cast,
                "content_rating": mpaa,
                "original_language": data.get("original_language", ""),
                "production_countries": [
                    c.get("name") for c in data.get("production_countries", []) if c.get("name")
                ],
                "keywords": keywords,
                "collection": collection_name,
                "poster_path": data.get("poster_path"),
                "backdrop_path": data.get("backdrop_path"),
                "tagline": data.get("tagline") or "",
                "success": True,
            }
    except Exception as e:
        return {"movie_id": movie_id, "success": False, "reason": str(e)}


def update_cards(card_path: Path, meta_map: dict, include_keywords: bool):
    """Patch enriched fields into a JSONL profile card file in-place."""
    if not card_path.exists() or not meta_map:
        return 0
    temp = card_path.with_suffix(".jsonl.tmp")
    updated = 0
    with open(card_path, encoding="utf-8") as inf, open(temp, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            card = json.loads(line)
            mid = card.get("movie_id")
            if mid in meta_map:
                meta = meta_map[mid]
                if meta.get("directors"):
                    card["directors"] = meta["directors"]
                if meta.get("actors"):
                    card["actors"] = meta["actors"]
                if meta.get("content_rating"):
                    card["content_rating"] = meta["content_rating"]
                if meta.get("original_language"):
                    card["original_language"] = meta["original_language"]
                if meta.get("production_countries"):
                    card["production_countries"] = meta["production_countries"]
                if meta.get("collection"):
                    card["collection"] = meta["collection"]
                if meta.get("poster_path"):
                    card["poster_path"] = meta["poster_path"]
                if meta.get("backdrop_path"):
                    card["backdrop_path"] = meta["backdrop_path"]
                if meta.get("tagline"):
                    card["tagline"] = meta["tagline"]
                if include_keywords and meta.get("keywords"):
                    card["keywords"] = meta["keywords"]
                updated += 1
            outf.write(json.dumps(card, ensure_ascii=False) + "\n")
    temp.replace(card_path)
    return updated


def run_reconciliation(limit=None):
    t0 = time.time()
    missing = find_unenriched_ids()
    print(f"🔍 Found {len(missing):,} Tier A/B movies with no TMDb enrichment.")
    if not missing:
        print("✅ Nothing to reconcile.")
        return

    items = list(missing.items())
    if limit:
        items = items[:limit]

    print(f"🚀 Searching TMDb for {len(items):,} movies (Workers: {MAX_WORKERS})...")

    # Phase 1: Search for correct tmdb_ids
    found_tmdb = {}
    search_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = {executor.submit(search_tmdb_id, mid, title, year): mid for mid, (title, year) in items}
        for fut in as_completed(futs):
            res = fut.result()
            if res.get("success"):
                found_tmdb[res["movie_id"]] = res["tmdb_id"]
            else:
                search_errors += 1

    print(f"  ✓ Found TMDb IDs for {len(found_tmdb):,} movies. Search failures: {search_errors:,}.")

    # Phase 2: Update tmdb_ids in SQLite.
    # Use OR IGNORE to silently skip any rows where the resolved tmdb_id would violate
    # the UNIQUE constraint — this covers both conflicts with existing rows AND cases
    # where two different movie_ids in found_tmdb resolved to the same tmdb_id.
    if found_tmdb:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        before = cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_id IS NOT NULL").fetchone()[0]
        cur.executemany(
            "UPDATE OR IGNORE movies SET tmdb_id = ? WHERE movie_id = ?",
            [(tmdb_id, mid) for mid, tmdb_id in found_tmdb.items()],
        )
        after = cur.execute("SELECT COUNT(*) FROM movies WHERE tmdb_id IS NOT NULL").fetchone()[0]
        conn.commit()
        conn.close()
        updated = after - before
        skipped = len(found_tmdb) - updated
        print(f"  ✓ Updated {updated:,} tmdb_ids in SQLite. Skipped {skipped:,} (UNIQUE conflicts).")



    # Phase 3: Fetch full TMDb detail for found movies
    print(f"🎬 Fetching full TMDb detail for {len(found_tmdb):,} movies...")
    enriched = []
    detail_errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futs = {executor.submit(fetch_tmdb_detail, mid, tmdb_id): mid for mid, tmdb_id in found_tmdb.items()}
        for fut in as_completed(futs):
            res = fut.result()
            if res.get("success"):
                enriched.append(res)
            else:
                detail_errors += 1

    print(f"  ✓ Enriched {len(enriched):,} movies. Detail failures: {detail_errors:,}.")

    if not enriched:
        print("No movies enriched — nothing to write.")
        return

    # Phase 4: Write back to SQLite and JSONL cards
    meta_map = {r["movie_id"]: r for r in enriched}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    batch = [
        (r["content_rating"], ", ".join(r["directors"]), ", ".join(r["actors"]), r["movie_id"])
        for r in enriched
    ]
    cur.executemany(
        "UPDATE movies SET content_rating = ?, directors = ?, actors = ? WHERE movie_id = ?", batch
    )
    conn.commit()
    conn.close()

    # Determine which IDs belong to Tier A vs Tier B (keywords only for B)
    tier_a_ids = set()
    if TIER_A_CARDS.exists():
        with open(TIER_A_CARDS) as f:
            for line in f:
                if line.strip():
                    card = json.loads(line)
                    tier_a_ids.add(card.get("movie_id"))

    tier_a_meta = {mid: m for mid, m in meta_map.items() if mid in tier_a_ids}
    tier_b_meta = {mid: m for mid, m in meta_map.items() if mid not in tier_a_ids}

    n_a = update_cards(TIER_A_CARDS, tier_a_meta, include_keywords=False)
    n_b = update_cards(TIER_B_CARDS, tier_b_meta, include_keywords=True)

    print(f"\n✅ Reconciliation complete in {time.time() - t0:.1f}s.")
    print(f"   Tier A patched: {n_a:,} cards")
    print(f"   Tier B patched: {n_b:,} cards")


if __name__ == "__main__":
    import sys
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_reconciliation(limit=limit_arg)
