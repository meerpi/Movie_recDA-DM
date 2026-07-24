#!/usr/bin/env python3
"""
nlp/hydrator.py — Step 6: Result Hydration Layer

Enriches raw (movie_id, rrf_score) search results with full movie metadata:
  • DB attributes: title, year, genres, average rating, rating count (from cinevault.db)
  • Profile card attributes: themes, tone, pacing, top_tags, comparable_films, etc.
  • Content tier label: "Tier A", "Tier B", or "Tier C"

Usage:
    from nlp.hydrator import ResultHydrator

    hydrator = ResultHydrator()
    enriched_results = hydrator.hydrate(fused_search_hits)
"""

import json
import sqlite3
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"

TIER_A_PATH  = PROJECT_ROOT / "tier_a_profile_cards_v3.jsonl"
TIER_B_PATH  = PROJECT_ROOT / "tier_b_profile_cards.jsonl"
TIER_C_PATH  = PROJECT_ROOT / "tier_c_profile_cards.jsonl"


class ResultHydrator:
    """
    Hydrates movie IDs with database ratings/genres and profile card details.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        t0 = time.time()
        print("Initializing Result Hydrator ...")
        self.db_path = db_path

        # In-memory card cache for fast lookup: movie_id -> card dict
        self.cards: dict[int, dict] = {}
        self._load_cards(TIER_A_PATH, "Tier A")
        self._load_cards(TIER_B_PATH, "Tier B")
        self._load_cards(TIER_C_PATH, "Tier C")

        print(f"Hydrator initialized with {len(self.cards):,} profile cards in {time.time() - t0:.2f}s.\n")

    def _load_cards(self, path: Path, default_tier: str) -> None:
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    mid = card.get("movie_id")
                    if mid is not None:
                        if "tier" not in card:
                            card["tier"] = default_tier
                        self.cards[int(mid)] = card
                except json.JSONDecodeError:
                    continue

    def hydrate(self, search_results: list[dict]) -> list[dict]:
        """
        Hydrates a list of search result dicts from retriever.search().

        Input item:
            {"movie_id": 143657, "rrf_score": 0.04723, "lanes": [...], "lane_ranks": {...}}

        Output item:
            Full enriched movie object with title, year, genres, avg_rating, ratings_count,
            and tier-specific profile fields.
        """
        if not search_results:
            return []

        mids = [r["movie_id"] for r in search_results]

        # ── 1. Batch SQL Query for DB Attributes ─────────────────────
        db_info: dict[int, dict] = {}
        if self.db_path.exists():
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(mids))

            query = f"""
                SELECT m.movie_id, m.title, m.year,
                       GROUP_CONCAT(DISTINCT g.name) as genres,
                       ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                WHERE m.movie_id IN ({placeholders})
                GROUP BY m.movie_id
            """
            cursor.execute(query, mids)
            for row in cursor.fetchall():
                mid, title, year, genres, avg_rating, num_ratings, pct_pos, pop_rank = row
                db_info[mid] = {
                    "title": title,
                    "year": year,
                    "genres": genres.split(",") if genres else [],
                    "avg_rating": avg_rating if (num_ratings and num_ratings > 0) else None,
                    "num_ratings": num_ratings or 0,
                    "pct_positive": pct_pos if (num_ratings and num_ratings > 0) else 0.0,
                    "popularity_rank": pop_rank or 999999,
                }
            conn.close()

        # ── 2. Assemble Enriched Output ──────────────────────────────
        hydrated: list[dict] = []
        for hit in search_results:
            mid  = hit["movie_id"]
            card = self.cards.get(mid, {})
            db   = db_info.get(mid, {})

            # Tier determination
            tier = card.get("tier")
            if not tier:
                if "themes" in card:
                    tier = "Tier A"
                elif "genome_vector" in card:
                    tier = "Tier B"
                else:
                    tier = "Tier C"

            item = {
                "movie_id": mid,
                "title": card.get("title") or db.get("title") or f"Movie #{mid}",
                "year": card.get("year") or db.get("year"),
                "tier": tier,
                "rrf_score": hit.get("rrf_score", 0.0),
                "lanes": hit.get("lanes", []),
                "lane_ranks": hit.get("lane_ranks", {}),
                "genres": db.get("genres") or card.get("genres", []),
                "avg_rating": db.get("avg_rating"),
                "num_ratings": db.get("num_ratings", 0),
                "pct_positive": db.get("pct_positive", 0.0),
                "popularity_rank": db.get("popularity_rank", 999999),
                # ── TMDb-enriched structured fields (all tiers) ──────────
                "actors": card.get("actors") or [],
                "directors": card.get("directors") or [],
                "content_rating": card.get("content_rating") or "",
                "original_language": card.get("original_language") or "",
                "production_countries": card.get("production_countries") or [],
                # collection is the primary franchise dedup key for MMR
                "collection": card.get("collection") or None,
                "poster_path": card.get("poster_path") or None,
                "backdrop_path": card.get("backdrop_path") or None,
                "tagline": card.get("tagline") or "",
            }

            # Tier A rich qualitative fields
            if "themes" in card:
                item["themes"] = card.get("themes", [])
                item["tone"] = card.get("tone", [])
                item["pacing"] = card.get("pacing", "")
                item["directorial_style_notes"] = card.get("directorial_style_notes", "")
                item["comparable_films"] = card.get("comparable_films", [])
                item["standout_performances"] = card.get("standout_performances", [])
                item["notable_criticisms"] = card.get("notable_criticisms", [])

            # Tag signal (Tier A + B top_tags or Tier C user_tags)
            if "top_tags" in card:
                item["top_tags"] = [t["tag"] if isinstance(t, dict) else t for t in card.get("top_tags", [])[:10]]
            elif "user_tags" in card:
                item["user_tags"] = card.get("user_tags", [])[:10]

            # Keywords (Tier B + C only — Tier A skips to preserve Tag Genome quality)
            if "keywords" in card:
                item["keywords"] = card.get("keywords", [])[:20]

            hydrated.append(item)

        return hydrated


# ---------------------------------------------------------------------------
# CLI test runner
# ---------------------------------------------------------------------------
def main():
    from nlp.retriever import CineVaultRetriever

    retriever = CineVaultRetriever()
    hydrator  = ResultHydrator()

    query = "atmospheric slow burn Korean thriller"
    hits = retriever.search(query, top_k=5)
    results = hydrator.hydrate(hits)

    print(f"Hydrated Results for '{query}':\n" + "=" * 60)
    for idx, r in enumerate(results, 1):
        print(f"#{idx}  {r['title']} ({r['year']})  [{r['tier']}]  Score: {r['rrf_score']}")
        print(f"    Genres: {', '.join(r['genres'])} | ★ {r['avg_rating']} ({r['num_ratings']:,} ratings)")
        if "themes" in r:
            print(f"    Themes: {', '.join(r['themes'])}")
            print(f"    Tone  : {', '.join(r['tone'])}")
        if "top_tags" in r:
            print(f"    Tags  : {', '.join(r['top_tags'][:6])}")
        print()


if __name__ == "__main__":
    main()
