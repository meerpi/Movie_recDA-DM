"""nlp/hydrator.py — Enriches movie IDs with DB stats and profile card fields."""

import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("cinevault.hydrator")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"
TIER_A_PATH  = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_PATH  = PROJECT_ROOT / "dirtywork" / "tier_b_voyage_cards.jsonl"
TIER_C_PATH  = PROJECT_ROOT / "dirtywork" / "tier_c_voyage_cards.jsonl"


class ResultHydrator:

    def __init__(self, db_path=DB_PATH):
        t0 = time.time()
        logger.info("Initializing Result Hydrator ...")
        self.db_path = db_path

        self.cards: dict[int, dict] = {}
        self._load_cards(TIER_A_PATH, "Tier A")
        self._load_cards(TIER_B_PATH, "Tier B")
        self._load_cards(TIER_C_PATH, "Tier C")

        logger.info(f"Hydrator ready: {len(self.cards):,} cards in {time.time() - t0:.2f}s.")

    def _load_cards(self, path, default_tier):
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    mid  = card.get("movie_id")
                    if mid is not None:
                        card.setdefault("tier", default_tier)
                        self.cards[int(mid)] = card
                except json.JSONDecodeError:
                    continue

    def hydrate(self, search_results):
        if not search_results:
            return []

        mids = [r["movie_id"] for r in search_results]
        db_info: dict[int, dict] = {}

        if self.db_path.exists():
            conn   = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            placeholders = ",".join("?" * len(mids))
            cursor.execute(f"""
                SELECT m.movie_id, m.title, m.year,
                       GROUP_CONCAT(DISTINCT g.name) as genres,
                       ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                WHERE m.movie_id IN ({placeholders})
                GROUP BY m.movie_id
            """, mids)
            for row in cursor.fetchall():
                mid, title, year, genres, avg_rating, num_ratings, pct_pos, pop_rank = row
                db_info[mid] = {
                    "title":           title,
                    "year":            year,
                    "genres":          genres.split(",") if genres else [],
                    "avg_rating":      avg_rating if (num_ratings and num_ratings > 0) else None,
                    "num_ratings":     num_ratings or 0,
                    "pct_positive":    pct_pos if (num_ratings and num_ratings > 0) else 0.0,
                    "popularity_rank": pop_rank or 999999,
                }
            conn.close()

        hydrated = []
        for hit in search_results:
            mid  = hit["movie_id"]
            card = self.cards.get(mid, {})
            db   = db_info.get(mid, {})

            tier = card.get("tier") or ("Tier A" if "themes" in card else "Tier B" if "genome_vector" in card else "Tier C")

            raw_title = card.get("title") or db.get("title") or f"Movie #{mid}"
            year = card.get("year") or db.get("year")
            if year and raw_title.endswith(f" ({year})"):
                raw_title = raw_title[:-len(f" ({year})")]

            item = {
                "movie_id":            mid,
                "title":               raw_title,
                "year":                year,
                "tier":                tier,
                "rrf_rank":            hit.get("rrf_rank"),
                "rrf_score":           hit.get("rrf_score", 0.0),
                "lanes":               hit.get("lanes", []),
                "lane_ranks":          hit.get("lane_ranks", {}),
                "genres":              db.get("genres") or card.get("genres", []),
                "avg_rating":          db.get("avg_rating"),
                "num_ratings":         db.get("num_ratings", 0),
                "pct_positive":        db.get("pct_positive", 0.0),
                "popularity_rank":     db.get("popularity_rank", 999999),
                "actors":              card.get("actors") or [],
                "directors":           card.get("directors") or [],
                "content_rating":      card.get("content_rating") or "",
                "original_language":   card.get("original_language") or "",
                "production_countries": card.get("production_countries") or [],
                "collection":          card.get("collection"),
                "poster_path":         card.get("poster_path"),
                "backdrop_path":       card.get("backdrop_path"),
                "tagline":             card.get("tagline") or "",
            }

            if "themes" in card:
                item["themes"]                 = card.get("themes", [])
                item["tone"]                   = card.get("tone", [])
                item["pacing"]                 = card.get("pacing", "")
                item["directorial_style_notes"] = card.get("directorial_style_notes", "")
                item["comparable_films"]       = card.get("comparable_films", [])
                item["standout_performances"]  = card.get("standout_performances", [])
                item["notable_criticisms"]     = card.get("notable_criticisms", [])

            if "top_tags" in card:
                item["top_tags"] = [t["tag"] if isinstance(t, dict) else t for t in card["top_tags"][:10]]
            elif "user_tags" in card:
                item["user_tags"] = card["user_tags"][:10]

            if "keywords" in card:
                item["keywords"] = card["keywords"][:20]

            hydrated.append(item)

        return hydrated
