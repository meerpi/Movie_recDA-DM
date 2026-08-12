"""nlp/router.py — Deterministic fast-path query routing."""

import re
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"

class QueryRouter:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.ro_db_uri = f"file:{self.db_path.resolve()}?mode=ro"

    def connect_ro(self):
        return sqlite3.connect(self.ro_db_uri, uri=True)

    def match_deterministic(self, query):
        q = query.lower().strip()

        if not q:
            return self.get_default_recommendations(limit=100)

        # 'movies like [Title]' — only fast-path when no qualifier modifiers in head/tail
        like_match = re.search(r"(?:movies like|similar to|recommendations for|if i liked)\s+([a-zA-Z0-9\s:'-]+)", q)
        if like_match:
            head = q[:like_match.start()].strip()
            head_clean = re.sub(r"^(?:show|find|give|get|recommend|search|i want|can you find|please)?\s*(?:me|us)?\s*", "", head).strip()

            title_search = like_match.group(1).strip()
            tail = q[like_match.end():].strip()
            has_modifiers = bool(re.search(
                r"\b(but|with|and|except|without|not|less|more)\b", tail
            ))

            if not head_clean and not has_modifiers:
                res = self.get_associated_movies(title_search)
                if res and res.get("type") == "association_rules":
                    return res

        # 'top N [genre] movies'
        top_match = re.search(r"(?:top|best|highest rated)\s+(\d+)?\s*([a-zA-Z]+)?\s*movies?", q)
        if top_match:
            limit = int(top_match.group(1)) if top_match.group(1) else 10
            genre = top_match.group(2) if top_match.group(2) else None
            if genre and genre not in ["all", "any"]:
                return self.get_top_movies_by_genre(genre=genre, limit=limit, sort_by="rating")
            else:
                return self.get_top_movies(limit=limit, sort_by="rating")

        # 'most popular [genre] movies'
        pop_match = re.search(r"(?:most popular|trending)\s+([a-zA-Z]+)?\s*movies?", q)
        if pop_match:
            genre = pop_match.group(1) if pop_match.group(1) else None
            if genre:
                return self.get_top_movies_by_genre(genre=genre, limit=10, sort_by="popularity")
            else:
                return self.get_top_movies(10, sort_by="popularity")

        return None

    def get_associated_movies(self, title, limit=10):
        conn = self.connect_ro()
        cur = conn.cursor()

        clean_title = title.lower().replace("the ", "").strip()

        # exact match first, then substring fallback
        cur.execute("""
            SELECT m.movie_id, m.title, m.year
            FROM movies m
            LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE LOWER(m.title) = ? OR LOWER(m.title) = ?
            ORDER BY ms.num_ratings DESC
            LIMIT 1
        """, (title.lower(), clean_title))
        row = cur.fetchone()

        if not row:
            cur.execute("""
                SELECT m.movie_id, m.title, m.year
                FROM movies m
                LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
                WHERE LOWER(m.title) LIKE ? OR LOWER(m.title) LIKE ?
                ORDER BY ms.num_ratings DESC
                LIMIT 1
            """, (f"%{title.lower()}%", f"%{clean_title}%"))
            row = cur.fetchone()

        if not row:
            conn.close()
            return {"type": "error", "message": f"Movie '{title}' not found in database."}

        mid, found_title, year = row

        cur.execute("""
            SELECT g.name FROM genres g
            JOIN movie_genres mg ON g.genre_id = mg.genre_id
            WHERE mg.movie_id = ?
        """, (mid,))
        source_genres = set(r[0] for r in cur.fetchall())
        is_source_comedy = "Comedy" in source_genres

        # filter out pure comedies if source isn't one
        comedy_clause = ""
        if not is_source_comedy:
            comedy_clause = """
              AND m.movie_id NOT IN (
                  SELECT mg_c.movie_id FROM movie_genres mg_c
                  JOIN genres g_c ON mg_c.genre_id = g_c.genre_id
                  WHERE g_c.name = 'Comedy'
              )
            """

        query = f"""
            SELECT DISTINCT m.movie_id, m.title, m.year, r.confidence, r.lift, ms.avg_rating, ms.num_ratings,
                   (SELECT GROUP_CONCAT(g2.name) FROM movie_genres mg2 JOIN genres g2 ON mg2.genre_id = g2.genre_id WHERE mg2.movie_id = m.movie_id) as all_genres
            FROM association_rules r
            JOIN movies m ON r.movie_id_b = m.movie_id
            JOIN movie_genres mg_b ON m.movie_id = mg_b.movie_id
            LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE r.movie_id_a = ?
              AND mg_b.genre_id IN (SELECT genre_id FROM movie_genres WHERE movie_id = ?)
              {comedy_clause}
            GROUP BY m.movie_id
            ORDER BY r.lift DESC, r.confidence DESC
            LIMIT ?
        """
        cur.execute(query, (mid, mid, limit))
        results = []
        for r in cur.fetchall():
            num_r = r[6] or 0
            results.append({
                "movie_id": r[0],
                "title": r[1],
                "year": r[2],
                "confidence": r[3],
                "lift": r[4],
                "avg_rating": r[5] if (r[5] is not None and num_r > 0) else None,
                "num_ratings": num_r
            })

        conn.close()

        return {
            "type": "association_rules",
            "source_movie": {"movie_id": mid, "title": found_title, "year": year},
            "results": results
        }

    def get_top_movies_by_genre(self, genre, limit=10, sort_by="rating"):
        conn = self.connect_ro()
        cur = conn.cursor()

        order = "ms.avg_rating DESC" if sort_by == "rating" else "ms.popularity_rank ASC"
        cur.execute(f"""
            SELECT m.movie_id, m.title, m.year, g.name as genre, ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
            FROM movies m
            JOIN movie_genres mg ON m.movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.genre_id
            JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE LOWER(g.name) = ? AND ms.num_ratings > 0
            ORDER BY {order}
            LIMIT ?
        """, (genre.lower(), limit))
        results = [
            {
                "movie_id": r[0], "title": r[1], "year": r[2], "genre": r[3],
                "avg_rating": r[4], "num_ratings": r[5], "pct_positive": r[6],
                "popularity_rank": r[7]
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return {"type": "genre_top", "genre": genre, "sort_by": sort_by, "results": results}

    def get_default_recommendations(self, limit=10):
        """Landing page recs: feature films ranked by Bayesian weighted rating.
        Excludes documentaries and entries with no year. Requires >= 1000 ratings."""
        conn = self.connect_ro()
        cur = conn.cursor()

        m, C = 500, 3.5  # prior strength, prior mean

        cur.execute(f"""
            SELECT m.movie_id, m.title, m.year,
                   ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank,
                   (ms.num_ratings * ms.avg_rating + {m} * {C}) / (ms.num_ratings + {m}) AS bayes_rating
            FROM movies m
            JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE ms.num_ratings >= 1000
              AND m.year IS NOT NULL
              AND m.movie_id NOT IN (
                  SELECT mg.movie_id FROM movie_genres mg
                  JOIN genres g ON mg.genre_id = g.genre_id
                  WHERE g.name = 'Documentary'
              )
            ORDER BY bayes_rating DESC
            LIMIT ?
        """, (limit,))
        results = [
            {
                "movie_id": r[0], "title": r[1], "year": r[2],
                "avg_rating": r[3], "num_ratings": r[4], "pct_positive": r[5],
                "popularity_rank": r[6]
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return {"type": "overall_top", "sort_by": "bayesian_rating", "results": results}

    def get_top_movies(self, limit=10, sort_by="rating"):
        conn = self.connect_ro()
        cur = conn.cursor()

        order = "ms.avg_rating DESC" if sort_by == "rating" else "ms.popularity_rank ASC"
        cur.execute(f"""
            SELECT m.movie_id, m.title, m.year, ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
            FROM movies m
            JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE ms.num_ratings >= 100
            ORDER BY {order}
            LIMIT ?
        """, (limit,))
        results = [
            {
                "movie_id": r[0], "title": r[1], "year": r[2],
                "avg_rating": r[3], "num_ratings": r[4], "pct_positive": r[5],
                "popularity_rank": r[6]
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return {"type": "overall_top", "sort_by": sort_by, "results": results}
