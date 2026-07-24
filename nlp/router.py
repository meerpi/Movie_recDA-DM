#!/usr/bin/env python3
"""
nlp/router.py — Step 13: Hybrid Query Router & LangChain SQL Agent Fallback

Routes natural language user queries:
 1. Deterministic Fast-Path:
    • Regex/Pattern matching for common analytical queries:
      - 'top N [genre] movies' / 'highest rated [genre] movies'
      - 'most popular [genre] movies'
      - 'movies like [title]' (uses association_rules table!)
 2. Scoped LLM Agent Fallback:
    • Uses LangChain SQLDatabase with a strict READ-ONLY SQLite connection (mode=ro).
    • Guarantees LLM generated SQL can NEVER mutate tables or profiles.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"

class QueryRouter:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.ro_db_uri = f"file:{self.db_path.resolve()}?mode=ro"

    def connect_ro(self):
        """Creates a strictly read-only SQLite connection."""
        return sqlite3.connect(self.ro_db_uri, uri=True)

    def match_deterministic(self, query: str) -> Optional[Dict[str, Any]]:
        q = query.lower().strip()

        # Pattern A: 'movies like [Title]'
        like_match = re.search(r"(?:movies like|similar to|recommendations for|if i liked)\s+([a-zA-Z0-9\s:'-]+)", q)
        if like_match:
            title_search = like_match.group(1).strip()
            return self.get_associated_movies(title_search)

        # Pattern B: 'top N [genre] movies'
        top_match = re.search(r"(?:top|best|highest rated)\s+(\d+)?\s*([a-zA-Z]+)?\s*movies?", q)
        if top_match:
            limit = int(top_match.group(1)) if top_match.group(1) else 10
            genre = top_match.group(2) if top_match.group(2) else None
            if genre and genre not in ["all", "any"]:
                return self.get_top_movies_by_genre(genre=genre, limit=limit, sort_by="rating")
            else:
                return self.get_top_movies(limit=limit, sort_by="rating")

        # Pattern C: 'most popular [genre] movies'
        pop_match = re.search(r"(?:most popular|trending)\s+([a-zA-Z]+)?\s*movies?", q)
        if pop_match:
            genre = pop_match.group(1) if pop_match.group(1) else None
            if genre:
                return self.get_top_movies_by_genre(genre=genre, limit=10, sort_by="popularity")
            else:
                return self.get_top_movies(limit=10, sort_by="popularity")

        return None

    def get_associated_movies(self, title: str, limit: int = 10) -> Dict[str, Any]:
        """Finds co-liked movies using pre-computed association rules."""
        conn = self.connect_ro()
        cur = conn.cursor()
        
        clean_title = title.lower().replace("the ", "").strip()
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

        # Get source movie genres
        cur.execute("""
            SELECT g.name FROM genres g
            JOIN movie_genres mg ON g.genre_id = mg.genre_id
            WHERE mg.movie_id = ?
        """, (mid,))
        source_genres = set(r[0] for r in cur.fetchall())
        is_source_comedy = "Comedy" in source_genres

        # Query association rules with genre consistency
        query = """
            SELECT DISTINCT m.movie_id, m.title, m.year, r.confidence, r.lift, ms.avg_rating, ms.num_ratings,
                   (SELECT GROUP_CONCAT(g2.name) FROM movie_genres mg2 JOIN genres g2 ON mg2.genre_id = g2.genre_id WHERE mg2.movie_id = m.movie_id) as all_genres
            FROM association_rules r
            JOIN movies m ON r.movie_id_b = m.movie_id
            JOIN movie_genres mg_b ON m.movie_id = mg_b.movie_id
            LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE r.movie_id_a = ?
              AND mg_b.genre_id IN (SELECT genre_id FROM movie_genres WHERE movie_id = ?)
            GROUP BY m.movie_id
            ORDER BY r.lift DESC, r.confidence DESC
            LIMIT ?
        """
        cur.execute(query, (mid, mid, limit * 3))
        results = []
        for r in cur.fetchall():
            num_r = r[6] or 0
            cand_genres = set(r[7].split(",")) if r[7] else set()

            # If source movie is non-comedy, drop any comedy candidates
            if not is_source_comedy and "Comedy" in cand_genres:
                continue

            results.append({
                "movie_id": r[0],
                "title": r[1],
                "year": r[2],
                "confidence": r[3],
                "lift": r[4],
                "avg_rating": r[5] if (r[5] is not None and num_r > 0) else None,
                "num_ratings": num_r
            })
            if len(results) >= limit:
                break

        conn.close()

        return {
            "type": "association_rules",
            "source_movie": {"movie_id": mid, "title": found_title, "year": year},
            "results": results
        }

    def get_top_movies_by_genre(self, genre: str, limit: int = 10, sort_by: str = "rating") -> Dict[str, Any]:
        """Queries top movies in a specific genre."""
        conn = self.connect_ro()
        cur = conn.cursor()

        order_clause = "ms.avg_rating DESC" if sort_by == "rating" else "ms.popularity_rank ASC"
        query = f"""
            SELECT m.movie_id, m.title, m.year, g.name as genre, ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
            FROM movies m
            JOIN movie_genres mg ON m.movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.genre_id
            JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE LOWER(g.name) = ? AND ms.num_ratings > 0
            ORDER BY {order_clause}
            LIMIT ?
        """
        cur.execute(query, (genre.lower(), limit))
        results = [
            {
                "movie_id": r[0],
                "title": r[1],
                "year": r[2],
                "genre": r[3],
                "avg_rating": r[4],
                "num_ratings": r[5],
                "pct_positive": r[6],
                "popularity_rank": r[7]
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return {"type": "genre_top", "genre": genre, "sort_by": sort_by, "results": results}

    def get_top_movies(self, limit: int = 10, sort_by: str = "rating") -> Dict[str, Any]:
        """Queries overall top movies."""
        conn = self.connect_ro()
        cur = conn.cursor()

        order_clause = "ms.avg_rating DESC" if sort_by == "rating" else "ms.popularity_rank ASC"
        query = f"""
            SELECT m.movie_id, m.title, m.year, ms.avg_rating, ms.num_ratings, ms.pct_positive, ms.popularity_rank
            FROM movies m
            JOIN movie_stats ms ON m.movie_id = ms.movie_id
            WHERE ms.num_ratings >= 100
            ORDER BY {order_clause}
            LIMIT ?
        """
        cur.execute(query, (limit,))
        results = [
            {
                "movie_id": r[0],
                "title": r[1],
                "year": r[2],
                "avg_rating": r[3],
                "num_ratings": r[4],
                "pct_positive": r[5],
                "popularity_rank": r[6]
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return {"type": "overall_top", "sort_by": sort_by, "results": results}

    def query_langchain_sql_agent(self, query: str) -> Dict[str, Any]:
        """Fallback for complex queries using LangChain SQLAgent over a read-only connection."""
        try:
            from sqlalchemy import create_engine
            from langchain_community.utilities import SQLDatabase
        except ImportError as e:
            return {"type": "llm_fallback_error", "error": f"LangChain/SQLAlchemy dependencies missing: {e}"}

        engine = create_engine(f"sqlite:///{self.db_path.resolve()}?mode=ro", connect_args={"uri": True})
        db = SQLDatabase(engine)

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return {
                "type": "llm_fallback_info",
                "message": "LLM API key not configured. Executing read-only schema inspection.",
                "usable_tables": db.get_usable_table_names(),
                "query": query
            }

        try:
            from langchain_community.agent_toolkits import create_sql_agent
            if os.environ.get("GEMINI_API_KEY"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
            else:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

            agent_executor = create_sql_agent(llm, db=db, verbose=False)
            res = agent_executor.invoke({"input": query})
            return {"type": "llm_sql_result", "output": res.get("output", "")}
        except Exception as err:
            return {"type": "llm_fallback_error", "error": str(err)}

    def route(self, query: str) -> Dict[str, Any]:
        result = self.match_deterministic(query)
        if result:
            result["routing_path"] = "deterministic_fast_path"
            return result

        res = self.query_langchain_sql_agent(query)
        res["routing_path"] = "read_only_sql_agent"
        return res

if __name__ == "__main__":
    router = QueryRouter()
    
    print("--- Test 1: Association Rules ('movies like Toy Story') ---")
    res1 = router.route("movies like Toy Story")
    print("Routing Path:", res1["routing_path"])
    if res1.get("results"):
        for r in res1["results"][:3]:
            print(f"  -> {r['title']} ({r['year']}) [Lift: {r['lift']}, Confidence: {r['confidence']}]")

    print("\n--- Test 2: Top Genre ('top 5 horror movies') ---")
    res2 = router.route("top 5 horror movies")
    print("Routing Path:", res2["routing_path"])
    if res2.get("results"):
        for r in res2["results"][:3]:
            print(f"  -> {r['title']} ({r['year']}) ★ {r['avg_rating']}")

    print("\n--- Test 3: Novel Query Fallback ---")
    res3 = router.route("What is the average rating of movies released in 1999?")
    print("Routing Path:", res3["routing_path"], "|", res3.get("type"))
