#!/usr/bin/env python3
"""
etl/compute_stats_and_rules.py — Step 13: Batch Analytics & Association Rules

Uses DuckDB as an ephemeral analytics engine to crunch 25M MovieLens ratings:
 1. Computes movie_stats (OLAP)
 2. Computes association rules (Data Mining Apriori self-join)
 3. Writes them back to db/cinevault.db using Python's sqlite3 executemany
    inside an idempotent transaction (staging tables -> rename).
"""

import sqlite3
import time
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("[ERROR] duckdb is not installed. Run: pip install duckdb")
    exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
RATINGS_CSV = PROJECT_ROOT / "data" / "ml-25m" / "ratings.csv"
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"

def compute_and_load():
    t0 = time.time()
    print("🚀 Initializing DuckDB memory engine...")
    con = duckdb.connect(database=':memory:')

    if not RATINGS_CSV.exists():
        print(f"[ERROR] {RATINGS_CSV} not found.")
        return

    print("📥 Loading ratings CSV into DuckDB memory engine...")
    con.execute(f"CREATE TEMP TABLE raw_ratings AS SELECT userId, movieId, rating FROM read_csv_auto('{RATINGS_CSV}')")

    # ---------------------------------------------------------
    # 1. Compute Movie Stats
    # ---------------------------------------------------------
    print("📊 Computing OLAP Movie Stats...")
    stats_query = """
        SELECT 
            movieId AS movie_id,
            ROUND(AVG(rating), 3) AS avg_rating,
            COUNT(rating) AS num_ratings,
            ROUND(SUM(CASE WHEN rating >= 4.0 THEN 1 ELSE 0 END) * 100.0 / COUNT(rating), 2) AS pct_positive,
            RANK() OVER (ORDER BY COUNT(rating) DESC, AVG(rating) DESC) AS popularity_rank
        FROM raw_ratings
        GROUP BY movieId
    """
    movie_stats = con.execute(stats_query).fetchall()
    print(f"  ✓ Computed stats for {len(movie_stats):,} movies.")

    # ---------------------------------------------------------
    # 2. Compute Association Rules (Apriori)
    # ---------------------------------------------------------
    print("🔗 Computing Association Rules (Apriori-pruned self-join)...")
    
    # We define "liked" as a rating >= 4.0
    # To prevent out-of-memory on self-join, we calculate total_users
    total_users_query = "SELECT COUNT(DISTINCT userId) FROM raw_ratings WHERE rating >= 4.0"
    total_users = con.execute(total_users_query).fetchone()[0]

    rules_query = f"""
        WITH raw_likes AS (
            SELECT userId, movieId 
            FROM raw_ratings 
            WHERE rating >= 4.0
            QUALIFY ROW_NUMBER() OVER(PARTITION BY userId ORDER BY rating DESC) <= 50
        ),
        frequent_movies AS (
            SELECT movieId, COUNT(DISTINCT userId) AS total_likes 
            FROM raw_likes 
            GROUP BY movieId
            HAVING COUNT(DISTINCT userId) >= 200
        ),
        highly_rated AS (
            SELECT r.userId, r.movieId 
            FROM raw_likes r
            JOIN frequent_movies fm ON r.movieId = fm.movieId
        ),
        co_likes AS (
            SELECT 
                a.movieId AS movie_id_a,
                b.movieId AS movie_id_b,
                COUNT(*) AS co_likes_count
            FROM highly_rated a
            JOIN highly_rated b 
              ON a.userId = b.userId 
             AND a.movieId != b.movieId
            GROUP BY a.movieId, b.movieId
            HAVING COUNT(*) >= 200
        ),
        scored_rules AS (
            SELECT 
                c.movie_id_a,
                c.movie_id_b,
                (c.co_likes_count * 1.0 / ca.total_likes) AS confidence,
                (c.co_likes_count * 1.0 * {total_users}) / (ca.total_likes * cb.total_likes) AS lift
            FROM co_likes c
            JOIN frequent_movies ca ON c.movie_id_a = ca.movieId
            JOIN frequent_movies cb ON c.movie_id_b = cb.movieId
        ),
        ranked_rules AS (
            SELECT 
                movie_id_a,
                movie_id_b,
                confidence,
                lift,
                ROW_NUMBER() OVER(PARTITION BY movie_id_a ORDER BY lift DESC, confidence DESC) as rnk
            FROM scored_rules
            -- Only consider rules with positive lift
            WHERE lift > 1.0
        )
        SELECT 
            movie_id_a, 
            movie_id_b, 
            ROUND(confidence, 4) AS confidence, 
            ROUND(lift, 4) AS lift
        FROM ranked_rules 
        WHERE rnk <= 15
    """
    t_rules = time.time()
    association_rules = con.execute(rules_query).fetchall()
    print(f"  ✓ Computed {len(association_rules):,} association rules in {time.time() - t_rules:.1f}s.")

    con.close()

    # ---------------------------------------------------------
    # 3. Idempotent SQLite Write-Back
    # ---------------------------------------------------------
    print(f"💾 Writing back to SQLite: {DB_PATH}")
    sqlite_con = sqlite3.connect(DB_PATH)
    cur = sqlite_con.cursor()

    try:
        cur.execute("BEGIN TRANSACTION;")

        # Create staging tables
        cur.execute("DROP TABLE IF EXISTS movie_stats_staging;")
        cur.execute("""
            CREATE TABLE movie_stats_staging (
                movie_id INTEGER PRIMARY KEY,
                avg_rating REAL,
                num_ratings INTEGER,
                pct_positive REAL,
                popularity_rank INTEGER
            );
        """)

        cur.execute("DROP TABLE IF EXISTS association_rules_staging;")
        cur.execute("""
            CREATE TABLE association_rules_staging (
                movie_id_a INTEGER,
                movie_id_b INTEGER,
                confidence REAL,
                lift REAL
            );
        """)

        # Bulk insert
        print("  -> Inserting into staging tables...")
        cur.executemany("""
            INSERT INTO movie_stats_staging (movie_id, avg_rating, num_ratings, pct_positive, popularity_rank) 
            VALUES (?, ?, ?, ?, ?)
        """, movie_stats)

        # Back-fill zero-stats rows for movies that have no ratings in the 25M dataset.
        # These are long-tail Tier C films. Without this they are silently absent from
        # movie_stats and sort with popularity_rank=999999 everywhere.
        max_rank = len(movie_stats) + 1
        cur.execute(f"""
            INSERT OR IGNORE INTO movie_stats_staging (movie_id, avg_rating, num_ratings, pct_positive, popularity_rank)
            SELECT m.movie_id, 0.0, 0, 0.0, {max_rank}
            FROM movies m
            WHERE m.movie_id NOT IN (SELECT movie_id FROM movie_stats_staging)
        """)
        backfill_count = cur.rowcount
        if backfill_count:
            print(f"  → Back-filled {backfill_count:,} zero-stats rows for unrated movies.")

        cur.executemany("""
            INSERT INTO association_rules_staging (movie_id_a, movie_id_b, confidence, lift) 
            VALUES (?, ?, ?, ?)
        """, association_rules)

        # Create Indices
        print("  -> Creating indices on join columns...")
        cur.execute("CREATE INDEX idx_rules_staging_movie_id_a ON association_rules_staging(movie_id_a);")
        # Note: movie_stats_staging.movie_id is PRIMARY KEY, so it is automatically indexed

        # Swap staging to live
        print("  -> Swapping staging tables to live...")
        cur.execute("DROP TABLE IF EXISTS movie_stats;")
        cur.execute("ALTER TABLE movie_stats_staging RENAME TO movie_stats;")

        cur.execute("DROP TABLE IF EXISTS association_rules;")
        cur.execute("ALTER TABLE association_rules_staging RENAME TO association_rules;")

        cur.execute("COMMIT;")
        print("  ✓ Transaction committed successfully.")

    except Exception as e:
        cur.execute("ROLLBACK;")
        print(f"[ERROR] Transaction failed, rolled back. Error: {e}")
        raise
    finally:
        sqlite_con.close()

    print(f"✅ Batch ETL completed in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    compute_and_load()
