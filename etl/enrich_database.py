import json
import sqlite3
import html
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
WIKI_JSONL = PROJECT_ROOT / "movie_descriptions.jsonl"
EBERT_JSONL = PROJECT_ROOT / "rogerebert_reviews.jsonl"
IMDB_JSONL = PROJECT_ROOT / "imdb_user_reviews.jsonl"

def setup_database_schema(conn):
    """Ensure the target columns for Wikipedia data exist in the movies table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(movies)")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Alter schema if wiki columns are missing
    altered = False
    if "wiki_intro" not in columns:
        conn.execute("ALTER TABLE movies ADD COLUMN wiki_intro TEXT")
        print("  [Schema] Added 'wiki_intro' column to 'movies' table.")
        altered = True
    if "wiki_plot" not in columns:
        conn.execute("ALTER TABLE movies ADD COLUMN wiki_plot TEXT")
        print("  [Schema] Added 'wiki_plot' column to 'movies' table.")
        altered = True
    
    if not altered:
        print("  [Schema] Wikipedia columns already exist in 'movies' table.")

def load_wikipedia_plots(conn):
    """Load and update movies table with Wikipedia intro and plot details."""
    if not WIKI_JSONL.exists():
        print(f"  [warn] Wikipedia data file not found at: {WIKI_JSONL}")
        return
        
    print("Starting Wikipedia ETL...")
    cursor = conn.cursor()
    
    # Read and update in batches for high speed
    batch_size = 1000
    batch = []
    updated_count = 0
    
    with open(WIKI_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                movie_id = record.get("movie_id")
                intro = record.get("intro")
                plot = record.get("plot")
                
                if movie_id is None:
                    continue
                    
                # Clean text strings (unescape HTML codes like &#39;)
                clean_intro = html.unescape(intro) if intro else None
                clean_plot = html.unescape(plot) if plot else None
                
                batch.append((clean_intro, clean_plot, int(movie_id)))
                
                if len(batch) >= batch_size:
                    cursor.executemany(
                        "UPDATE movies SET wiki_intro = ?, wiki_plot = ? WHERE movie_id = ?",
                        batch
                    )
                    conn.commit()
                    updated_count += len(batch)
                    print(f"  Processed {updated_count:,} Wikipedia plots...")
                    batch = []
            except Exception as e:
                print(f"  [warn] Error processing Wikipedia line: {e}", file=sys.stderr)
                
        # Process remaining
        if batch:
            cursor.executemany(
                "UPDATE movies SET wiki_intro = ?, wiki_plot = ? WHERE movie_id = ?",
                batch
            )
            conn.commit()
            updated_count += len(batch)
            
    print(f"Finished Wikipedia ETL! Successfully updated {updated_count:,} movies.")

def load_ebert_reviews(conn):
    """Load and insert Roger Ebert reviews into the reviews table."""
    if not EBERT_JSONL.exists():
        print(f"  [warn] Roger Ebert review file not found at: {EBERT_JSONL}")
        return
        
    print("Starting Roger Ebert ETL...")
    cursor = conn.cursor()
    
    # We want to clear out any old Roger Ebert reviews to prevent duplicates on rerun
    cursor.execute("DELETE FROM reviews WHERE source = 'rogerebert'")
    conn.commit()
    
    batch_size = 1000
    batch = []
    inserted_count = 0
    
    with open(EBERT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                movie_id = record.get("movie_id")
                review_text = record.get("review_text")
                rating = record.get("rating")
                date = record.get("date_published")
                
                if movie_id is None or not review_text:
                    continue
                    
                # Clean and parse elements
                clean_text = html.unescape(review_text)
                try:
                    score = float(rating) if rating is not None else None
                except ValueError:
                    score = None
                    
                # Insert schema layout matching schema.sql:
                # (movie_id, source, domain, review_text, score, review_date)
                batch.append((
                    int(movie_id),
                    "rogerebert",
                    "critic",
                    clean_text,
                    score,
                    date
                ))
                
                if len(batch) >= batch_size:
                    cursor.executemany(
                        "INSERT INTO reviews (movie_id, source, domain, review_text, score, review_date) VALUES (?, ?, ?, ?, ?, ?)",
                        batch
                    )
                    conn.commit()
                    inserted_count += len(batch)
                    print(f"  Processed {inserted_count:,} Roger Ebert reviews...")
                    batch = []
            except Exception as e:
                print(f"  [warn] Error processing Roger Ebert line: {e}", file=sys.stderr)
                
        # Process remaining
        if batch:
            cursor.executemany(
                "INSERT INTO reviews (movie_id, source, domain, review_text, score, review_date) VALUES (?, ?, ?, ?, ?, ?)",
                batch
            )
            conn.commit()
            inserted_count += len(batch)
            
    print(f"Finished Roger Ebert ETL! Successfully inserted {inserted_count:,} reviews.")

def load_imdb_reviews(conn):
    """Load and insert IMDb user reviews into the reviews table."""
    if not IMDB_JSONL.exists():
        print(f"  [info] IMDb user reviews file not found at: {IMDB_JSONL}. Skipping.")
        return
        
    print("Starting IMDb User Reviews ETL...")
    cursor = conn.cursor()
    
    # Clear out old IMDb reviews to prevent duplicates on rerun
    cursor.execute("DELETE FROM reviews WHERE source = 'imdb'")
    conn.commit()
    
    batch_size = 1000
    batch = []
    inserted_count = 0
    
    with open(IMDB_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                movie_id = record.get("movie_id")
                reviews = record.get("reviews", [])
                
                if movie_id is None or not reviews:
                    continue
                    
                for rev in reviews:
                    summary = rev.get("summary") or ""
                    text = rev.get("text") or ""
                    rating = rev.get("rating")
                    date = rev.get("date")
                    
                    full_text = f"{summary}\n\n{text}".strip()
                    if not full_text:
                        continue
                        
                    clean_text = html.unescape(full_text)
                    try:
                        score = float(rating) if rating is not None else None
                    except ValueError:
                        score = None
                        
                    # Insert matching:
                    # (movie_id, source, domain, review_text, score, review_date)
                    batch.append((
                        int(movie_id),
                        "imdb",
                        "audience",
                        clean_text,
                        score,
                        date
                    ))
                    
                    if len(batch) >= batch_size:
                        cursor.executemany(
                            "INSERT INTO reviews (movie_id, source, domain, review_text, score, review_date) VALUES (?, ?, ?, ?, ?, ?)",
                            batch
                        )
                        conn.commit()
                        inserted_count += len(batch)
                        batch = []
            except Exception as e:
                print(f"  [warn] Error processing IMDb review line: {e}", file=sys.stderr)
                
        # Process remaining
        if batch:
            cursor.executemany(
                "INSERT INTO reviews (movie_id, source, domain, review_text, score, review_date) VALUES (?, ?, ?, ?, ?, ?)",
                batch
            )
            conn.commit()
            inserted_count += len(batch)
            
    print(f"Finished IMDb ETL! Successfully inserted {inserted_count:,} user reviews.")

def main():
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    
    try:
        setup_database_schema(conn)
        load_wikipedia_plots(conn)
        load_ebert_reviews(conn)
        load_imdb_reviews(conn)
        
        # Optimize database with VACUUM/ANALYZE post-load
        print("Optimizing database storage...")
        conn.execute("ANALYZE")
        conn.commit()
        print("ETL complete and optimized!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
