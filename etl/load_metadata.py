import csv
import html
import json
import re
import sqlite3
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
TOP_CSV = PROJECT_ROOT / "top_10000_movies.csv"
RT_CSV = PROJECT_ROOT / "archive(1)" / "rotten_tomatoes_movies.csv"
LINKS_CSV = PROJECT_ROOT / "data" / "ml-25m" / "links.csv"
TODO_JSON = PROJECT_ROOT / "metadata_todo.json"

def setup_schema(conn):
    """Ensure the target metadata columns exist in the movies table."""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(movies)")
    columns = [col[1] for col in cursor.fetchall()]
    
    new_cols = {
        "content_rating": "TEXT",
        "directors": "TEXT",
        "actors": "TEXT",
        "writers": "TEXT",
        "rt_critic_score": "REAL",
        "rt_audience_score": "REAL"
    }
    
    for col_name, col_type in new_cols.items():
        if col_name not in columns:
            conn.execute(f"ALTER TABLE movies ADD COLUMN {col_name} {col_type}")
            print(f"  [Schema] Added '{col_name}' column to 'movies' table.")

def format_imdb_id(raw_id: str) -> str:
    raw_id = raw_id.strip()
    if not raw_id:
        return ""
    if len(raw_id) < 7:
        raw_id = raw_id.zfill(7)
    return raw_id

def load_id_mappings():
    if not LINKS_CSV.exists():
        print(f"Error: {LINKS_CSV} not found.", file=sys.stderr)
        return {}
    
    mapping = {}
    with open(LINKS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m_id = row.get("movieId")
            imdb_id = row.get("imdbId")
            if m_id and imdb_id:
                mapping[int(m_id)] = format_imdb_id(imdb_id)
    return mapping

def normalize_title(title):
    if not title:
        return ""
    title = title.lower().strip()
    
    # Strip leading articles
    for article in ["the ", "a ", "an "]:
        if title.startswith(article):
            title = title[len(article):].strip()
            break
            
    # Strip trailing articles
    for suffix in [", the", ", a", ", an", " the", " a", " an"]:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
            break
            
    title = re.sub(r'[^a-z0-9]', '', title)
    return title

def get_variants(title):
    variants = set()
    if not title:
        return variants
    
    variants.add(normalize_title(title))
    
    if '(' in title:
        parts = title.split('(')
        main_title = parts[0].strip()
        variants.add(normalize_title(main_title))
        
        paren_content = parts[1].split(')')[0].strip()
        if paren_content.lower().startswith("a.k.a."):
            aka_title = paren_content[6:].strip()
            variants.add(normalize_title(aka_title))
        else:
            variants.add(normalize_title(paren_content))
            
    return [v for v in variants if v]

def load_rt_movies():
    if not RT_CSV.exists():
        print(f"Error: {RT_CSV} not found.", file=sys.stderr)
        return {}
        
    print("Loading Rotten Tomatoes movies...")
    rt_movies = {}
    with open(RT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("movie_title")
            date = row.get("original_release_date")
            if not title or not date:
                continue
            year_match = re.search(r'\d{4}', date)
            if not year_match:
                continue
            year = int(year_match.group())
            norm = normalize_title(title)
            
            # Store in dict
            rt_movies[(norm, year)] = row
    print(f"Loaded {len(rt_movies):,} Rotten Tomatoes records.")
    return rt_movies

def run_etl():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode = WAL")
    setup_schema(conn)
    
    id_mapping = load_id_mappings()
    rt_movies = load_rt_movies()
    
    unmatched = []
    matched_count = 0
    total_count = 0
    
    cursor = conn.cursor()
    
    with open(TOP_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_count += 1
            m_id = int(row["movie_id"])
            title = row["title"]
            year_str = row["year"]
            
            if not year_str:
                unmatched.append({"movie_id": m_id, "title": title, "imdb_id": id_mapping.get(m_id)})
                continue
                
            year = int(year_str)
            # Try to match
            rt_row = None
            variants = get_variants(title)
            for var in variants:
                for y_offset in [0, -1, 1]:
                    rt_row = rt_movies.get((var, year + y_offset))
                    if rt_row:
                        break
                if rt_row:
                    break
                    
            if rt_row:
                matched_count += 1
                
                # Parse scores
                try:
                    critic_score = float(rt_row.get("tomatometer_rating")) if rt_row.get("tomatometer_rating") else None
                except ValueError:
                    critic_score = None
                    
                try:
                    audience_score = float(rt_row.get("audience_rating")) if rt_row.get("audience_rating") else None
                except ValueError:
                    audience_score = None
                
                # Directors, writers (authors), and actors (cast)
                directors = rt_row.get("directors") or ""
                writers = rt_row.get("authors") or ""
                actors = rt_row.get("actors") or ""
                content_rating = rt_row.get("content_rating") or ""
                
                cursor.execute(
                    """
                    UPDATE movies
                    SET content_rating = ?,
                        directors = ?,
                        actors = ?,
                        writers = ?,
                        rt_critic_score = ?,
                        rt_audience_score = ?
                    WHERE movie_id = ?
                    """,
                    (content_rating, directors, actors, writers, critic_score, audience_score, m_id)
                )
            else:
                unmatched.append({
                    "movie_id": m_id,
                    "title": title,
                    "imdb_id": id_mapping.get(m_id),
                    "year": year
                })
                
    conn.commit()
    
    # Load any already scraped IMDb metadata from output jsonl file
    load_imdb_metadata(conn)
    
    conn.close()
    
    print(f"\nETL Summary:")
    print(f"  Total MovieLens movies: {total_count:,}")
    print(f"  Successfully matched & enriched: {matched_count:,} ({matched_count/total_count*100:.2f}%)")
    print(f"  Unmatched: {len(unmatched):,}")
    
    # Save unmatched to JSON todo list
    with open(TODO_JSON, "w", encoding="utf-8") as f:
        json.dump(unmatched, f, indent=2, ensure_ascii=False)
    print(f"Saved todo list of {len(unmatched)} unmatched movies to {TODO_JSON.name}")

def load_imdb_metadata(conn):
    imdb_meta_path = PROJECT_ROOT / "imdb_metadata_results.jsonl"
    if not imdb_meta_path.exists():
        print("  [info] No scraped IMDb metadata file found. Skipping.")
        return
        
    print("Loading scraped IMDb metadata...")
    cursor = conn.cursor()
    count = 0
    with open(imdb_meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                movie_id = rec.get("movie_id")
                content_rating = rec.get("content_rating") or ""
                directors = rec.get("directors") or ""
                actors = rec.get("actors") or ""
                writers = rec.get("writers") or ""
                
                cursor.execute(
                    """
                    UPDATE movies
                    SET content_rating = ?,
                        directors = ?,
                        actors = ?,
                        writers = ?
                    WHERE movie_id = ?
                    """,
                    (content_rating, directors, actors, writers, int(movie_id))
                )
                count += 1
            except Exception as e:
                print(f"  [warn] Error loading IMDb metadata line: {e}", file=sys.stderr)
                
    conn.commit()
    print(f"Successfully loaded {count} scraped IMDb metadata records into the database.")

if __name__ == "__main__":
    run_etl()
