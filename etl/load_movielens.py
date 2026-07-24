import sqlite3, pandas as pd, sqlite_vec
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ml-25m"
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"
SCHEMA_PATH = PROJECT_ROOT / "schema.sql"
CHUNK_SIZE = 500_000

conn = sqlite3.connect(DB_PATH)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.execute("PRAGMA journal_mode = WAL")
conn.execute("PRAGMA foreign_keys = ON")
conn.execute("PRAGMA synchronous = NORMAL")

with open(SCHEMA_PATH) as f:
    conn.executescript(f.read())

movies = pd.read_csv(DATA_DIR / "movies.csv")
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype("Int64")
movies["title"] = (
    movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
)
movies[["movieId", "title", "year"]].rename(columns={"movieId": "movie_id"}).to_sql(
    "movies", conn, if_exists="append", index=False
)

for _, row in movies.iterrows():
    if row["genres"] == "(no genres listed)":
        continue
    for g in row["genres"].split("|"):
        conn.execute("INSERT OR IGNORE INTO genres (name) VALUES (?)", (g,))
        gid = conn.execute("SELECT genre_id FROM genres WHERE name=?", (g,)).fetchone()[
            0
        ]
        conn.execute(
            "INSERT OR IGNORE INTO movie_genres VALUES (?,?)",
            (int(row["movieId"]), gid),
        )
conn.commit()
print(f"Movies: {len(movies):,} rows, genres normalized.")

# ── Links ──
links = pd.read_csv(DATA_DIR / "links.csv")
links["imdb_id"] = links["imdbId"].apply(
    lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
)
links["tmdb_id"] = links["tmdbId"].apply(lambda x: int(x) if pd.notna(x) else None)
links[["movieId", "imdb_id", "tmdb_id"]].rename(columns={"movieId": "movie_id"}).to_sql(
    "links", conn, if_exists="append", index=False
)
conn.commit()
print(f"Links: {len(links):,} rows.")

# ── Ratings ──
total = 0
for chunk in pd.read_csv(DATA_DIR / "ratings.csv", chunksize=CHUNK_SIZE):
    chunk.rename(
        columns={"userId": "user_id", "movieId": "movie_id", "timestamp": "rated_at"}
    ).to_sql("ratings", conn, if_exists="append", index=False)
    total += len(chunk)
conn.commit()
print(f"Ratings: {total:,} rows.")

# ── User Tags ──
tags = pd.read_csv(DATA_DIR / "tags.csv").dropna(subset=["tag"])
tags.rename(
    columns={"userId": "user_id", "movieId": "movie_id", "timestamp": "tagged_at"}
).to_sql("user_tags", conn, if_exists="append", index=False)
conn.commit()
print(f"User tags: {len(tags):,} rows.")

# ── Genome ──
pd.read_csv(DATA_DIR / "genome-tags.csv").rename(columns={"tagId": "tag_id"}).to_sql(
    "genome_tags", conn, if_exists="append", index=False
)
total = 0
for chunk in pd.read_csv(DATA_DIR / "genome-scores.csv", chunksize=CHUNK_SIZE):
    chunk.rename(columns={"movieId": "movie_id", "tagId": "tag_id"}).to_sql(
        "genome_scores", conn, if_exists="append", index=False
    )
    total += len(chunk)
conn.execute("""
    INSERT OR IGNORE INTO movie_embeddings (movie_id, has_genome)
    SELECT DISTINCT movie_id, 1 FROM genome_scores
""")
conn.execute("""
    INSERT OR IGNORE INTO movie_embeddings (movie_id, has_genome)
    SELECT movie_id, 0 FROM movies WHERE movie_id NOT IN (SELECT movie_id FROM movie_embeddings)
""")
conn.commit()
print(f"Genome: {total:,} score rows loaded.")
conn.close()

# ── Compute and load dense genome vectors (Tier B) ──
from load_genome_vectors import load_genome_vectors
load_genome_vectors()

print("Done. Database ready.")
