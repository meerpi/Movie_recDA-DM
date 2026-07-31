-- ============================================================
-- schema.sql — CineVault Database Schema Reference
-- ============================================================
-- Canonical documentation of all 16 tables in cinevault.db.
-- Tables marked [ETL] are populated by ETL scripts.
-- Tables marked [RUNTIME] are created/managed at runtime.
-- Tables marked [PRECOMPUTED] are built by enrichment scripts.
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- CORE CATALOG [ETL]
-- ─────────────────────────────────────────────────────────────

CREATE TABLE movies (
    movie_id    INTEGER PRIMARY KEY,    -- MovieLens movie_id
    title       TEXT NOT NULL,
    year        INTEGER,
    runtime     INTEGER,                -- minutes
    tmdb_id     INTEGER UNIQUE,
    imdb_id     TEXT UNIQUE,
    budget      INTEGER,
    overview    TEXT
);

CREATE TABLE genres (
    genre_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL
);

CREATE TABLE movie_genres (
    movie_id    INTEGER REFERENCES movies(movie_id),
    genre_id    INTEGER REFERENCES genres(genre_id),
    PRIMARY KEY (movie_id, genre_id)
);

-- External ID linkage table (MovieLens ↔ TMDB ↔ IMDB)
CREATE TABLE links (
    movie_id    INTEGER PRIMARY KEY REFERENCES movies(movie_id),
    imdb_id     TEXT,
    tmdb_id     INTEGER
);


-- ─────────────────────────────────────────────────────────────
-- RATINGS & REVIEWS [ETL + RUNTIME]
-- ─────────────────────────────────────────────────────────────

-- MovieLens 25M crowd ratings + user-submitted ratings (via TUI)
CREATE TABLE ratings (
    user_id     INTEGER,
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      REAL,
    rated_at    INTEGER,                -- unix timestamp
    PRIMARY KEY (user_id, movie_id)
);

-- Critic & audience text reviews (NYT, Letterboxd, IMDb) + TUI submissions
CREATE TABLE reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id    INTEGER REFERENCES movies(movie_id),
    source      TEXT,                   -- 'nyt', 'letterboxd', 'imdb', 'cinevault_user'
    domain      TEXT,                   -- 'critic' or 'audience'
    review_text TEXT,
    score       REAL,
    review_date TEXT
);

-- User-generated tags from TUI review submissions [RUNTIME]
-- user_id stored as hash(user_id_string) % 1000000 (INTEGER)
CREATE TABLE user_tags (
    user_id     INTEGER,                -- hashed from string user_id
    movie_id    INTEGER REFERENCES movies(movie_id),
    tag         TEXT,
    tagged_at   INTEGER                 -- unix timestamp
);
CREATE INDEX IF NOT EXISTS idx_user_tags_user   ON user_tags(user_id);
CREATE INDEX IF NOT EXISTS idx_user_tags_movie  ON user_tags(movie_id);
CREATE INDEX IF NOT EXISTS idx_user_tags_triple ON user_tags(user_id, movie_id, tag);


-- ─────────────────────────────────────────────────────────────
-- GENOME SYSTEM [ETL — MovieLens Tag Genome]
-- ─────────────────────────────────────────────────────────────

-- 1,128 genome tag vocabulary
CREATE TABLE genome_tags (
    tag_id      INTEGER PRIMARY KEY,
    tag         TEXT UNIQUE NOT NULL
);

-- Per-movie relevance score for each of the 1,128 genome tags (float 0.0-1.0)
-- 62,423 movies × 1,128 tags = ~70M rows
CREATE TABLE genome_scores (
    movie_id    INTEGER REFERENCES movies(movie_id),
    tag_id      INTEGER REFERENCES genome_tags(tag_id),
    relevance   REAL NOT NULL,
    PRIMARY KEY (movie_id, tag_id)
);


-- ─────────────────────────────────────────────────────────────
-- PRECOMPUTED SIGNALS [PRECOMPUTED]
-- ─────────────────────────────────────────────────────────────

-- Aggregated rating statistics per movie (precomputed from ratings table)
CREATE TABLE movie_stats (
    movie_id        INTEGER PRIMARY KEY REFERENCES movies(movie_id),
    avg_rating      REAL,
    num_ratings     INTEGER,
    pct_positive    REAL,               -- fraction with rating >= 4.0
    popularity_rank INTEGER             -- rank by num_ratings desc
);

-- Association rules mined from co-rating patterns (Apriori/FP-Growth)
-- Used by the fast-path Router for "movies like X" queries
CREATE TABLE association_rules (
    antecedent_id   INTEGER REFERENCES movies(movie_id),
    consequent_id   INTEGER REFERENCES movies(movie_id),
    support         REAL,
    confidence      REAL,
    lift            REAL,
    PRIMARY KEY (antecedent_id, consequent_id)
);

-- Dense embedding vectors for semantic search (VoyageAI + genome HNSW)
-- v_critic / v_audience: 1024d VoyageAI float32 BLOBs
-- v_genome: 1128d MovieLens genome vector BLOB
CREATE TABLE movie_embeddings (
    movie_id        INTEGER PRIMARY KEY REFERENCES movies(movie_id),
    v_critic        BLOB,
    v_audience      BLOB,
    v_genome        BLOB,               -- NULL if movie not covered by genome
    has_genome      INTEGER DEFAULT 0   -- 0 or 1 explicit flag
);


-- ─────────────────────────────────────────────────────────────
-- USER PROFILES [RUNTIME]
-- ─────────────────────────────────────────────────────────────

-- Serialized UserProfile JSON blobs — one row per named user
-- All 6 signal layers stored in a single JSON document for portability.
-- Schema managed by: user_profile/schema.py (UserProfile.to_dict/from_dict)
-- Storage managed by: user_profile/store.py (UserProfileStore)
CREATE TABLE user_profiles (
    user_id         TEXT PRIMARY KEY,
    profile_json    TEXT NOT NULL,      -- full UserProfile serialized to JSON
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
