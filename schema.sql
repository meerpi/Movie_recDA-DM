CREATE TABLE movies(
    movie_id    INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    year        INTEGER,
    runtime     INTEGER,
    tmdb_id     INTEGER UNIQUE,
    imdb_id     TEXT UNIQUE,
    budget      INTEGER,
    overview    TEXT

);

CREATE TABLE genres(
    genre_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL
);

CREATE TABLE movie_genres (
    movie_id    INTEGER REFERENCES movies(movie_id),
    genre_id    INTEGER REFERENCES genres(genre_id),
    PRIMARY KEY (movie_id, genre_id)
);

CREATE TABLE ratings (
    user_id     INTEGER,
    movie_id    INTEGER REFERENCES movies(movie_id),
    rating      REAL,
    rated_at    INTEGER,             -- unix timestamp
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE reviews (
    review_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_id    INTEGER REFERENCES movies(movie_id),
    source      TEXT,               -- 'nyt', 'letterboxd', 'imdb'
    domain      TEXT,               -- 'critic' or 'audience'
    review_text TEXT,
    score       REAL,
    review_date TEXT
);

-- Vector table (added once sqlite-vec is loaded)
-- v_critic and v_audience stored as BLOB (float32 arrays)
CREATE TABLE movie_embeddings (
    movie_id        INTEGER PRIMARY KEY,
    v_critic        BLOB,
    v_audience      BLOB,
    v_genome        BLOB,        -- NULL if not covered
    has_genome      INTEGER DEFAULT 0  -- explicit flag, 0 or 1
);
