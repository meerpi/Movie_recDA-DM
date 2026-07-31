#!/usr/bin/env python3
"""
scripts/ingest_profile_cards.py — Ingest enriched profile cards into cinevault.db

Creates two new structures in cinevault.db:
  1. profile_cards table — Gemini/Voyage enriched semantic metadata for 20,185 movies
  2. movies_fts FTS5 table — BM25 full-text search over title + themes + tone + comparables

Sources (priority order — highest quality wins):
  Tier A: 9,526  — Gemini-extracted (source='gemini')
  Tier B: 4,249  — Voyage k-NN wiki/imdb/synthetic (source='voyage_wiki' or 'voyage_synthetic')
  Tier C: 6,410  — Voyage k-NN wiki (source='voyage_wiki')

Total: 20,185 profile cards ingested
"""

from __future__ import annotations
import json
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH      = PROJECT_ROOT / "db" / "cinevault.db"

TIER_A_FILE  = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_FILE  = PROJECT_ROOT / "dirtywork" / "tier_b_voyage_cards.jsonl"
TIER_C_FILE  = PROJECT_ROOT / "dirtywork" / "tier_c_voyage_cards.jsonl"


def load_cards(path: Path, tier: str) -> dict:
    cards = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                mid = int(c["movie_id"])
                c["_tier"] = tier
                cards[mid] = c
            except Exception:
                continue
    return cards


def main():
    print("=" * 64)
    print("Ingesting enriched profile cards into cinevault.db")
    print("=" * 64)

    # ── Load all enriched cards (Tier A wins over B wins over C on conflict) ──
    print("\n[Step 1] Loading enriched profile cards...")
    cards_c = load_cards(TIER_C_FILE, "C")
    cards_b = load_cards(TIER_B_FILE, "B")
    cards_a = load_cards(TIER_A_FILE, "A")

    # Merge: A overrides B overrides C
    merged = {}
    merged.update(cards_c)
    merged.update(cards_b)
    merged.update(cards_a)

    print(f"  Tier A: {len(cards_a):,} | Tier B: {len(cards_b):,} | Tier C: {len(cards_c):,}")
    print(f"  Total merged (unique movies): {len(merged):,}")

    # ── Connect to cinevault.db ──
    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    cur = con.cursor()

    # ── Step 2: Create profile_cards table ──
    print("\n[Step 2] Creating profile_cards table...")
    cur.executescript("""
        DROP TABLE IF EXISTS profile_cards;

        CREATE TABLE profile_cards (
            movie_id                INTEGER PRIMARY KEY REFERENCES movies(movie_id),
            themes                  TEXT,
            tone                    TEXT,
            moral_complexity        TEXT,
            directorial_style_notes TEXT,
            comparable_films        TEXT,
            standout_performances   TEXT,
            enrichment_tier         TEXT,
            enrichment_source       TEXT,
            voyage_confidence       REAL,
            low_confidence          INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_pc_tier       ON profile_cards(enrichment_tier);
        CREATE INDEX IF NOT EXISTS idx_pc_confidence ON profile_cards(voyage_confidence);
    """)
    con.commit()
    print("  profile_cards table created.")

    # ── Step 3: Ingest cards ──
    print("\n[Step 3] Ingesting profile cards...")

    def to_json(v):
        if v is None:
            return None
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return v

    def determine_source(card: dict, tier: str) -> str:
        if tier == "A":
            return "gemini"
        src = card.get("enrichment_text_source", "")
        if "wiki" in src.lower():
            return "voyage_wiki"
        if "imdb" in src.lower():
            return "voyage_imdb_review"
        return "voyage_synthetic"

    rows = []
    for mid, card in merged.items():
        tier = card.get("_tier", "?")
        rows.append((
            mid,
            to_json(card.get("themes")),
            to_json(card.get("tone")),
            card.get("moral_complexity"),
            card.get("directorial_style_notes"),
            to_json(card.get("comparable_films")),
            to_json(card.get("standout_performances")),
            tier,
            determine_source(card, tier),
            card.get("nearest_tier_a_max_sim"),
            1 if card.get("low_confidence") else 0,
        ))

    cur.executemany("""
        INSERT OR REPLACE INTO profile_cards
        (movie_id, themes, tone, moral_complexity, directorial_style_notes,
         comparable_films, standout_performances, enrichment_tier, enrichment_source,
         voyage_confidence, low_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()
    print(f"  Inserted {len(rows):,} profile cards.")

    # ── Step 4: Build FTS5 full-text search index ──
    print("\n[Step 4] Building FTS5 full-text search index (movies_fts)...")
    cur.executescript("""
        DROP TABLE IF EXISTS movies_fts;

        CREATE VIRTUAL TABLE movies_fts USING fts5(
            movie_id  UNINDEXED,
            title,
            overview,
            themes,
            tone,
            comparable_films,
            directors,
            tokenize = 'porter unicode61'
        );
    """)

    # Populate FTS from JOIN of movies + profile_cards
    cur.execute("""
        INSERT INTO movies_fts (movie_id, title, overview, themes, tone, comparable_films, directors)
        SELECT
            m.movie_id,
            m.title,
            COALESCE(m.overview, ''),
            COALESCE(pc.themes, ''),
            COALESCE(pc.tone, ''),
            COALESCE(pc.comparable_films, ''),
            COALESCE(m.directors, '')
        FROM movies m
        LEFT JOIN profile_cards pc ON m.movie_id = pc.movie_id
    """)
    con.commit()

    fts_count = cur.execute("SELECT COUNT(*) FROM movies_fts").fetchone()[0]
    print(f"  FTS5 index built: {fts_count:,} documents indexed.")

    # ── Step 5: Verify ──
    print("\n[Step 5] Verification...")
    a_count = cur.execute("SELECT COUNT(*) FROM profile_cards WHERE enrichment_tier='A'").fetchone()[0]
    b_count = cur.execute("SELECT COUNT(*) FROM profile_cards WHERE enrichment_tier='B'").fetchone()[0]
    c_count = cur.execute("SELECT COUNT(*) FROM profile_cards WHERE enrichment_tier='C'").fetchone()[0]
    low_conf = cur.execute("SELECT COUNT(*) FROM profile_cards WHERE low_confidence=1").fetchone()[0]

    print(f"  profile_cards breakdown:")
    print(f"    Tier A (Gemini):        {a_count:,}")
    print(f"    Tier B (Voyage):        {b_count:,}")
    print(f"    Tier C (Voyage wiki):   {c_count:,}")
    print(f"    Low confidence flagged: {low_conf:,}")

    # Test FTS5 search
    results = cur.execute("""
        SELECT m.title, pc.themes
        FROM movies_fts fts
        JOIN movies m ON m.movie_id = fts.movie_id
        LEFT JOIN profile_cards pc ON pc.movie_id = fts.movie_id
        WHERE movies_fts MATCH 'isolation survival post-apocalyptic'
        ORDER BY rank
        LIMIT 4
    """).fetchall()
    print(f"\n  FTS5 test query: 'isolation survival post-apocalyptic'")
    for title, themes in results:
        print(f"    -> {title} | themes: {themes}")

    con.close()
    print(f"\n{'='*64}")
    print(f"DONE! cinevault.db updated:")
    print(f"  + profile_cards table: {len(rows):,} enriched movie cards")
    print(f"  + movies_fts table:    {fts_count:,} documents (all 62,423 movies)")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
