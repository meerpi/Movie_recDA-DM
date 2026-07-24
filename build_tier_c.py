#!/usr/bin/env python3
"""
build_tier_c.py — Build Tier C profile cards for all remaining MovieLens movies.

Tier C covers movies that have NEITHER:
  - an LLM profile card (Tier A), NOR
  - a genome tag-relevance vector (Tier B)

These ~48,607 movies only have:
  - genre labels from movies.csv  (pipe-separated, e.g. "Action|Adventure|Sci-Fi")
  - free-text crowdsourced tags from tags.csv (userId, movieId, tag, timestamp)

They cannot be found via semantic queries ("slow-paced existential dread") but
are still recommendable via collaborative filtering (ALS on ratings.csv) and
genre-overlap matching.

Output schema per record:
  {
    "movie_id"  : int,
    "title"     : str,
    "year"      : int | null,
    "tier"      : "C",
    "genres"    : ["Action", "Adventure", ...],
    "user_tags" : ["cult classic", "twist ending", ...]   // deduplicated, lowercased
  }

USAGE:
    python build_tier_c.py
    python build_tier_c.py --limit 200    # smoke test

RUNTIME: ~20-40s (reads tags.csv once, 1M rows).
No API calls needed — pure data transformation.
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TIER_A_CARDS   = Path("tier_a_profile_cards_v2.jsonl")
TIER_B_CARDS   = Path("tier_b_profile_cards.jsonl")
GENOME_SCORES  = Path("data/ml-25m/genome-scores.csv")   # used as fallback to determine Tier B IDs
MOVIES_CSV     = Path("data/ml-25m/movies.csv")
TAGS_CSV       = Path("data/ml-25m/tags.csv")
OUTPUT_PATH    = Path("tier_c_profile_cards.jsonl")

# Minimum number of users that applied a tag for it to be included.
# This filters noise ("garbage tag applied by 1 person") vs signal.
MIN_TAG_USERS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def extract_year(ml_title: str) -> tuple[str, int | None]:
    """Split 'Toy Story (1995)' into ('Toy Story', 1995)."""
    m = _YEAR_RE.search(ml_title)
    if m:
        year = int(m.group(1))
        clean = ml_title[: m.start()].strip()
        return clean, year
    return ml_title.strip(), None


def load_ids_from_jsonl(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["movie_id"])
                except Exception:
                    continue
    return ids


def load_genome_movie_ids(path: Path) -> set:
    """Stream genome-scores.csv to collect unique movieIds (Tier B candidates)."""
    ids = set()
    if not path.exists():
        return ids
    print("[INFO] Scanning genome-scores.csv for Tier B movie IDs ...")
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(int(row["movieId"]))
    print(f"       {len(ids)} genome-covered IDs found in {time.time()-t0:.1f}s")
    return ids


def load_user_tags(path: Path, tier_c_ids: set) -> dict:
    """
    Read tags.csv and return {movie_id: Counter({tag: user_count})}.
    Only loads tags for movies in tier_c_ids.
    """
    tag_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    print(f"[INFO] Loading user tags from {path} ...")
    t0 = time.time()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movieId"])
            if mid not in tier_c_ids:
                continue
            tag = row["tag"].strip().lower()
            if tag:
                tag_counts[mid][tag] += 1
    print(f"       Done in {time.time()-t0:.1f}s — {len(tag_counts)} movies have tags")
    return tag_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Tier C profile cards.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process N movies (smoke-test mode)")
    parser.add_argument("--min-tag-users", type=int, default=MIN_TAG_USERS,
                        help=f"Minimum users who applied a tag to include it (default: {MIN_TAG_USERS})")
    args = parser.parse_args()

    for p in [MOVIES_CSV, TAGS_CSV]:
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    # ------------------------------------------------------------------ #
    # STEP 1: Determine which movie IDs are already in Tier A or Tier B   #
    # ------------------------------------------------------------------ #
    print("[STEP 1] Loading Tier A IDs ...")
    tier_a_ids = load_ids_from_jsonl(TIER_A_CARDS)
    print(f"         {len(tier_a_ids)} Tier A movies (will be skipped).")

    print("[STEP 2] Determining Tier B IDs ...")
    if TIER_B_CARDS.exists():
        tier_b_ids = load_ids_from_jsonl(TIER_B_CARDS)
        print(f"         Loaded {len(tier_b_ids)} Tier B IDs from {TIER_B_CARDS}.")
    else:
        # Fallback: derive from genome-scores.csv directly
        print(f"         {TIER_B_CARDS} not found — deriving from genome-scores.csv ...")
        tier_b_ids = load_genome_movie_ids(GENOME_SCORES)
        tier_b_ids -= tier_a_ids   # Tier B excludes Tier A by definition

    already_covered = tier_a_ids | tier_b_ids

    # ------------------------------------------------------------------ #
    # STEP 3: Read movies.csv and identify Tier C candidates               #
    # ------------------------------------------------------------------ #
    print("\n[STEP 3] Reading movies.csv ...")
    tier_c_movies = []
    with open(MOVIES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = int(row["movieId"])
            if mid in already_covered:
                continue
            raw_title = row["title"]
            clean_title, year = extract_year(raw_title)
            raw_genres = row["genres"]
            genres = (
                []
                if raw_genres.strip() == "(no genres listed)"
                else [g.strip() for g in raw_genres.split("|") if g.strip()]
            )
            tier_c_movies.append({
                "movie_id": mid,
                "title": clean_title,
                "year": year,
                "genres": genres,
            })

    print(f"         {len(tier_c_movies)} Tier C candidate movies found.")

    if args.limit:
        print(f"         [SMOKE TEST] Limiting to {args.limit} movies.")
        tier_c_movies = tier_c_movies[: args.limit]

    tier_c_ids = {m["movie_id"] for m in tier_c_movies}

    # ------------------------------------------------------------------ #
    # STEP 4: Load user tags for Tier C movies only                        #
    # ------------------------------------------------------------------ #
    print("\n[STEP 4] Loading user tags ...")
    tag_counts = load_user_tags(TAGS_CSV, tier_c_ids)
    min_users = args.min_tag_users

    # ------------------------------------------------------------------ #
    # STEP 5: Assemble and write records                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[STEP 5] Writing Tier C profile cards (min_tag_users={min_users}) ...")
    written = 0
    has_tags = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for m in tier_c_movies:
            mid = m["movie_id"]
            movie_tags = tag_counts.get(mid, {})

            # Only keep tags applied by >= min_users distinct users
            user_tags = sorted(
                [tag for tag, cnt in movie_tags.items() if cnt >= min_users],
                key=lambda t: -movie_tags[t],  # sort by popularity desc
            )

            record = {
                "movie_id": mid,
                "title": m["title"],
                "year": m["year"],
                "tier": "C",
                "genres": m["genres"],
                "user_tags": user_tags,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if user_tags:
                has_tags += 1

            if written % 5000 == 0:
                print(f"  ... {written} records written")

    # ------------------------------------------------------------------ #
    # SUMMARY                                                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    print("TIER C BUILD COMPLETE")
    print("=" * 55)
    print(f"  Output file            : {OUTPUT_PATH}")
    print(f"  Tier C cards written   : {written}")
    print(f"  With ≥1 user tag       : {has_tags} ({100*has_tags/written:.1f}%)" if written else "")
    print(f"  Without user tags      : {written - has_tags}" if written else "")
    print(f"  Min tag users filter   : {min_users}")

    if written > 0:
        print("\nSample records (first 2):")
        with open(OUTPUT_PATH) as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                r = json.loads(line)
                print(f"  [{i+1}] movie_id={r['movie_id']} | title={r['title']!r} | "
                      f"year={r['year']} | genres={r['genres']} | user_tags={r['user_tags'][:5]}")

    print("=" * 55)
    print("\nNote: Tier C movies have NO dense vector.")
    print("      They are recommendable via ALS collaborative filtering + genre matching only.")
    print("      Semantic queries ('slow-paced existential dread') will not surface these.")


if __name__ == "__main__":
    main()
