#!/usr/bin/env python3
"""
build_tier_b.py — Build Tier B profile cards from the MovieLens 25M genome data.

Tier B covers movies that have genome tag-relevance vectors but were NOT
processed in Tier A (i.e., they lack critic/audience reviews or a plot summary).

Result: 4,290 movies each with:
  - genome_vector  : 1128-dim float list (tagId order 1..1128), suitable for HNSW indexing
  - top_tags       : human-readable list of {tag, relevance} for tags with relevance >= TOP_TAG_THRESHOLD
  - tier            : "B"

USAGE:
    python build_tier_b.py
    python build_tier_b.py --top-tag-threshold 0.5   # default
    python build_tier_b.py --limit 50               # smoke test on 50 movies

RUNTIME: ~60-120s (streams 435MB genome-scores.csv once).
No API calls needed — pure data transformation.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — all relative to project root (run from /home/meerpi/curr_project/movie_rec)
# ---------------------------------------------------------------------------

TIER_A_CARDS    = Path("tier_a_profile_cards_v2.jsonl")
GENOME_SCORES   = Path("data/ml-25m/genome-scores.csv")
GENOME_TAGS     = Path("data/ml-25m/genome-tags.csv")
MOVIES_CSV      = Path("data/ml-25m/movies.csv")
OUTPUT_PATH     = Path("tier_b_profile_cards.jsonl")

# Relevance threshold for top_tags (genome relevance is 0..1).
# 0.5 means "this tag is at least moderately relevant to the movie."
DEFAULT_TOP_TAG_THRESHOLD = 0.5

NUM_TAGS = 1128  # confirmed from genome-tags.csv (tagId 1..1128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tier_a_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        print(f"[WARN] Tier A file not found: {path}")
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


def load_tag_names(path: Path) -> dict:
    """Returns {tag_id (int): tag_name (str)}"""
    names = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[int(row["tagId"])] = row["tag"]
    return names


def load_ml_titles(path: Path) -> dict:
    """Returns {movie_id (int): title (str)}"""
    titles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[int(row["movieId"])] = row["title"]
    return titles


def stream_genome_scores(path: Path, tier_a_ids: set, limit: int | None):
    """
    Stream genome-scores.csv and yield (movie_id, vector, top_tag_pairs) per movie.

    Builds the full 1128-dim vector in memory per movie (only one movie at a time),
    so peak RAM is very low (~4,290 × 1128 floats = ~19 MB total when flushed).
    """
    tag_names = load_tag_names(GENOME_TAGS)

    print(f"[INFO] Streaming {path} ...")
    t0 = time.time()

    current_id = None
    current_vec = [0.0] * NUM_TAGS  # index 0 = tagId 1

    yielded = 0

    def flush(movie_id, vec, threshold):
        top_tags = [
            {"tag": tag_names[i + 1], "relevance": round(v, 6)}
            for i, v in enumerate(vec)
            if v >= threshold
        ]
        top_tags.sort(key=lambda x: x["relevance"], reverse=True)
        return top_tags, [round(v, 6) for v in vec]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movieId"])
            tid = int(row["tagId"]) - 1   # 0-indexed
            rel = float(row["relevance"])

            if mid != current_id:
                # Flush previous movie if valid
                if current_id is not None and current_id not in tier_a_ids:
                    yield current_id, current_vec[:]
                    yielded += 1
                    if limit and yielded >= limit:
                        break
                current_id = mid
                current_vec = [0.0] * NUM_TAGS

            current_vec[tid] = rel

        else:
            # Flush last movie (loop exited normally, not via break)
            if current_id is not None and current_id not in tier_a_ids:
                yield current_id, current_vec[:]

    elapsed = time.time() - t0
    print(f"[INFO] Genome scan done in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Tier B profile cards from genome data.")
    parser.add_argument("--top-tag-threshold", type=float, default=DEFAULT_TOP_TAG_THRESHOLD,
                        help=f"Min relevance to include in top_tags (default: {DEFAULT_TOP_TAG_THRESHOLD})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process N movies (smoke-test mode)")
    args = parser.parse_args()

    # Validate inputs
    for p in [GENOME_SCORES, GENOME_TAGS, MOVIES_CSV]:
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    # Load reference data
    print("[STEP 1] Loading Tier A IDs...")
    tier_a_ids = load_tier_a_ids(TIER_A_CARDS)
    print(f"         Tier A has {len(tier_a_ids)} movies — these will be skipped.")

    print("[STEP 2] Loading MovieLens title map...")
    ml_titles = load_ml_titles(MOVIES_CSV)

    print("[STEP 3] Loading genome tag names...")
    tag_names = load_tag_names(GENOME_TAGS)
    print(f"         {len(tag_names)} tags loaded (max tagId = {max(tag_names)}).")

    threshold = args.top_tag_threshold

    # Stream and write
    print(f"\n[STEP 4] Building Tier B cards (top_tag threshold = {threshold}) ...")
    if args.limit:
        print(f"         [SMOKE TEST] limiting to {args.limit} movies.")

    written = 0
    skipped_tier_a = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for movie_id, vec in stream_genome_scores(GENOME_SCORES, tier_a_ids, args.limit):
            if movie_id in tier_a_ids:
                skipped_tier_a += 1
                continue

            top_tags = [
                {"tag": tag_names[i + 1], "relevance": round(v, 6)}
                for i, v in enumerate(vec)
                if v >= threshold
            ]
            top_tags.sort(key=lambda x: x["relevance"], reverse=True)

            record = {
                "movie_id": movie_id,
                "title": ml_titles.get(movie_id, ""),
                "tier": "B",
                "top_tags": top_tags,
                "genome_vector": [round(v, 6) for v in vec],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 500 == 0:
                print(f"  ... {written} Tier B cards written")

    # Summary
    print("\n" + "=" * 55)
    print("TIER B BUILD COMPLETE")
    print("=" * 55)
    print(f"  Output file      : {OUTPUT_PATH}")
    print(f"  Tier B cards     : {written}")
    print(f"  Skipped (Tier A) : {skipped_tier_a}")
    print(f"  Top-tag threshold: {threshold}")

    if written > 0:
        # Quick eyeball check
        print("\nSample record (first):")
        with open(OUTPUT_PATH) as f:
            sample = json.loads(f.readline())
        print(f"  movie_id    : {sample['movie_id']}")
        print(f"  title       : {sample['title']}")
        print(f"  top_tags    : {sample['top_tags'][:5]} ... ({len(sample['top_tags'])} total)")
        print(f"  vector dims : {len(sample['genome_vector'])} (should be {NUM_TAGS})")

    print("=" * 55)


if __name__ == "__main__":
    main()
