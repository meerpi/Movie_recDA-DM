#!/usr/bin/env python3
"""
backfill_tier_a_genome.py — STEP 0: Add genome_vector + top_tags + title + year to Tier A profile cards.

Every Tier A movie has a genome vector (confirmed: 100% coverage).
This script does two pure JOINs:
  1. movies.csv  → adds title + year (Tier A cards have neither)
  2. genome-scores.csv → adds genome_vector + top_tags

Fields added to each record:
    title         : str            — from data/ml-25m/movies.csv
    year          : int | None     — parsed from ML title string e.g. 'Toy Story (1995)'
    genome_vector : List[float]    — 1128-dim, tagId order 1..1128
    top_tags      : List[{tag: str, relevance: float}]  — relevance >= threshold

Zero API calls. Runtime: ~60-90s (streams 435MB genome-scores.csv once).

USAGE:
    python backfill_tier_a_genome.py
    python backfill_tier_a_genome.py --threshold 0.6   # default: 0.6
    python backfill_tier_a_genome.py --threshold 0.5   # more tags per movie
    python backfill_tier_a_genome.py --threshold 0.7   # stricter, sharper tags
    python backfill_tier_a_genome.py --dry-run         # validate without writing
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TIER_A_V2     = Path("tier_a_profile_cards_v2.jsonl")
TIER_A_V3     = Path("tier_a_profile_cards_v3.jsonl")
GENOME_SCORES = Path("data/ml-25m/genome-scores.csv")
GENOME_TAGS   = Path("data/ml-25m/genome-tags.csv")
MOVIES_CSV    = Path("data/ml-25m/movies.csv")

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

NUM_TAGS = 1128

# Tier A movies have denser, sharper genome distributions than the full corpus
# (more raters per film → scores cluster near 0 or 1, fewer mid-range values).
# 0.6 is a good default: cleaner signal than 0.5 without being overly restrictive.
DEFAULT_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ml_titles(path: Path) -> dict[int, tuple[str, int | None]]:
    """Returns {movie_id: (clean_title, year)} from MovieLens movies.csv."""
    titles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["title"].strip()
            m = _YEAR_RE.search(raw)
            if m:
                year = int(m.group(1))
                clean = raw[: m.start()].strip()
            else:
                year = None
                clean = raw
            titles[int(row["movieId"])] = (clean, year)
    return titles


def load_tag_names(path: Path) -> dict[int, str]:
    """Returns {tag_id (int): tag_name (str)}"""
    names = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names[int(row["tagId"])] = row["tag"]
    return names


def load_tier_a_cards(path: Path) -> dict[int, dict]:
    """Load all Tier A v2 records into memory keyed by movie_id."""
    cards = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
                cards[card["movie_id"]] = card
            except Exception as e:
                print(f"[WARN] Skipping malformed line: {e}")
    return cards


def stream_genome_for_tier_a(
    path: Path,
    tier_a_ids: set[int],
    tag_names: dict[int, str],
    threshold: float,
) -> dict[int, dict]:
    """
    Stream genome-scores.csv once.
    For each Tier A movie encountered, build its genome_vector and top_tags.
    Returns {movie_id: {"genome_vector": [...], "top_tags": [...]}}
    """
    genome_data: dict[int, dict] = {}

    current_id: int | None = None
    current_vec: list[float] = [0.0] * NUM_TAGS

    print(f"[INFO] Streaming {path} ...")
    t0 = time.time()
    rows_read = 0
    tier_a_found = 0

    def flush(movie_id: int, vec: list[float]) -> None:
        nonlocal tier_a_found
        top_tags = [
            {"tag": tag_names[i + 1], "relevance": round(v, 6)}
            for i, v in enumerate(vec)
            if v >= threshold
        ]
        top_tags.sort(key=lambda x: x["relevance"], reverse=True)
        genome_data[movie_id] = {
            "genome_vector": [round(v, 6) for v in vec],
            "top_tags": top_tags,
        }
        tier_a_found += 1

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_read += 1
            mid = int(row["movieId"])
            tid = int(row["tagId"]) - 1  # 0-indexed
            rel = float(row["relevance"])

            if mid != current_id:
                # Flush previous movie if it was a Tier A movie
                if current_id is not None and current_id in tier_a_ids:
                    flush(current_id, current_vec)
                current_id = mid
                current_vec = [0.0] * NUM_TAGS

            current_vec[tid] = rel

            if rows_read % 2_000_000 == 0:
                elapsed = time.time() - t0
                print(f"  ... {rows_read:,} rows scanned, "
                      f"{tier_a_found} Tier A movies collected ({elapsed:.0f}s)")

        # Flush the last movie
        if current_id is not None and current_id in tier_a_ids:
            flush(current_id, current_vec)

    elapsed = time.time() - t0
    print(f"[INFO] Genome scan complete: {rows_read:,} rows in {elapsed:.1f}s")
    print(f"[INFO] Tier A movies with genome data: {tier_a_found}/{len(tier_a_ids)}")

    return genome_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backfill genome_vector + top_tags into Tier A profile cards."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Min genome relevance to include in top_tags (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and report stats without writing output."
    )
    parser.add_argument(
        "--output", type=Path, default=TIER_A_V3,
        help=f"Output path (default: {TIER_A_V3})"
    )
    args = parser.parse_args()

    # Validate inputs
    for p in [TIER_A_V2, GENOME_SCORES, GENOME_TAGS, MOVIES_CSV]:
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    print("=" * 60)
    print("STEP 0 — Backfill Tier A Genome Data")
    print("=" * 60)
    print(f"  Input  : {TIER_A_V2}")
    print(f"  Output : {args.output}")
    print(f"  Threshold: {args.threshold} (tags with relevance >= this are included)")
    if args.dry_run:
        print("  [DRY RUN] No output will be written.")
    print()

    # STEP 1: Load tag names
    print("[STEP 1] Loading genome tag names...")
    tag_names = load_tag_names(GENOME_TAGS)
    print(f"         {len(tag_names)} tags loaded.")

    # STEP 1b: Load ML titles (Tier A cards have no title/year)
    print("[STEP 1b] Loading MovieLens title/year map...")
    ml_titles = load_ml_titles(MOVIES_CSV)
    print(f"          {len(ml_titles)} titles loaded.")

    # STEP 2: Load Tier A cards
    print("[STEP 2] Loading Tier A v2 profile cards...")
    cards = load_tier_a_cards(TIER_A_V2)
    tier_a_ids = set(cards.keys())
    print(f"         {len(cards)} Tier A cards loaded.")

    # STEP 3: Stream genome scores
    print("[STEP 3] Scanning genome-scores.csv for Tier A movies...")
    genome_data = stream_genome_for_tier_a(
        GENOME_SCORES, tier_a_ids, tag_names, args.threshold
    )

    # STEP 4: Validate coverage
    missing = tier_a_ids - set(genome_data.keys())
    if missing:
        print(f"[WARN] {len(missing)} Tier A movies have NO genome data:")
        for mid in sorted(missing)[:10]:
            print(f"  movie_id={mid}  title={cards[mid].get('title','?')}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print(f"[OK] 100% coverage — all {len(tier_a_ids)} Tier A movies have genome vectors.")

    # STEP 5: Stats on top_tags distribution
    tag_counts = [len(genome_data[mid]["top_tags"]) for mid in genome_data]
    avg_tags = sum(tag_counts) / len(tag_counts) if tag_counts else 0
    print(f"\n[STATS] top_tags per movie at threshold={args.threshold}:")
    print(f"  Min : {min(tag_counts)}")
    print(f"  Max : {max(tag_counts)}")
    print(f"  Avg : {avg_tags:.1f}")
    print(f"  Median: {sorted(tag_counts)[len(tag_counts)//2]}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without writing output.")
        return

    # STEP 6: Merge and write v3
    print(f"\n[STEP 4] Writing {args.output} ...")
    t0 = time.time()
    written = 0
    no_genome = 0

    with open(args.output, "w", encoding="utf-8") as out:
        for movie_id, card in cards.items():
            # Join title + year from movies.csv (not present in v2 cards)
            if movie_id in ml_titles:
                title, year = ml_titles[movie_id]
                card["title"] = title
                card["year"]  = year
            else:
                card["title"] = ""
                card["year"]  = None

            if movie_id in genome_data:
                card["genome_vector"] = genome_data[movie_id]["genome_vector"]
                card["top_tags"] = genome_data[movie_id]["top_tags"]
            else:
                # Should not happen given 100% coverage, but handle gracefully
                card["genome_vector"] = None
                card["top_tags"] = []
                no_genome += 1

            out.write(json.dumps(card, ensure_ascii=False) + "\n")
            written += 1

    elapsed = time.time() - t0
    print(f"         {written} records written in {elapsed:.1f}s")

    # STEP 7: Verify output
    print("\n[STEP 5] Verifying output...")
    with open(args.output, encoding="utf-8") as f:
        sample = json.loads(f.readline())

    assert "genome_vector" in sample, "genome_vector missing from output!"
    assert "top_tags" in sample, "top_tags missing from output!"
    assert len(sample["genome_vector"]) == NUM_TAGS, \
        f"genome_vector has {len(sample['genome_vector'])} dims, expected {NUM_TAGS}"

    print(f"\nSample record (first):")
    print(f"  movie_id      : {sample['movie_id']}")
    print(f"  title         : {sample.get('title', 'MISSING')}")
    print(f"  year          : {sample.get('year', 'MISSING')}")
    print(f"  themes        : {sample.get('themes', [])[:3]}")
    print(f"  genome_vector : [{sample['genome_vector'][0]}, ..., "
          f"{sample['genome_vector'][-1]}]  ({len(sample['genome_vector'])} dims)")
    print(f"  top_tags      : {sample['top_tags'][:5]} ... ({len(sample['top_tags'])} total)")
    print(f"  has moral_complexity     : {'moral_complexity' in sample}")
    print(f"  has directorial_style_notes : {'directorial_style_notes' in sample}")
    print(f"  has standout_performances   : {'standout_performances' in sample}")

    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Output          : {args.output}")
    print(f"  Records written : {written}")
    print(f"  With genome     : {written - no_genome}")
    print(f"  Without genome  : {no_genome}  (should be 0)")
    print(f"  Threshold used  : {args.threshold}")
    print(f"  Avg top_tags    : {avg_tags:.1f} per movie")
    print("=" * 60)
    print(f"\nNext step: python nlp/build_bm25.py")


if __name__ == "__main__":
    main()
