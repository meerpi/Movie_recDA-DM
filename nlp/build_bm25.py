#!/usr/bin/env python3
"""
build_bm25.py — STEP 1: Build BM25 index over Tier A profile cards.

Indexes all 9,526 Tier A movies from tier_a_profile_cards_v3.jsonl.
Each movie becomes a weighted text document from its structured fields.

Output:
    nlp/bm25_index.pkl      — serialized BM25Okapi index + corpus
    nlp/bm25_id_map.json    — {list_position: movie_id} for result lookup

USAGE:
    python nlp/build_bm25.py
    python nlp/build_bm25.py --input tier_a_profile_cards_v3.jsonl
    python nlp/build_bm25.py --dry-run     # print sample docs, no index built
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to project root — run from movie_rec/)
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path("tier_a_profile_cards_v3.jsonl")
INDEX_OUT     = Path("nlp/bm25_index.pkl")
ID_MAP_OUT    = Path("nlp/bm25_id_map.json")

# ---------------------------------------------------------------------------
# Field weights
# How many times to repeat a field's tokens to simulate boosting in BM25Okapi.
# BM25Okapi has no native field weighting, so repetition is the standard trick.
# ---------------------------------------------------------------------------
FIELD_WEIGHTS = {
    "themes":               2,    # strongest semantic signal
    "tone":                 2,    # second strongest
    "comparable_films":     2,    # "movies like X" queries are common
    "top_tags":             2,    # crowdsourced behavioral tags — high signal
    "standout_performances":1,    # actor/performance queries
    "directorial_style_notes": 1,
    "notable_criticisms":   1,
    "moral_complexity":     1,
    "pacing":               1,
    "title":                3,    # exact title match should dominate
}


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop empty tokens."""
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


# ---------------------------------------------------------------------------
# Document serializer
# ---------------------------------------------------------------------------

def card_to_tokens(card: dict) -> list[str]:
    """
    Convert a Tier A profile card to a flat token list with field weighting.
    Field repetition = boosting: themes appear 2x → BM25 weights them higher.
    """
    tokens: list[str] = []

    def add(text: str, weight: int) -> None:
        toks = tokenize(text)
        tokens.extend(toks * weight)

    def add_list(items: list, weight: int) -> None:
        for item in items:
            if isinstance(item, str):
                add(item, weight)
            elif isinstance(item, dict) and "tag" in item:
                add(item["tag"], weight)

    # Title + year (title gets highest weight — exact title queries must hit)
    title_str = f"{card.get('title', '')} {card.get('year', '')}"
    add(title_str, FIELD_WEIGHTS["title"])

    # List fields
    add_list(card.get("themes", []),              FIELD_WEIGHTS["themes"])
    add_list(card.get("tone", []),                FIELD_WEIGHTS["tone"])
    add_list(card.get("comparable_films", []),    FIELD_WEIGHTS["comparable_films"])
    add_list(card.get("top_tags", []),            FIELD_WEIGHTS["top_tags"])
    add_list(card.get("standout_performances", []), FIELD_WEIGHTS["standout_performances"])
    add_list(card.get("notable_criticisms", []),  FIELD_WEIGHTS["notable_criticisms"])

    # Free-text fields
    add(card.get("directorial_style_notes", ""), FIELD_WEIGHTS["directorial_style_notes"])
    add(card.get("moral_complexity", ""),         FIELD_WEIGHTS["moral_complexity"])
    add(card.get("pacing", ""),                   FIELD_WEIGHTS["pacing"])

    return tokens


def card_to_preview(card: dict) -> str:
    """Human-readable serialization for --dry-run inspection."""
    themes   = ", ".join(card.get("themes", []))
    tone     = ", ".join(card.get("tone", []))
    tags     = ", ".join(t["tag"] for t in card.get("top_tags", [])[:8])
    comps    = ", ".join(card.get("comparable_films", [])[:4])
    style    = card.get("directorial_style_notes", "")[:80]
    perfs    = ", ".join(card.get("standout_performances", []))
    return (
        f"  TITLE      : {card.get('title')} ({card.get('year', '?')})\n"
        f"  themes     : {themes}\n"
        f"  tone       : {tone}\n"
        f"  top_tags   : {tags}\n"
        f"  comparable : {comps}\n"
        f"  style      : {style}...\n"
        f"  performers : {perfs}\n"
        f"  pacing     : {card.get('pacing', '')[:60]}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build BM25 index from Tier A profile cards (v3)."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Path to Tier A v3 JSONL (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sample documents and token counts, do not build index."
    )
    parser.add_argument(
        "--sample", type=int, default=5,
        help="Number of sample docs to print in --dry-run (default: 5)"
    )
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(
            f"[ERROR] Input not found: {args.input}\n"
            f"        Run backfill_tier_a_genome.py first to generate tier_a_profile_cards_v3.jsonl"
        )

    print("=" * 60)
    print("STEP 1 — Build BM25 Index (Lane 1 of retrieval)")
    print("=" * 60)
    print(f"  Input  : {args.input}")
    print(f"  Output : {INDEX_OUT} + {ID_MAP_OUT}")

    # ── Load cards ───────────────────────────────────────────────────────────
    print("\n[STEP 1] Loading Tier A profile cards...")
    t0 = time.time()
    cards: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cards.append(json.loads(line))
                except Exception as e:
                    print(f"[WARN] Skipping malformed line: {e}")

    print(f"         {len(cards)} cards loaded in {time.time()-t0:.1f}s")

    # Check top_tags presence (v3 has them; v2 doesn't)
    has_top_tags = sum(1 for c in cards if c.get("top_tags"))
    print(f"         Cards with top_tags : {has_top_tags}/{len(cards)}")
    if has_top_tags == 0:
        print("[WARN] No top_tags found — did you run backfill_tier_a_genome.py first?")
        print("       BM25 will still work but won't include genome tag signal.")

    # ── Tokenize ─────────────────────────────────────────────────────────────
    print("\n[STEP 2] Tokenizing cards...")
    t0 = time.time()
    corpus: list[list[str]] = []
    id_map: list[int] = []   # position → movie_id

    for card in cards:
        tokens = card_to_tokens(card)
        corpus.append(tokens)
        id_map.append(card["movie_id"])

    token_lengths = [len(t) for t in corpus]
    print(f"         Tokenized {len(corpus)} documents in {time.time()-t0:.1f}s")
    print(f"         Avg tokens/doc : {sum(token_lengths)/len(token_lengths):.0f}")
    print(f"         Min tokens/doc : {min(token_lengths)}")
    print(f"         Max tokens/doc : {max(token_lengths)}")

    # ── Dry run mode ─────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[DRY RUN] Sample of {args.sample} documents:\n")
        for card, toks in zip(cards[:args.sample], corpus[:args.sample]):
            print(card_to_preview(card))
            print(f"  token_count: {len(toks)}")
            print(f"  sample_tokens: {toks[:20]}")
            print()
        print("[DRY RUN] Exiting without building index.")
        return

    # ── Build BM25 index ─────────────────────────────────────────────────────
    print("\n[STEP 3] Building BM25Okapi index...")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit(
            "[ERROR] rank_bm25 not installed.\n"
            "        Run: pip install rank-bm25"
        )

    t0 = time.time()
    bm25 = BM25Okapi(corpus)
    elapsed = time.time() - t0
    print(f"         Index built in {elapsed:.1f}s")
    print(f"         Vocabulary size: {len(bm25.idf):,} unique terms")

    # ── Serialize ────────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Saving index to {INDEX_OUT} ...")
    INDEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    payload = {
        "bm25":   bm25,
        "corpus": corpus,   # needed for BM25Okapi.get_top_n()
        "id_map": id_map,
    }
    with open(INDEX_OUT, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(ID_MAP_OUT, "w") as f:
        json.dump(id_map, f)

    size_mb = INDEX_OUT.stat().st_size / 1_048_576
    print(f"         Saved in {time.time()-t0:.1f}s  ({size_mb:.1f} MB)")

    # ── Smoke test ───────────────────────────────────────────────────────────
    print("\n[STEP 5] Smoke test queries...")
    test_queries = [
        "atmospheric slow burn",
        "Stanley Kubrick",
        "dark nihilistic",
        "feel good comedy",
        "based on true story war",
    ]

    # Build reverse id_map for display
    id_to_card = {c["movie_id"]: c for c in cards}

    for query in test_queries:
        q_tokens = tokenize(query)
        scores = bm25.get_scores(q_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        print(f"\n  Query: '{query}'")
        for rank, idx in enumerate(top_indices):
            mid = id_map[idx]
            card = id_to_card[mid]
            title = card.get("title", "?")
            year  = card.get("year", "?")
            print(f"    #{rank+1}  {title} ({year})  [score={scores[idx]:.3f}]")

    # Summary
    print("\n" + "=" * 60)
    print("BM25 INDEX COMPLETE")
    print("=" * 60)
    print(f"  Index file   : {INDEX_OUT}  ({size_mb:.1f} MB)")
    print(f"  ID map       : {ID_MAP_OUT}")
    print(f"  Documents    : {len(corpus)}")
    print(f"  Vocabulary   : {len(bm25.idf):,} terms")
    print(f"  Field weights: title×{FIELD_WEIGHTS['title']}, "
          f"themes×{FIELD_WEIGHTS['themes']}, "
          f"tone×{FIELD_WEIGHTS['tone']}, "
          f"top_tags×{FIELD_WEIGHTS['top_tags']}")
    print("=" * 60)
    print("\nNext step: python nlp/build_genome_hnsw.py")


if __name__ == "__main__":
    main()
