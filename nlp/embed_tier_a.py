#!/usr/bin/env python3
"""
nlp/embed_tier_a.py — Step 3: Generate Dense Voyage AI Embeddings (local run)
==============================================================================

Calls the Voyage AI API to embed all Tier A profile cards into 1024-dim float32
vectors, then saves two files ready for Step 4 (build_dense_hnsw.py):

  nlp/tier_a_voyage_1024d.npy    — embedding matrix, shape [N, 1024]
  nlp/tier_a_voyage_id_map.json  — [movie_id, ...] in row order

USAGE:
    export VOYAGE_API_KEY="pa-..."
    cd /home/meerpi/curr_project/movie_rec
    .venv/bin/python nlp/embed_tier_a.py

    # Resume a partial run (safe to Ctrl+C and rerun):
    .venv/bin/python nlp/embed_tier_a.py

FREE TIER: voyage-3-large has 200M free tokens per account.
This job uses ~1.8M tokens (0.9% of free tier). Cost: $0.00
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import voyageai

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).parent.parent
INPUT_PATH     = PROJECT_ROOT / "tier_a_profile_cards_v3.jsonl"
OUTPUT_NPY     = PROJECT_ROOT / "nlp" / "tier_a_voyage_1024d.npy"
OUTPUT_ID_MAP  = PROJECT_ROOT / "nlp" / "tier_a_voyage_id_map.json"
CHECKPOINT_DIR = PROJECT_ROOT / "nlp" / "embed_checkpoints"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL      = "voyage-4-large"   # 200M free tokens, 1024-dim output
OUTPUT_DIM = 1024

# Standard rate limits (payment method added): use full batch size.
# Free-tier fallback: set BATCH_SIZE=30, REQUEST_DELAY_S=21
BATCH_SIZE        = 128          # Voyage recommended batch size
REQUEST_DELAY_S   = 0           # 0 = no artificial delay (standard limits)
INPUT_TYPE        = "document"  # "document" for content to be indexed


# ---------------------------------------------------------------------------
# Card serializer — same format as kaggle_embed_notebook.py
# ---------------------------------------------------------------------------
def serialize_card(card: dict) -> str:
    top_tags = card.get("top_tags", [])
    tag_str  = ", ".join(t["tag"] for t in top_tags[:20]) if top_tags else ""

    themes = ", ".join(card.get("themes", []))
    tone   = ", ".join(card.get("tone", []))
    pacing = card.get("pacing", "").strip()
    style  = card.get("directorial_style_notes", "").strip()
    comps  = ", ".join(card.get("comparable_films", []))
    crits  = "; ".join(card.get("notable_criticisms", []))
    moral  = card.get("moral_complexity", "").strip()
    perfs  = ", ".join(card.get("standout_performances", []))

    parts = [f"{card.get('title', '?')} ({card.get('year', '?')})."]
    if themes:  parts.append(f"Themes: {themes}.")
    if tone:    parts.append(f"Tone: {tone}.")
    if pacing:  parts.append(f"Pacing: {pacing}.")
    if style:   parts.append(f"Directorial style: {style}")
    if moral:   parts.append(f"Moral complexity: {moral}")
    if perfs:   parts.append(f"Standout performances: {perfs}.")
    if crits:   parts.append(f"Criticisms: {crits}.")
    if tag_str: parts.append(f"Tags: {tag_str}.")
    if comps:   parts.append(f"Similar to: {comps}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("CineVault — Embed Tier A Profile Cards (Step 3)")
    print("=" * 60)

    # ── API key ──────────────────────────────────────────────────
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "\n[ERROR] VOYAGE_API_KEY environment variable not set.\n"
            "  Get your free key at: https://dash.voyageai.com\n"
            "  Then run:  export VOYAGE_API_KEY='pa-...'\n"
        )
    vo = voyageai.Client(api_key=api_key)
    print(f"Model: {MODEL}  |  Output dim: {OUTPUT_DIM}  |  Batch size: {BATCH_SIZE}")

    # ── Load cards ───────────────────────────────────────────────
    if not INPUT_PATH.exists():
        sys.exit(f"[ERROR] Input file not found: {INPUT_PATH}")

    print(f"\nLoading cards from {INPUT_PATH.name} ...")
    movie_ids: list[int] = []
    texts:     list[str] = []

    with open(INPUT_PATH, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] line {lineno}: {e} — skipping", file=sys.stderr)
                continue
            mid = card.get("movie_id")
            if mid is None:
                continue
            movie_ids.append(int(mid))
            texts.append(serialize_card(card))

    N = len(movie_ids)
    approx_tok = sum(len(t) for t in texts) / 4
    print(f"  {N:,} cards loaded  (~{approx_tok:,.0f} tokens, "
          f"{approx_tok/200_000_000*100:.2f}% of 200M free tier)")

    # ── Resume support: load checkpoint if it exists ─────────────
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = np.zeros((N, OUTPUT_DIM), dtype=np.float32)
    done_batches: set[int] = set()

    existing_checkpoints = sorted(CHECKPOINT_DIR.glob("batch_*.npy"))
    if existing_checkpoints:
        print(f"\n  Found {len(existing_checkpoints)} checkpoint(s) — resuming ...")
        for ckpt in existing_checkpoints:
            batch_idx = int(ckpt.stem.split("_")[1])
            lo = batch_idx * BATCH_SIZE
            hi = min(lo + BATCH_SIZE, N)
            embeddings[lo:hi] = np.load(str(ckpt))
            done_batches.add(batch_idx)
        print(f"  Resumed {len(done_batches)} batches ({len(done_batches)*BATCH_SIZE:,} texts)")

    # ── Embed in batches ──────────────────────────────────────────
    num_batches  = math.ceil(N / BATCH_SIZE)
    todo_batches = [i for i in range(num_batches) if i not in done_batches]

    if not todo_batches:
        print("  All batches already embedded (from checkpoints).")
    else:
        print(f"\n  Embedding {len(todo_batches)} remaining batch(es) "
              f"({num_batches} total) via Voyage API ...")
        t_start     = time.time()
        tokens_used = 0

        for step, batch_idx in enumerate(todo_batches, 1):
            lo          = batch_idx * BATCH_SIZE
            hi          = min(lo + BATCH_SIZE, N)
            batch_texts = texts[lo:hi]

            # Throttle to respect rate limits before every request
            if REQUEST_DELAY_S > 0 and step > 1:
                time.sleep(REQUEST_DELAY_S)

            # Retry with exponential backoff
            for attempt in range(8):
                try:
                    result = vo.embed(batch_texts, model=MODEL, input_type=INPUT_TYPE)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "payment method" in err_str or "rate limit" in err_str.lower() or "429" in err_str:
                        wait = max(REQUEST_DELAY_S, 20 * (attempt + 1))
                        print(f"  [RATE-LIMIT] batch {batch_idx} attempt {attempt+1}: "
                              f"sleeping {wait}s before retry...", file=sys.stderr)
                    else:
                        wait = 2 ** attempt
                        print(f"  [WARN] batch {batch_idx} attempt {attempt+1}: {e}. "
                              f"Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Batch {batch_idx} failed after 8 attempts.")

            # Store in matrix
            batch_vecs = np.array(result.embeddings, dtype=np.float32)
            embeddings[lo:hi] = batch_vecs

            # Save checkpoint
            ckpt_path = CHECKPOINT_DIR / f"batch_{batch_idx:05d}.npy"
            np.save(str(ckpt_path), batch_vecs)

            tokens_used += result.total_tokens
            elapsed      = time.time() - t_start
            rate         = step / elapsed if elapsed > 0 else 1
            remaining    = (len(todo_batches) - step) / rate

            print(f"  [{step:4d}/{len(todo_batches)}]  "
                  f"texts {lo:,}-{hi-1:,}  "
                  f"| {elapsed:5.0f}s elapsed  "
                  f"| ~{remaining/60:.1f} min left  "
                  f"| {tokens_used:,} tokens used")

        print(f"\n  Done. Total tokens used this run: {tokens_used:,}")

    # ── Save final outputs ────────────────────────────────────────
    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_NPY), embeddings)
    npy_mb = OUTPUT_NPY.stat().st_size / 1_048_576
    print(f"\n  Saved matrix  → {OUTPUT_NPY}  ({npy_mb:.1f} MB, shape {embeddings.shape})")

    with open(OUTPUT_ID_MAP, "w", encoding="utf-8") as f:
        json.dump(movie_ids, f)
    print(f"  Saved id map  → {OUTPUT_ID_MAP}  ({N} entries)")

    # ── Sanity check ──────────────────────────────────────────────
    print("\n  Sanity check:")
    for i in [0, N // 2, N - 1]:
        norm = float(np.linalg.norm(embeddings[i]))
        print(f"    row {i:5d}  movie_id={movie_ids[i]:6d}  norm={norm:.4f}")

    v0  = embeddings[0]  / np.linalg.norm(embeddings[0])
    v1  = embeddings[1]  / np.linalg.norm(embeddings[1])
    sim = float(np.dot(v0, v1))
    print(f"    Cosine sim between row 0 and row 1: {sim:.4f}")

    # Clean up checkpoints now that final file is written
    for ckpt in CHECKPOINT_DIR.glob("batch_*.npy"):
        ckpt.unlink()
    CHECKPOINT_DIR.rmdir()
    print("  Checkpoints cleaned up.")

    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"  nlp/tier_a_voyage_1024d.npy     ({npy_mb:.1f} MB)")
    print(f"  nlp/tier_a_voyage_id_map.json   ({N} entries)")
    print(f"\n  Next: run  nlp/build_dense_hnsw.py  (Step 4)")


if __name__ == "__main__":
    main()
