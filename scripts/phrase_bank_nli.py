#!/usr/bin/env python3
"""
scripts/phrase_bank_nli.py — Phrase-Bank NLI Enrichment for Tier B & Tier C

Strategy:
  - Extract a curated vocabulary of 200 themes + 100 tones live from Tier A cards.
  - Use 30 curated Moral Complexity and 30 Directorial Style archetypes.
  - For each Tier B / Tier C movie's Wikipedia plot summary, run
    MoritzLaurer/ModernBERT-large-zeroshot-v2.0 on CUDA GPU (zero-shot NLI) to
    assign confidence-scored labels from each category.
  - All output vocabulary matches Tier A exactly — cross-tier alignment guaranteed.

Output files:
  dirtywork/tier_b_nli_cards.jsonl
  dirtywork/tier_c_nli_cards.jsonl
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
TIER_A_CARDS = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_CARDS = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
TIER_B_WIKI  = PROJECT_ROOT / "dirtywork" / "tier_b_wikipedia.jsonl"
TIER_C_WIKI  = PROJECT_ROOT / "dirtywork" / "tier_c_wikipedia.jsonl"
OUT_TIER_B   = PROJECT_ROOT / "dirtywork" / "tier_b_nli_cards.jsonl"
OUT_TIER_C   = PROJECT_ROOT / "dirtywork" / "tier_c_nli_cards.jsonl"

MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"

# NLI confidence thresholds
THEME_THRESHOLD  = 0.40
TONE_THRESHOLD   = 0.40
MORAL_THRESHOLD  = 0.35
STYLE_THRESHOLD  = 0.35
MAX_THEMES       = 5
MAX_TONES        = 4

# Label count caps — controls speed vs coverage tradeoff
# Lower = faster (each label = 1 GPU forward pass)
N_THEMES = 40    # top-40 themes from Tier A (was 200)
N_TONES  = 25   # top-25 tones from Tier A (was 100)

# ---------------------------------------------------------------------------
# Curated archetypes (distilled from Tier A free-form descriptions)
# ---------------------------------------------------------------------------
MORAL_COMPLEXITY_ARCHETYPES = [
    "High moral ambiguity with ethically grey main characters",
    "Clear distinction between righteous heroes and evil villains",
    "Flawed anti-hero driven by revenge or survival",
    "Tragic protagonist forced into impossible ethical choices",
    "Explores corporate or institutional corruption and cover-ups",
    "Sympathetic portrayal of outlaws or criminals",
    "Relatable antagonist with understandable and justifiable motives",
    "Explores vigilante justice versus systemic law and order",
    "Exposes war crimes and the ethical horror of military conflict",
    "Deconstructs the traditional concept of good versus evil",
    "Examines greed, ambition, and the corruption of the human soul",
    "Moral redemption arc for a deeply flawed or guilty character",
    "Selfless sacrifice for the greater good of others",
    "Explores the cycle of violence and its generational toll",
    "Lighthearted morality story with clear positive life lessons",
]

DIRECTORIAL_STYLE_ARCHETYPES = [
    "Atmospheric slow-burn with lingering wide shots and creeping dread",
    "Fast-paced kinetic editing with explosive action choreography",
    "Gritty realistic handheld visual style with naturalistic lighting",
    "Surreal dreamlike visual style with avant-garde symbolism",
    "Claustrophobic framing and intense psychological score",
    "Gothic atmospheric lighting with heavy shadows and eerie imagery",
    "Documentary realism with authentic non-professional improvisational feel",
    "Vibrant bold color palettes with whimsical and playful framing",
    "Expansive panoramic landscape cinematography with epic orchestral score",
    "Minimalist restrained direction with quiet emotional intimacy",
    "Tense claustrophobic staging in a single isolated confined location",
    "Retro nostalgic visual aesthetic paying homage to classic cinema",
    "Psychological horror staging prioritizing dread over cheap jump scares",
    "Brisk slapstick comedic timing with exaggerated visual gags",
    "Poetic visual storytelling with minimal dialogue and ambient score",
]


# ---------------------------------------------------------------------------
# Build live phrase bank from Tier A
# ---------------------------------------------------------------------------
def build_phrase_bank(tier_a_path: Path, n_themes: int = 200, n_tones: int = 100):
    themes_c: Counter = Counter()
    tones_c: Counter = Counter()

    if not tier_a_path.exists():
        raise FileNotFoundError(f"Tier A cards not found: {tier_a_path}")

    with open(tier_a_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                card = json.loads(line)
                for t in card.get("themes", []):
                    clean = t.lower().strip()
                    if len(clean) > 2:
                        themes_c[clean] += 1
                for t in card.get("tone", []):
                    clean = t.lower().strip()
                    if len(clean) > 2:
                        tones_c[clean] += 1
            except Exception:
                continue

    top_themes = [t for t, _ in themes_c.most_common(n_themes)]
    top_tones  = [t for t, _ in tones_c.most_common(n_tones)]
    return top_themes, top_tones

# Total label evaluations per movie = N_THEMES + N_TONES + len(MORAL) + len(STYLE)
# = 40 + 25 + 15 + 15 = 95  (vs 360 before — 3.8x faster)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_wiki_index(path: Path) -> dict:
    index = {}
    if not path.exists():
        return index
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                mid = rec.get("movie_id")
                if mid is not None:
                    index[int(mid)] = rec
            except Exception:
                continue
    return index


def load_completed_ids(path: Path) -> set:
    completed = set()
    if not path.exists():
        return completed
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["movie_id"]))
            except Exception:
                continue
    return completed


def load_base_cards(path: Path) -> dict:
    cards = {}
    if not path.exists():
        return cards
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                cards[int(c["movie_id"])] = c
            except Exception:
                continue
    return cards


# ---------------------------------------------------------------------------
# NLI inference — 4 passes per movie
# ---------------------------------------------------------------------------
def run_nli_for_movie(classifier, text: str, top_themes: list, top_tones: list) -> dict:
    # Pass 1: Themes (multi-label)
    res_t = classifier(text, candidate_labels=top_themes, multi_label=True)
    themes = [
        lbl for lbl, score in zip(res_t["labels"], res_t["scores"])
        if score >= THEME_THRESHOLD
    ][:MAX_THEMES]

    # Pass 2: Tones (multi-label)
    res_to = classifier(text, candidate_labels=top_tones, multi_label=True)
    tones = [
        lbl for lbl, score in zip(res_to["labels"], res_to["scores"])
        if score >= TONE_THRESHOLD
    ][:MAX_TONES]

    # Pass 3: Moral Complexity (single best archetype)
    res_m = classifier(text, candidate_labels=MORAL_COMPLEXITY_ARCHETYPES, multi_label=False)
    moral_complexity = res_m["labels"][0] if res_m["scores"][0] >= MORAL_THRESHOLD else None

    # Pass 4: Directorial Style (single best archetype)
    res_d = classifier(text, candidate_labels=DIRECTORIAL_STYLE_ARCHETYPES, multi_label=False)
    directorial_style = res_d["labels"][0] if res_d["scores"][0] >= STYLE_THRESHOLD else None

    return {
        "themes":                  themes,
        "tone":                    tones,
        "moral_complexity":        moral_complexity,
        "directorial_style_notes": directorial_style,
    }


# ---------------------------------------------------------------------------
# Tier enrichment loop
# ---------------------------------------------------------------------------
def enrich_tier(label, wiki_index, base_cards, output_path, classifier, top_themes, top_tones):
    completed = load_completed_ids(output_path)
    print(f"\n--- {label}: {len(completed):,} already enriched ---")

    queue = []
    for mid, wiki in wiki_index.items():
        if mid in completed:
            continue
        text = wiki.get("plot") or wiki.get("intro") or ""
        if not text.strip():
            continue
        queue.append((mid, wiki.get("title", ""), text[:1500]))

    print(f"Queued: {len(queue):,} movies to enrich for {label}")
    if not queue:
        print(f"All {label} movies already enriched!")
        return

    t_start = time.time()
    n_done = 0
    n_skip = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for mid, title, text in queue:
            try:
                nli_fields = run_nli_for_movie(classifier, text, top_themes, top_tones)
                base = base_cards.get(mid, {"movie_id": mid, "title": title})
                record = {**base, **nli_fields, "nli_enriched": True}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1

                if n_done % 50 == 0:
                    elapsed = time.time() - t_start
                    rate    = n_done / elapsed
                    eta_m   = (len(queue) - n_done) / rate / 60
                    print(
                        f"  [{label}] [{n_done:,}/{len(queue):,}] "
                        f"rate={rate:.2f} movies/sec ({rate*60:.0f}/min)  "
                        f"ETA={eta_m:.1f}m",
                        flush=True,
                    )
            except Exception as e:
                n_skip += 1
                print(f"  [warn] Skipped movie_id={mid} ({title[:40]}): {e}", flush=True)

    elapsed = time.time() - t_start
    print(
        f"\nDONE — {label}: {n_done:,} enriched, {n_skip:,} skipped "
        f"in {elapsed:.0f}s  ({n_done / elapsed * 60:.0f} movies/min)"
    )
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print("Phrase-Bank NLI Enrichment Pipeline")
    print(f"Model : {MODEL_ID}")
    print("=" * 64)

    # 1. Build phrase bank from live Tier A data
    print("\n[Step 1] Building phrase bank from Tier A cards...")
    top_themes, top_tones = build_phrase_bank(TIER_A_CARDS, n_themes=N_THEMES, n_tones=N_TONES)
    print(f"  Themes : {len(top_themes)}  |  Tones: {len(top_tones)}")
    print(f"  Moral archetypes : {len(MORAL_COMPLEXITY_ARCHETYPES)}")
    print(f"  Style archetypes : {len(DIRECTORIAL_STYLE_ARCHETYPES)}")

    # 2. Load NLI pipeline on CUDA GPU
    device = 0 if torch.cuda.is_available() else -1
    print(f"\n[Step 2] Loading NLI pipeline on {'CUDA GPU 0' if device == 0 else 'CPU'}...")
    t0 = time.time()
    classifier = pipeline("zero-shot-classification", model=MODEL_ID, device=device)
    print(f"  Pipeline ready in {time.time() - t0:.1f}s")

    # 3. Load Tier B
    print("\n[Step 3] Loading Tier B Wikipedia plots + base cards...")
    tier_b_wiki  = load_wiki_index(TIER_B_WIKI)
    tier_b_cards = load_base_cards(TIER_B_CARDS)
    print(f"  Wiki plots : {len(tier_b_wiki):,}  |  Base cards: {len(tier_b_cards):,}")

    # 4. Enrich Tier B
    print("\n[Step 4] Enriching Tier B...")
    enrich_tier("Tier B", tier_b_wiki, tier_b_cards, OUT_TIER_B, classifier, top_themes, top_tones)

    # 5. Load Tier C
    print("\n[Step 5] Loading Tier C Wikipedia plots...")
    tier_c_wiki = load_wiki_index(TIER_C_WIKI)
    print(f"  Wiki plots : {len(tier_c_wiki):,}")

    # 6. Enrich Tier C
    print("\n[Step 6] Enriching Tier C...")
    enrich_tier("Tier C", tier_c_wiki, {}, OUT_TIER_C, classifier, top_themes, top_tones)

    print("\n" + "=" * 64)
    print("All Done!")
    print(f"  Tier B -> {OUT_TIER_B}")
    print(f"  Tier C -> {OUT_TIER_C}")
    print("=" * 64)


if __name__ == "__main__":
    main()
