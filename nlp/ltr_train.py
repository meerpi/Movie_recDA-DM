"""
nlp/ltr_train.py — Phase 3 LTR ranker training script.

WHAT THIS DOES
--------------
1. Loads all movie metadata into a global in-process cache from cinevault.db + JSONL cards.
2. Monkey-patches eval.replay._fetch_movie_card to use that cache across all
   replay_profile() calls (avoids redundant per-user DB fetches).
3. Samples N_TRAIN_USERS from ML-25M ratings already in cinevault.db,
   chronological-splits each user (last N_TEST = held out), replays the profile
   at the training cutoff, and extracts 12 features per (user, training_movie) pair.
4. Trains an XGBRanker(objective='rank:ndcg') grouped by user_id.
5. Saves the model to model/ltr_model.ubj.
6. Evaluates closed-pool NDCG@10 on N_EVAL_USERS held-out users with BOTH
   the linear baseline (score_candidates) and the LTR model (score_candidates_ltr).

DESIGN DECISIONS
----------------
- rerank_score is NOT a training feature. Training corpus is rated movies (not
  retrieved candidates), so no cross-encoder score is available at training time.
  At inference, score_candidates_ltr() blends the LTR output with norm_rr using
  the same lambda=0.7 formula as score_candidates() — the LTR model replaces norm_prof
  (the personalization term) only.
- synth_* user IDs are excluded from both training and eval pools.
- Evaluation uses a closed candidate pool (all rated movies, train + test).
  Recall@100 is N/A; only NDCG@10 is reported.

Run from project root:
    .venv/bin/python nlp/ltr_train.py
"""

import json
import math
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── constants ────────────────────────────────────────────────────────────────

DB_PATH     = PROJECT_ROOT / "db" / "cinevault.db"
TIER_A_PATH = PROJECT_ROOT / "dirtywork" / "tier_a_profile_cards_v3.jsonl"
TIER_B_PATH = PROJECT_ROOT / "dirtywork" / "tier_b_profile_cards.jsonl"
TIER_C_PATH = PROJECT_ROOT / "dirtywork" / "tier_c_profile_cards.jsonl"
MODEL_DIR   = PROJECT_ROOT / "model"
MODEL_PATH  = MODEL_DIR / "ltr_model.ubj"

RANDOM_SEED   = 42
N_TRAIN_USERS = 4_000
N_EVAL_USERS  = 500
MIN_RATINGS   = 40
N_TEST        = 5

FEATURE_NAMES = [
    "bayesian_avg", "log_volume", "norm_quality", "pop_rank_log",
    "genre_score", "actor_score", "director_score", "tag_score",
    "era_score", "content_rating_score", "watch_history_size_log",
    "profile_composite",
]

# ── movie data cache ─────────────────────────────────────────────────────────

_MOVIE_DATA: Dict[int, dict] = {}


def _parse_list_field(raw_val: Optional[str]) -> List[str]:
    if not raw_val:
        return []
    try:
        parsed = json.loads(raw_val)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [s.strip() for s in raw_val.split(",") if s.strip()]


def build_movie_data_cache(db_path: Path = DB_PATH) -> None:
    global _MOVIE_DATA
    t0 = time.time()
    print("Building movie data cache...", flush=True)

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    rows = c.execute("""
        SELECT m.movie_id, m.title, m.year, m.content_rating, m.actors, m.directors,
               ms.avg_rating, ms.num_ratings, ms.popularity_rank
        FROM movies m
        LEFT JOIN movie_stats ms ON m.movie_id = ms.movie_id
    """).fetchall()

    for row in rows:
        mid = int(row[0])
        _MOVIE_DATA[mid] = {
            "movie_id":          mid,
            "title":             row[1],
            "year":              row[2],
            "content_rating":    row[3] or "",
            "actors":            _parse_list_field(row[4]),
            "directors":         _parse_list_field(row[5]),
            "original_language": "",
            "genres":            [],
            "top_tags":          [],
            "avg_rating":        float(row[6]) if row[6] is not None else None,
            "num_ratings":       int(row[7] or 0),
            "popularity_rank":   int(row[8] or 999_999),
        }

    genre_rows = c.execute("""
        SELECT mg.movie_id, g.name FROM movie_genres mg
        JOIN genres g ON mg.genre_id = g.genre_id
    """).fetchall()
    for (mid, gname) in genre_rows:
        mid = int(mid)
        if mid in _MOVIE_DATA:
            _MOVIE_DATA[mid]["genres"].append(gname)

    conn.close()
    print(f"  {len(_MOVIE_DATA):,} movies loaded from DB.", flush=True)

    n_tagged = 0
    for tier_path in (TIER_A_PATH, TIER_B_PATH, TIER_C_PATH):
        if not tier_path.exists():
            print(f"  [skip] {tier_path.name} not found", flush=True)
            continue
        with open(tier_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    card = json.loads(line)
                    mid = card.get("movie_id")
                    if mid is None:
                        continue
                    mid = int(mid)
                    tags = card.get("top_tags") or []
                    if mid in _MOVIE_DATA and tags:
                        _MOVIE_DATA[mid]["top_tags"] = tags
                        n_tagged += 1
                except Exception:
                    continue
        print(f"  Tags loaded from {tier_path.name}.", flush=True)

    print(f"  Cache ready: {len(_MOVIE_DATA):,} movies, {n_tagged:,} with tags.  ({time.time()-t0:.1f}s)", flush=True)


# ── monkey-patch eval.replay._fetch_movie_card ───────────────────────────────

import eval.replay as _replay_module


def _cached_fetch_movie_card(conn, movie_id):
    mid = int(movie_id)
    if mid in _MOVIE_DATA:
        return _MOVIE_DATA[mid]
    card = _ORIGINAL_fetch_movie_card(conn, mid)
    _MOVIE_DATA[mid] = card
    return card


_ORIGINAL_fetch_movie_card = _replay_module._fetch_movie_card
_replay_module._fetch_movie_card = _cached_fetch_movie_card

from eval.replay import replay_profile
from nlp.scorer import score_candidates
from nlp.ltr_scorer import score_candidates_ltr, load_ltr_model

# ── feature extraction ───────────────────────────────────────────────────────


def extract_features(profile, movie_data: dict) -> List[float]:
    num_ratings = float(movie_data.get("num_ratings") or 0)
    avg_rating  = movie_data.get("avg_rating")
    pop_rank    = float(movie_data.get("popularity_rank") or 999_999)

    raw_avg      = float(avg_rating) if avg_rating is not None else 3.0
    bayesian_avg = (num_ratings * raw_avg + 100.0 * 3.2) / (num_ratings + 100.0)
    log_volume   = min(1.0, math.log10(num_ratings + 1.0) / 4.5)
    norm_quality = (
        0.70 * max(0.0, min(1.0, (bayesian_avg - 1.0) / 4.0))
        + 0.30 * log_volume
    )
    pop_rank_log = math.log10(pop_rank + 1.0)

    genre_score = sum(
        profile.genre_affinity.get(g, 0.0) * profile.genre_weight
        for g in (movie_data.get("genres") or [])
    )

    actor_score = 0.0
    for actor in (movie_data.get("actors") or []):
        if actor in profile.actor_affinity:
            conf = 1.0 if profile.actor_confidence.get(actor, 0) >= 3 else 0.5
            actor_score += profile.actor_affinity[actor] * profile.actor_weight * conf

    director_score = 0.0
    for d in (movie_data.get("directors") or []):
        if d in profile.director_affinity:
            conf = 1.0 if profile.director_confidence.get(d, 0) >= 3 else 0.5
            director_score += profile.director_affinity[d] * profile.director_weight * conf

    tag_score = 0.0
    for tag_item in (movie_data.get("top_tags") or []):
        tag = tag_item if isinstance(tag_item, str) else (
            tag_item.get("tag", "") if isinstance(tag_item, dict) else ""
        )
        if tag:
            tag_score += profile.tag_affinity.get(tag, 0.0) * profile.tag_weight

    era_score = 0.0
    year = movie_data.get("year")
    if year and profile.era_affinity:
        try:
            era = profile.get_era_from_year(int(year))
            era_score = profile.era_affinity.get(era, 0.0) * profile.era_weight
        except (ValueError, TypeError):
            pass

    cr_score = 0.0
    cr = movie_data.get("content_rating") or ""
    if cr and profile.content_rating_affinity:
        cr_score = profile.content_rating_affinity.get(cr, 0.0) * 0.3

    watch_size_log = math.log10(len(profile.watch_history) + 1.0)
    composite = genre_score + actor_score + director_score + tag_score + era_score + cr_score

    return [
        bayesian_avg, log_volume, norm_quality, pop_rank_log,
        genre_score, actor_score, director_score, tag_score,
        era_score, cr_score, watch_size_log, composite,
    ]


# ── user sampling ─────────────────────────────────────────────────────────────


def sample_eligible_users(db_path, n_train, n_eval, min_ratings, seed):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT user_id, COUNT(*) as cnt FROM ratings
        WHERE user_id NOT LIKE 'synth_%'
        GROUP BY user_id HAVING cnt >= ? ORDER BY user_id
    """, (min_ratings,)).fetchall()
    conn.close()

    eligible = [str(r[0]) for r in rows]
    rng = random.Random(seed)
    rng.shuffle(eligible)

    total_needed = n_train + n_eval
    if len(eligible) < total_needed:
        print(f"[warn] Only {len(eligible)} eligible users (need {total_needed}).", flush=True)
        n_train = min(n_train, len(eligible))
        n_eval  = min(n_eval, len(eligible) - n_train)

    train_users = eligible[:n_train]
    eval_users  = eligible[n_train:n_train + n_eval]
    print(f"Sampled {len(train_users):,} train / {len(eval_users):,} eval users from {len(eligible):,} eligible.", flush=True)
    return train_users, eval_users


# ── training data builder ─────────────────────────────────────────────────────


def build_training_data(user_ids, db_path, n_test=N_TEST):
    """
    Builds feature matrix using a chronological half-split to avoid label leakage.

    WHY THE HALF-SPLIT
    ------------------
    In the naive approach (profile built from ALL training movies, features extracted
    for those same movies), the profile already "knows" every training movie — movies
    the user liked contributed their genres/actors to genre_affinity/actor_affinity,
    so those exact movies score artificially high on profile features.  The XGBRanker
    then learns: "items that are in the profile score high" rather than "items that
    match the profile score high".  At inference this inverts: test/unseen items are
    not in the profile, so they score low — NDCG collapses.

    The fix: for each user's n_train events sorted by time:
      - First half  (events 0..mid)   → replay_profile() at mid-point cutoff
      - Second half (events mid..n_train) → extract features using that profile
    Features for the second half targets are computed from a profile that does NOT
    contain those movies.  The model then learns genuine profile-movie affinity, not
    profile-membership.

    Users with fewer than 2*MIN_PROFILE_HALF events in training are skipped.
    """
    MIN_PROFILE_HALF = 15  # profile half must have at least this many events

    X_rows, y_rows, groups = [], [], []
    n_done = n_skip = 0
    t0 = time.time()

    for uid in user_ids:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT movie_id, rating, rated_at FROM ratings WHERE user_id=? ORDER BY rated_at ASC",
            (uid,),
        ).fetchall()
        conn.close()

        if len(rows) <= n_test:
            n_skip += 1
            continue

        train_rows = rows[:len(rows) - n_test]

        # Chronological half-split to avoid leakage
        mid = len(train_rows) // 2
        if mid < MIN_PROFILE_HALF:
            n_skip += 1
            continue

        profile_rows = train_rows[:mid]           # builds the profile
        target_rows  = train_rows[mid:]            # features extracted here

        # Profile at mid-point: does NOT contain any of the target movies
        profile_cutoff_ts = int(profile_rows[-1][2])
        try:
            profile = replay_profile(uid, cutoff_ts=profile_cutoff_ts, db_path=db_path)
        except Exception as e:
            print(f"  [skip] {uid}: {e}", flush=True)
            n_skip += 1
            continue

        group_count = 0
        for (movie_id, rating, _) in target_rows:
            mid_id = int(movie_id)
            md     = _MOVIE_DATA.get(mid_id)
            if md is None:
                continue
            label = max(1, min(10, int(round(float(rating) * 2))))
            X_rows.append(extract_features(profile, md))
            y_rows.append(label)
            group_count += 1

        if group_count > 0:
            groups.append(group_count)
        n_done += 1

        if n_done % 500 == 0:
            print(f"  [{n_done}/{len(user_ids)}] pairs: {len(X_rows):,}  elapsed: {time.time()-t0:.0f}s", flush=True)

    print(f"Training data: {len(X_rows):,} pairs from {n_done:,} users ({n_skip} skipped).  ({time.time()-t0:.1f}s)", flush=True)
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32), np.array(groups, dtype=np.int32)


# ── closed-pool evaluation ────────────────────────────────────────────────────


def ndcg_at_k(relevances, scores, k=10):
    from sklearn.metrics import ndcg_score as sk_ndcg
    if sum(relevances) == 0:
        return 0.0
    y_true  = np.array(relevances, dtype=np.float32).reshape(1, -1)
    y_score = np.array(scores, dtype=np.float32).reshape(1, -1)
    try:
        return float(sk_ndcg(y_true, y_score, k=k))
    except Exception:
        return 0.0


def evaluate_closed_pool(user_ids, db_path, ltr_model, n_test=N_TEST, min_star=4.0):
    baseline_ndcgs, ltr_ndcgs = [], []
    n_skip = 0

    for uid in user_ids:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT movie_id, rating, rated_at FROM ratings WHERE user_id=? ORDER BY rated_at ASC",
            (uid,),
        ).fetchall()
        conn.close()

        if len(rows) <= n_test:
            n_skip += 1
            continue

        train_rows = rows[:len(rows) - n_test]
        test_rows  = rows[len(rows) - n_test:]
        cutoff_ts  = int(train_rows[-1][2])

        try:
            profile = replay_profile(uid, cutoff_ts=cutoff_ts, db_path=db_path)
        except Exception:
            n_skip += 1
            continue

        all_mids = [int(r[0]) for r in rows]
        test_ratings = {int(r[0]): float(r[1]) for r in test_rows}

        candidates = [
            {**(_MOVIE_DATA.get(mid, {"movie_id": mid})), "rerank_score": 0.0, "rrf_score": 0.0}
            for mid in all_mids
        ]
        relevances = [
            1.0 if (mid in test_ratings and test_ratings[mid] >= min_star) else 0.0
            for mid in all_mids
        ]
        if sum(relevances) == 0:
            n_skip += 1
            continue

        # Temporarily clear watch_history so calculate_profile_boost() does not
        # return -10.0 for training items (watched).  Without this, the baseline
        # artificially suppresses ALL training movies (norm_prof ≈ 0.007) while
        # test movies get real profile boosts — inflating baseline NDCG.  The LTR
        # scorer bypasses the veto in feature extraction, so some 5-star training
        # movies outrank test items → NDCG collapses.  Clearing watch_history puts
        # both scorers on equal footing.
        saved_watch = profile.watch_history
        profile.watch_history = set()

        b_scored  = score_candidates(profile=profile, candidates=candidates,
                                     personalization_lambda=0.7, include_watched=False)
        b_map     = {c["movie_id"]: c["final_score"] for c in b_scored}
        b_scores  = [b_map.get(mid, -99.0) for mid in all_mids]

        ltr_scored = score_candidates_ltr(profile=profile, candidates=candidates,
                                          ltr_model=ltr_model, personalization_lambda=0.7,
                                          include_watched=False)
        ltr_map    = {c["movie_id"]: c["final_score"] for c in ltr_scored}
        ltr_scores = [ltr_map.get(mid, -99.0) for mid in all_mids]

        profile.watch_history = saved_watch  # restore

        baseline_ndcgs.append(ndcg_at_k(relevances, b_scores))
        ltr_ndcgs.append(ndcg_at_k(relevances, ltr_scores))

    print(f"Eval: {len(baseline_ndcgs)} users evaluated, {n_skip} skipped.", flush=True)
    mean_base = float(np.mean(baseline_ndcgs)) if baseline_ndcgs else 0.0
    mean_ltr  = float(np.mean(ltr_ndcgs))      if ltr_ndcgs      else 0.0
    return mean_base, mean_ltr


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    import xgboost as xgb

    print("=" * 72)
    print(f"Phase 3 LTR Training  (seed={RANDOM_SEED}, n_train={N_TRAIN_USERS}, n_eval={N_EVAL_USERS})")
    print("=" * 72)

    build_movie_data_cache(DB_PATH)

    train_users, eval_users = sample_eligible_users(
        DB_PATH, N_TRAIN_USERS, N_EVAL_USERS, MIN_RATINGS, RANDOM_SEED
    )

    print("\nBuilding training feature matrix...")
    X_train, y_train, groups_train = build_training_data(train_users, DB_PATH)
    print(f"X_train shape: {X_train.shape}  groups: {len(groups_train)}")
    print(f"Label range: [{y_train.min()}, {y_train.max()}]  mean={y_train.mean():.2f}")

    print("\nTraining XGBRanker(objective='rank:ndcg')...")
    t0 = time.time()
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, group=groups_train, verbose=False)
    print(f"Training complete in {time.time()-t0:.1f}s")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved to {MODEL_PATH}  ({MODEL_PATH.stat().st_size//1024} KB)")

    importances = model.get_booster().get_score(importance_type="gain")
    print("\nFeature importances (gain):")
    for i, name in enumerate(FEATURE_NAMES):
        score = importances.get(f"f{i}", 0.0)
        print(f"  {name:<30} {score:.2f}")

    print("\nEvaluating closed-pool NDCG@10 on held-out users...")
    ltr_model = load_ltr_model(MODEL_PATH)
    mean_base, mean_ltr = evaluate_closed_pool(eval_users, DB_PATH, ltr_model)

    print()
    print("=" * 72)
    print(f"REAL HOLDOUT — Closed-Pool NDCG@10  (n_eval={len(eval_users)}, n_test={N_TEST})")
    print("Candidate pool: all movies rated by the user (train + test).")
    print("Recall@100 N/A — test items are always in the pool by construction.")
    print("=" * 72)
    print(f"{'scorer':<22}  NDCG@10")
    print(f"{'linear_baseline':<22}  {mean_base:.4f}")
    print(f"{'ltr_rank_ndcg':<22}  {mean_ltr:.4f}")
    delta = mean_ltr - mean_base
    sign  = "+" if delta >= 0 else ""
    print(f"\nDelta (LTR - baseline): {sign}{delta:.4f}")


if __name__ == "__main__":
    main()
