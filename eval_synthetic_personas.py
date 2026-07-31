"""
eval_synthetic_personas.py — Runs the 6 synthetic personas through CineVault's
real pipeline components and computes Recall@100 / NDCG@10 / ILD@10 via
eval/harness.py.

WHY THIS DOESN'T JUST CALL pipeline.recommend(query, user_id=...):
  UserProfileStore.load_profile() only reads the `user_profiles` table (a
  persisted JSON snapshot written by save_profile()). Our synthetic personas
  only exist as raw rows in `ratings` / `user_tags` — they were never saved
  via save_profile(), so load_profile() would silently hand back an empty
  default profile. This script instead reconstructs each profile with
  eval.replay.replay_profile() and injects it directly into score_candidates(),
  bypassing the profile store entirely.

NOTE ON n_test: time_split_user() defaults to n_test=5. synth_contradictory
was designed with only 2 held-out items — using the default there would
silently pull 3 training-set movies into the "test" set. This is handled
via N_TEST_OVERRIDES below; if you add/edit personas, keep this in sync.

NOTE ON A SIMPLIFICATION vs. pipeline.py's recommend(): the production code
preserves each candidate's original rrf_rank when backfilling the -50.0
pinned score for items past the top-100 rerank cutoff (pipeline.py lines
~243-251). That backfill has no effect on any metric this script computes
(rrf_rank isn't read by score_candidates() or MMR once final_score is set),
so it's omitted here for clarity.

NOTE ON original_language: eval/replay.py's _fetch_movie_card() previously
SELECTed a nonexistent `original_language` column from the `movies` table,
crashing replay_profile() immediately. This was fixed directly in eval/replay.py
("Fix eval/replay.py: remove reference to nonexistent original_language column").
In the live pipeline, original_language is sourced from the Tier A/B/C profile
card JSONL files (nlp/hydrator.py), not from the movies table.

NOTE ON QUL CACHE (eval/qul_cache.json):
  The Gemini API returns different expanded_query text even at temperature=0
  (server-side nondeterminism). This causes the dense retrieval lane to
  embed a different query vector on each run, producing different candidate
  pools, which makes recall/NDCG metrics non-reproducible run-to-run.

  To fix this for the eval harness ONLY (without touching nlp/qul.py's
  production behavior), cached_qul_parse() wraps parse_query():
    - On the first call for a given raw query string, parse_query() is
      called once and the result is written to eval/qul_cache.json.
    - On subsequent calls (including across separate process invocations)
      the cached result is returned directly, bypassing the API entirely.
  The cache is keyed by exact raw query string and versioned by QUL_CACHE_VERSION.
  To force a fresh expansion (e.g. after a model or prompt change), either
  delete eval/qul_cache.json or change QUL_CACHE_VERSION below.
  eval/qul_cache.json is a runtime artifact — it is .gitignored.
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.replay import replay_profile, time_split_user
from eval.harness import evaluate_user, load_embedding_from_db
from nlp.pipeline import CineVaultPipeline
from nlp.scorer import score_candidates

EVAL_DB_PATH = PROJECT_ROOT / "db" / "cinevault_eval.db"
SYNTHETIC_JSON_PATH = PROJECT_ROOT / "synthetic_personas.json"
QUL_CACHE_PATH = PROJECT_ROOT / "eval" / "qul_cache.json"

# Bump this string whenever QUL's prompt, model, or schema changes in a way
# that should invalidate previously cached expansions.
QUL_CACHE_VERSION = "temp0-v1"

N_TEST_OVERRIDES = {
    "synth_contradictory": 2,
}

# (condition_label, query_text) per persona. "generic" is identical across
# personas on purpose — it's the ablation that isolates whether ANY profile
# signal can reach retrieval at all, since retrieval never sees the profile.
# "topical" is genre/theme-level but deliberately avoids name-dropping
# (e.g. no "Ridley Scott") so it tests whether ranking correctly favors the
# profile-matching item once candidates are actually in the pool.
QUERIES = {
    "synth_scifi_fan":       [("generic", "a good movie to watch"), ("topical", "science fiction movie")],
    "synth_contradictory":   [("generic", "a good movie to watch"), ("topical", "science fiction movie")],
    "synth_cold_start":      [("generic", "a good movie to watch"), ("topical", "horror movie")],
    "synth_director_driven": [("generic", "a good movie to watch"), ("topical", "sci-fi adventure movie")],
    "synth_obscure_indie":   [("generic", "a good movie to watch"), ("topical", "hidden gem indie movie")],
}

OBSCURE_KEYWORDS = (
    "obscure", "indie", "hidden gem", "cult", "rare", "underrated",
    "under the radar", "unknown", "b-movie", "b movie", "niche",
)

# ─────────────────────────────────────────────────────────────────────────────
# QUL RESULT CACHE
# Scoped to this eval script only — does not affect nlp/qul.py in production.
# ─────────────────────────────────────────────────────────────────────────────

_qul_cache: dict = {}  # in-process dict; populated from disk at startup


def _load_qul_cache() -> None:
    """Loads eval/qul_cache.json into _qul_cache at process start."""
    global _qul_cache
    if QUL_CACHE_PATH.exists():
        try:
            with open(QUL_CACHE_PATH, encoding="utf-8") as f:
                _qul_cache = json.load(f)
            print(f"[qul_cache] Loaded {len(_qul_cache)} cached expansions from {QUL_CACHE_PATH}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[qul_cache] Failed to load cache ({e}), starting empty.")
            _qul_cache = {}
    else:
        print("[qul_cache] No cache file found — will populate on first run.")
        _qul_cache = {}


def _save_qul_cache() -> None:
    """Writes the current in-process cache to disk immediately."""
    QUL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_qul_cache, f, indent=2, ensure_ascii=False)


def cached_qul_parse(pipeline, raw_query: str) -> dict:
    """
    Wraps pipeline.qul.parse_query() with an on-disk cache keyed by the
    exact raw_query string and QUL_CACHE_VERSION.

    Cache hit  → return stored result immediately, no API call.
    Cache miss → call parse_query() once, store result, write cache to disk,
                 return result.

    Writing immediately on miss means a partially-completed run still
    persists its expansions for the next run.
    """
    cache_key = f"{QUL_CACHE_VERSION}::{raw_query}"
    if cache_key in _qul_cache:
        entry = _qul_cache[cache_key]
        # Verify version tag matches (guards against stale data from an old version key)
        if entry.get("qul_version") == QUL_CACHE_VERSION:
            return entry

    # Cache miss — call the API
    result = pipeline.qul.parse_query(raw_query)
    entry = {
        "qul_version":      QUL_CACHE_VERSION,
        "expanded_query":   result.get("expanded_query", raw_query),
        "bm25_keywords":    result.get("bm25_keywords") or [],
        "is_obscure_intent": bool(result.get("is_obscure_intent", False)),
    }
    _qul_cache[cache_key] = entry
    _save_qul_cache()  # write-through on every miss
    return entry


def insert_synthetic_rows(json_path: Path, db_path: Path) -> None:
    """Bulk-inserts the Gemini-generated synthetic ratings/tags into the eval DB copy."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()
    for row in data.get("ratings_insert", []):
        c.execute(
            "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, rated_at) VALUES (?, ?, ?, ?)",
            (row["user_id"], row["movie_id"], row["rating"], row["rated_at"]),
        )
    for row in data.get("user_tags_insert", []):
        c.execute(
            "INSERT INTO user_tags (user_id, movie_id, tag, tagged_at) VALUES (?, ?, ?, ?)",
            (row["user_id"], row["movie_id"], row["tag"], row["tagged_at"]),
        )
    conn.commit()
    conn.close()
    print(f"Inserted {len(data.get('ratings_insert', []))} ratings and "
          f"{len(data.get('user_tags_insert', []))} tag events into {db_path}")


def run_pipeline_for_profile(pipeline, profile, query, top_k=10, candidates_k=250, use_voyage=True):
    """
    Replicates CineVaultPipeline.recommend()'s semantic-search path manually,
    injecting `profile` directly instead of going through profile_store.

    Returns (retrieved_ids, final_results, rerank_score_by_id) — the last dict
    lets the caller check whether a held-out movie survived the top-100
    cross-encoder cutoff (real score) or was pinned to -50.0.
    """
    query_str = query.strip()

    expanded_query = query_str
    bm25_keywords = []
    is_obscure_intent = False
    try:
        qul_result = cached_qul_parse(pipeline, query_str)
        expanded_query = qul_result["expanded_query"]
        bm25_keywords = qul_result["bm25_keywords"]
        is_obscure_intent = qul_result["is_obscure_intent"]
    except Exception as e:
        print(f"    [warn] QUL expansion failed, using raw query: {e}")

    bm25_query_str = " ".join(bm25_keywords) if bm25_keywords else None

    search_hits = pipeline.retriever.search(
        expanded_query, top_k=candidates_k, use_voyage=use_voyage, bm25_query=bm25_query_str
    )
    retrieved_ids = [h["movie_id"] for h in search_hits]

    if not search_hits:
        return retrieved_ids, [], {}

    hydrated_candidates = pipeline.hydrator.hydrate(search_hits)

    to_rerank = hydrated_candidates[:100]
    remaining = hydrated_candidates[100:]

    reranked_top = pipeline.reranker.rerank(query=expanded_query, candidates=to_rerank, top_k=len(to_rerank))
    for r_item in remaining:
        r_item["rerank_score"] = -50.0
    reranked_candidates = reranked_top + remaining
    rerank_score_by_id = {c["movie_id"]: c.get("rerank_score") for c in reranked_candidates}

    is_obscure_query = is_obscure_intent or any(w in query_str.lower() for w in OBSCURE_KEYWORDS)

    scored = score_candidates(
        profile=profile,
        candidates=reranked_candidates,
        personalization_lambda=0.7,
        include_watched=False,
        is_obscure_query=is_obscure_query,
    )

    final_results = pipeline.mmr.filter_diverse(scored, top_k=top_k)
    for rank_idx, item in enumerate(final_results, 1):
        item["final_rank"] = rank_idx

    return retrieved_ids, final_results, rerank_score_by_id


def print_test_item_diagnostics(test, retrieved_ids, rerank_score_by_id, final_results):
    retrieved_set = set(retrieved_ids)
    final_rank_by_id = {r["movie_id"]: r["final_rank"] for r in final_results}
    for t in test:
        mid = t["movie_id"]
        in_pool = mid in retrieved_set
        rerank_score = rerank_score_by_id.get(mid)
        survived_cutoff = rerank_score is not None and rerank_score != -50.0
        final_rank = final_rank_by_id.get(mid)
        print(
            f"      movie_id={mid:<7} rating={t['rating']:<4} "
            f"in_top250={in_pool!s:<6} survived_rerank_cutoff={survived_cutoff!s:<6} "
            f"final_top10_rank={final_rank if final_rank else '—'}"
        )


def main():
    if not EVAL_DB_PATH.exists():
        print(f"ERROR: {EVAL_DB_PATH} does not exist.")
        print(f"Run:  cp db/cinevault.db {EVAL_DB_PATH}")
        print("Never insert synthetic rows into the production DB directly.")
        return

    if not SYNTHETIC_JSON_PATH.exists():
        print(f"ERROR: expected the Gemini-generated persona JSON at {SYNTHETIC_JSON_PATH}")
        return

    insert_synthetic_rows(SYNTHETIC_JSON_PATH, EVAL_DB_PATH)

    _load_qul_cache()

    print("\nInitializing pipeline against the EVAL DB copy (loads reranker + indexes)...")
    pipeline = CineVaultPipeline(load_dense=True, db_path=EVAL_DB_PATH, lazy_load_models=False)
    embedding_fn = load_embedding_from_db(db_path=EVAL_DB_PATH, column="v_genome")

    results_table = []

    for user_id in QUERIES:
        n_test = N_TEST_OVERRIDES.get(user_id, 5)
        train, test = time_split_user(user_id, n_test=n_test, db_path=EVAL_DB_PATH)

        if not train:
            print(f"[skip] {user_id}: insufficient history for a train/test split")
            continue

        cutoff_ts = train[-1]["rated_at"]
        profile = replay_profile(user_id, cutoff_ts=cutoff_ts, db_path=EVAL_DB_PATH)

        print(f"\n{'=' * 78}\n{user_id}  (train={len(train)}, test={len(test)}, cutoff_ts={cutoff_ts})\n{'=' * 78}")

        for condition_label, query in QUERIES[user_id]:
            retrieved_ids, final_results, rerank_score_by_id = run_pipeline_for_profile(
                pipeline, profile, query, top_k=10, candidates_k=250
            )

            metrics = evaluate_user(
                user_id=user_id,
                test_interactions=test,
                ranked_results=final_results,
                retrieved_ids=retrieved_ids,
                embedding_fn=embedding_fn,
                k_recall=100,
                k_ndcg=10,
                k_ild=10,
                min_star_relevant=4.0,
            )
            results_table.append({"user_id": user_id, "condition": condition_label, "query": query, **metrics})

            ild_str = f"{metrics['ild']:.3f}" if metrics["ild"] is not None else "N/A"
            print(f"\n  [{condition_label}] query={query!r}")
            print(f"    Recall@100={metrics['recall_at_k']:.3f}  NDCG@10={metrics['ndcg_at_k']:.3f}  ILD@10={ild_str}")
            print(f"    Held-out test item trace:")
            print_test_item_diagnostics(test, retrieved_ids, rerank_score_by_id, final_results)

    # synth_tag_only: no ratings, no train/test split — inspect the tag-affinity
    # path in isolation instead of computing recall/NDCG.
    print(f"\n{'=' * 78}\nsynth_tag_only  (tag-affinity path only)\n{'=' * 78}")
    tag_profile = replay_profile("synth_tag_only", cutoff_ts=2000000000, db_path=EVAL_DB_PATH)
    print(f"  tag_affinity: {tag_profile.tag_affinity}")
    print(f"  watch_history size: {len(tag_profile.watch_history)}")

    print(f"\n\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'user_id':<24}{'condition':<10}{'recall@100':<12}{'ndcg@10':<10}{'ild@10'}")
    for r in results_table:
        ild_str = f"{r['ild']:.3f}" if r["ild"] is not None else "N/A"
        print(f"{r['user_id']:<24}{r['condition']:<10}{r['recall_at_k']:<12.3f}{r['ndcg_at_k']:<10.3f}{ild_str}")


if __name__ == "__main__":
    main()