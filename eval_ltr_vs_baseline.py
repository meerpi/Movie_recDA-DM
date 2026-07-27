"""
eval_ltr_vs_baseline.py — Side-by-side comparison of the linear baseline
(score_candidates) vs the LTR model (score_candidates_ltr) on the 6 synthetic
personas defined in eval_synthetic_personas.py.

Both scorers are evaluated against the SAME retrieved and reranked candidate
pool for each (persona, condition) pair.  Recall@100, NDCG@10, and ILD@10 are
reported for each scorer.  The QUL expansion cache from eval_synthetic_personas.py
is reused so that candidate pools are stable across runs.

Run from project root:
    .venv/bin/python eval_ltr_vs_baseline.py
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse cache infrastructure from eval_synthetic_personas
from eval_synthetic_personas import (
    EVAL_DB_PATH, SYNTHETIC_JSON_PATH,
    QUL_CACHE_PATH, QUL_CACHE_VERSION,
    N_TEST_OVERRIDES, QUERIES, OBSCURE_KEYWORDS,
    _load_qul_cache, cached_qul_parse,
    insert_synthetic_rows, print_test_item_diagnostics,
)
from eval.replay import replay_profile, time_split_user
from eval.harness import evaluate_user, load_embedding_from_db
from nlp.pipeline import CineVaultPipeline
from nlp.scorer import score_candidates
from nlp.ltr_scorer import score_candidates_ltr, load_ltr_model
from nlp.retrieval_augment import entity_backstop, FallbackDenseRetriever

FALLBACK_INDEX_PATH = PROJECT_ROOT / "dirtywork" / "fallback_dense.hnsw"


LTR_MODEL_PATH = PROJECT_ROOT / "model" / "ltr_model.ubj"


def run_retrieve_and_rerank(pipeline, profile, query, candidates_k=500, use_voyage=True, _fallback_retriever=None):
    """
    Runs the pipeline up to and including the reranking step.
    Returns (retrieved_ids, reranked_candidates) without scoring.
    The caller applies both scorers to reranked_candidates.
    """
    query_str = query.strip()
    expanded_query = query_str
    bm25_keywords  = []
    is_obscure_intent = False
    try:
        qul_result = cached_qul_parse(pipeline, query_str)
        expanded_query    = qul_result["expanded_query"]
        bm25_keywords     = qul_result["bm25_keywords"]
        is_obscure_intent = qul_result["is_obscure_intent"]
    except Exception as e:
        print(f"    [warn] QUL cache failed, using raw query: {e}")

    bm25_query_str = " ".join(bm25_keywords) if bm25_keywords else None

    # Fix 1: raise lane_k 100→200 for 2x per-lane recall
    search_hits = pipeline.retriever.search(
        expanded_query, top_k=candidates_k, lane_k=200,
        use_voyage=use_voyage, bm25_query=bm25_query_str
    )
    retrieved_ids_set = {h["movie_id"] for h in search_hits}

    # Fix 2: profile-entity SQL backstop (directors / actors)
    backstop_hits = entity_backstop(
        profile, db_path=pipeline.db_path,
        already_retrieved=retrieved_ids_set,
        n_directors=3, n_actors=5, max_per_entity=20,
    )
    search_hits = search_hits + backstop_hits
    retrieved_ids_set = {h["movie_id"] for h in search_hits}

    # Fix 3: local fallback dense retrieval (sentence-transformer 384-dim index)
    if _fallback_retriever is not None:
        fallback_hits = _fallback_retriever.search(
            query=expanded_query,
            already_retrieved=retrieved_ids_set,
            top_k=50,
        )
        search_hits = search_hits + fallback_hits
        retrieved_ids_set = {h["movie_id"] for h in search_hits}

    retrieved_ids = list(retrieved_ids_set)

    if not search_hits:
        return retrieved_ids, [], expanded_query, is_obscure_intent

    hydrated = pipeline.hydrator.hydrate(search_hits)

    # Still send only top 100 (by rrf_score) to the expensive cross-encoder
    # Rerank top 250 candidates (including entity backstop hits)
    # Entity backstop items are prioritized for reranking if they match director/actor affinities
    entity_hits = [c for c in hydrated if any(l in c.get("lanes", []) for l in ("entity_director", "entity_actor", "fallback_dense"))]
    regular_hits = [c for c in hydrated if c not in entity_hits]
    
    # Rerank up to 200 regular hits + all entity/fallback hits
    to_rerank = regular_hits[:200] + entity_hits
    remaining = regular_hits[200:]
    reranked_top = pipeline.reranker.rerank(
        query=expanded_query, candidates=to_rerank, top_k=len(to_rerank)
    )
    for r in remaining:
        r["rerank_score"] = -50.0
    reranked_candidates = reranked_top + remaining

    return retrieved_ids, reranked_candidates, expanded_query, is_obscure_intent


def run_both_scorers(pipeline, profile, query, ltr_model, top_k=10, candidates_k=500, _fallback_retriever=None):
    """
    Returns:
        retrieved_ids         — for Recall@100 (same for both scorers)
        baseline_results      — top_k results from score_candidates()
        ltr_results           — top_k results from score_candidates_ltr()
        rerank_score_by_id    — for test-item diagnostics (same for both)
    """
    retrieved_ids, reranked_candidates, expanded_query, is_obscure_intent = \
        run_retrieve_and_rerank(pipeline, profile, query, candidates_k=candidates_k,
                                _fallback_retriever=_fallback_retriever)

    if not reranked_candidates:
        return retrieved_ids, [], [], {}

    query_str = query.strip()
    is_obscure_query = is_obscure_intent or any(w in query_str.lower() for w in OBSCURE_KEYWORDS)

    rerank_score_by_id = {c["movie_id"]: c.get("rerank_score") for c in reranked_candidates}

    # ── linear baseline ──
    b_scored = score_candidates(
        profile=profile, candidates=reranked_candidates,
        personalization_lambda=0.7, include_watched=False,
        is_obscure_query=is_obscure_query,
    )
    b_diverse = pipeline.mmr.filter_diverse(b_scored, top_k=top_k)
    for i, item in enumerate(b_diverse, 1):
        item["final_rank"] = i

    # ── LTR ──
    # Make a fresh copy of reranked_candidates so scorer can modify dicts safely
    reranked_copy = [dict(c) for c in reranked_candidates]
    ltr_scored = score_candidates_ltr(
        profile=profile, candidates=reranked_copy,
        ltr_model=ltr_model, personalization_lambda=0.7,
        include_watched=False, is_obscure_query=is_obscure_query,
    )
    ltr_diverse = pipeline.mmr.filter_diverse(ltr_scored, top_k=top_k)
    for i, item in enumerate(ltr_diverse, 1):
        item["final_rank"] = i

    return retrieved_ids, b_diverse, ltr_diverse, rerank_score_by_id


def main():
    if not EVAL_DB_PATH.exists():
        print(f"ERROR: {EVAL_DB_PATH} does not exist.")
        print(f"Run:  cp db/cinevault.db {EVAL_DB_PATH}")
        return

    if not SYNTHETIC_JSON_PATH.exists():
        print(f"ERROR: synthetic_personas.json not found.")
        return

    if not LTR_MODEL_PATH.exists():
        print(f"ERROR: LTR model not found at {LTR_MODEL_PATH}.")
        print("Run nlp/ltr_train.py first to train the model.")
        return

    insert_synthetic_rows(SYNTHETIC_JSON_PATH, EVAL_DB_PATH)
    _load_qul_cache()

    print("\nInitializing pipeline and loading LTR model...")
    pipeline  = CineVaultPipeline(load_dense=True, db_path=EVAL_DB_PATH, lazy_load_models=False)
    ltr_model = load_ltr_model(LTR_MODEL_PATH)

    # Fix 3: load fallback dense retriever if the index exists
    fallback_retriever = None
    if FALLBACK_INDEX_PATH.exists():
        try:
            fallback_retriever = FallbackDenseRetriever()
            print(f"  Fallback dense index loaded.")
        except Exception as e:
            print(f"  [warn] Fallback dense retriever failed to load: {e}")
    else:
        print(f"  [info] No fallback dense index found at {FALLBACK_INDEX_PATH}.")
        print(f"         Run: .venv/bin/python scripts/build_fallback_index.py")
    embedding_fn = load_embedding_from_db(db_path=EVAL_DB_PATH, column="v_genome")

    baseline_table = []
    ltr_table      = []

    for user_id in QUERIES:
        n_test  = N_TEST_OVERRIDES.get(user_id, 5)
        train, test = time_split_user(user_id, n_test=n_test, db_path=EVAL_DB_PATH)

        if not train:
            print(f"[skip] {user_id}: insufficient history")
            continue

        cutoff_ts = train[-1]["rated_at"]
        profile   = replay_profile(user_id, cutoff_ts=cutoff_ts, db_path=EVAL_DB_PATH)

        print(f"\n{'='*78}\n{user_id}  (train={len(train)}, test={len(test)}, cutoff_ts={cutoff_ts})\n{'='*78}")

        for condition_label, query in QUERIES[user_id]:
            retrieved_ids, b_results, ltr_results, rerank_score_by_id = run_both_scorers(
                pipeline, profile, query, ltr_model, top_k=10, candidates_k=500,
                _fallback_retriever=fallback_retriever,
            )

            b_metrics = evaluate_user(
                user_id=user_id, test_interactions=test, ranked_results=b_results,
                retrieved_ids=retrieved_ids, embedding_fn=embedding_fn,
                k_recall=100, k_ndcg=10, k_ild=10, min_star_relevant=4.0,
            )
            ltr_metrics = evaluate_user(
                user_id=user_id, test_interactions=test, ranked_results=ltr_results,
                retrieved_ids=retrieved_ids, embedding_fn=embedding_fn,
                k_recall=100, k_ndcg=10, k_ild=10, min_star_relevant=4.0,
            )

            baseline_table.append({"user_id": user_id, "condition": condition_label, "query": query, **b_metrics})
            ltr_table.append({"user_id": user_id, "condition": condition_label, "query": query, **ltr_metrics})

            def ild_s(m): return f"{m['ild']:.3f}" if m["ild"] is not None else "N/A"
            print(f"\n  [{condition_label}] query={query!r}")
            print(f"    [baseline] R@100={b_metrics['recall_at_k']:.3f}  NDCG@10={b_metrics['ndcg_at_k']:.3f}  ILD@10={ild_s(b_metrics)}")
            print(f"    [ltr     ] R@100={ltr_metrics['recall_at_k']:.3f}  NDCG@10={ltr_metrics['ndcg_at_k']:.3f}  ILD@10={ild_s(ltr_metrics)}")
            print(f"    Held-out test item trace (pool same for both):")
            print_test_item_diagnostics(test, retrieved_ids, rerank_score_by_id, b_results)

    print(f"\n\n{'='*78}")
    print("SYNTHETIC PERSONA SUMMARY — Linear Baseline vs LTR Model")
    print(f"{'='*78}")
    print(f"{'user_id':<24}{'cond':<10}{'scorer':<10}{'R@100':<10}{'NDCG@10':<10}{'ILD@10'}")
    for b_row, ltr_row in zip(baseline_table, ltr_table):
        ild_b   = f"{b_row['ild']:.3f}"   if b_row['ild']   is not None else "N/A"
        ild_ltr = f"{ltr_row['ild']:.3f}" if ltr_row['ild'] is not None else "N/A"
        uid  = b_row['user_id']
        cond = b_row['condition']
        print(f"{uid:<24}{cond:<10}{'baseline':<10}{b_row['recall_at_k']:<10.3f}{b_row['ndcg_at_k']:<10.3f}{ild_b}")
        print(f"{'':<24}{'':<10}{'ltr':<10}{ltr_row['recall_at_k']:<10.3f}{ltr_row['ndcg_at_k']:<10.3f}{ild_ltr}")
        print()


if __name__ == "__main__":
    main()
