"""
nlp/ltr_scorer.py — LTR scoring function for Phase 3 (togglable path).

Provides score_candidates_ltr(), which mirrors score_candidates() from
nlp/scorer.py but replaces the fixed linear profile-boost with an XGBoost
ranking model trained on ML-25M data.

DESIGN
------
The LTR model was trained on 12 profile+quality features extracted per
(user, movie) pair (see nlp/ltr_train.py for the feature list). It was NOT
trained with rerank_score as a feature (not available at train time, since
training uses rated movies, not retrieved candidates).

At inference, this function:
  1. Extracts the same 12 features for each candidate.
  2. Calls model.inplace_predict(X) to get raw LTR scores.
  3. Normalises LTR scores via sigmoid (same as normalize_profile_boost in
     nlp/scorer.py), mapping them to (0, 1).
  4. Blends with the normalised rerank/RRF score and quality gate using the
     same formula as score_candidates(), with ltr_norm replacing norm_prof:

       combined = 0.75 * norm_rr + 0.25 * norm_rating
       base     = lambda * combined + (1 - lambda) * ltr_norm
       final    = base * w_rating

INTERFACE COMPATIBILITY
-----------------------
score_candidates_ltr(profile, candidates, ltr_model, ...) has the same
signature and return shape as score_candidates(), with one extra required
parameter: ltr_model (a loaded XGBRanker).  Callers that want a direct
drop-in must load the model separately via load_ltr_model().

score_candidates() in nlp/scorer.py is NOT modified.  Both functions remain
callable for direct comparison.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from user_profile.schema import UserProfile


# Default model location — can be overridden by callers.
_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "model" / "ltr_model.ubj"


def load_ltr_model(path: Optional[Path] = None):
    """
    Loads and returns a saved XGBRanker model.

    Parameters
    ----------
    path : Path or None
        Path to the saved model file (.ubj).  Defaults to model/ltr_model.ubj.

    Returns
    -------
    XGBRanker
        Loaded model ready for inference.

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    ImportError      if xgboost is not installed.
    """
    import xgboost as xgb

    model_path = Path(path) if path is not None else _DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"LTR model not found at {model_path}. "
            "Run nlp/ltr_train.py to train and save the model first."
        )
    model = xgb.XGBRanker()
    model.load_model(str(model_path))
    return model


def _extract_features_for_candidate(profile: UserProfile, c: dict) -> List[float]:
    """
    Extracts the 12-feature vector for a single (profile, candidate) pair.
    Matches the feature definition in nlp/ltr_train.py exactly.
    Does NOT apply the watch_history veto (watch filtering is handled
    at the final_score level below).
    """
    num_ratings = float(c.get("num_ratings") or 0)
    avg_rating  = c.get("avg_rating")
    pop_rank    = float(c.get("popularity_rank") or 999_999)

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
        for g in (c.get("genres") or [])
    )

    actor_score = 0.0
    for actor in (c.get("actors") or []):
        if actor in profile.actor_affinity:
            conf = 1.0 if profile.actor_confidence.get(actor, 0) >= 3 else 0.5
            actor_score += profile.actor_affinity[actor] * profile.actor_weight * conf

    director_score = 0.0
    for d in (c.get("directors") or []):
        if d in profile.director_affinity:
            conf = 1.0 if profile.director_confidence.get(d, 0) >= 3 else 0.5
            director_score += profile.director_affinity[d] * profile.director_weight * conf

    tag_score = 0.0
    for tag_item in (c.get("top_tags") or []):
        tag = tag_item if isinstance(tag_item, str) else (
            tag_item.get("tag", "") if isinstance(tag_item, dict) else ""
        )
        if tag:
            tag_score += profile.tag_affinity.get(tag, 0.0) * profile.tag_weight

    era_score = 0.0
    year = c.get("year")
    if year and profile.era_affinity:
        try:
            era = profile.get_era_from_year(int(year))
            era_score = profile.era_affinity.get(era, 0.0) * profile.era_weight
        except (ValueError, TypeError):
            pass

    cr_score = 0.0
    cr = c.get("content_rating") or ""
    if cr and profile.content_rating_affinity:
        cr_score = profile.content_rating_affinity.get(cr, 0.0) * 0.3

    watch_size_log = math.log10(len(profile.watch_history) + 1.0)
    composite      = genre_score + actor_score + director_score + tag_score + era_score + cr_score

    return [
        bayesian_avg, log_volume, norm_quality, pop_rank_log,
        genre_score, actor_score, director_score, tag_score,
        era_score, cr_score, watch_size_log, composite,
    ]


def _sigmoid(val: float) -> float:
    """Numerically stable sigmoid."""
    if val >= 0:
        return 1.0 / (1.0 + math.exp(-val))
    z = math.exp(val)
    return z / (1.0 + z)


def score_candidates_ltr(
    profile: UserProfile,
    candidates: List[Dict[str, Any]],
    ltr_model,
    personalization_lambda: float = 0.7,
    include_watched: bool = False,
    is_obscure_query: bool = False,
) -> List[Dict[str, Any]]:
    """
    LTR-based scoring function.  Drop-in replacement for score_candidates()
    with an additional required parameter: ltr_model (loaded XGBRanker).

    The LTR model's output replaces the norm_prof (profile-boost) term in the
    score_candidates() blending formula.  The rerank/quality blend and quality
    gate are carried over unchanged.

    Parameters
    ----------
    profile : UserProfile
    candidates : list of dict
        Same format as score_candidates() — each dict must have at minimum
        "movie_id".  rerank_score/rrf_score are used if present.
    ltr_model : XGBRanker
        Loaded via load_ltr_model().
    personalization_lambda : float
        Same meaning as in score_candidates() (0.7 = 70% query relevance).
    include_watched : bool
        If False (default), already-watched items get final_score = -10.0.
    is_obscure_query : bool
        Disables quality gate for obscure/indie queries.

    Returns
    -------
    list of dict
        Candidates augmented with "final_score", "ltr_score_raw",
        "ltr_score_norm", and "norm_rerank_score".  Sorted by
        (final_score DESC, movie_id ASC).
    """
    if not candidates:
        return []

    import numpy as np

    # ── 1. Extract features for all candidates ──
    X = np.array(
        [_extract_features_for_candidate(profile, c) for c in candidates],
        dtype=np.float32,
    )

    # ── 2. LTR model inference ──
    ltr_raw_scores = ltr_model.predict(X)  # shape: (N,)

    # ── 3. Normalise LTR scores via sigmoid (scale=2.0, same as normalize_profile_boost) ──
    ltr_norm_scores = [_sigmoid(float(s) / 2.0) for s in ltr_raw_scores]

    scored: List[Dict[str, Any]] = []
    for c, ltr_raw, ltr_norm in zip(candidates, ltr_raw_scores, ltr_norm_scores):
        cand = dict(c)
        mid  = cand["movie_id"]

        # Hard veto: already watched (unless include_watched)
        if mid in profile.watch_history and not include_watched:
            cand["final_score"]        = -10.0
            cand["ltr_score_raw"]      = round(float(ltr_raw), 4)
            cand["ltr_score_norm"]     = round(ltr_norm, 4)
            cand["norm_rerank_score"]  = 0.0
            scored.append(cand)
            continue

        # ── Rerank/RRF signal (same as score_candidates) ──
        s_rr = float(
            cand["rerank_score"] if cand.get("rerank_score") is not None
            else cand.get("rrf_score", 0.0)
        )
        norm_rr = _sigmoid(s_rr)

        # ── Quality score (same formula as score_candidates) ──
        raw_rating  = float(c.get("avg_rating", 3.0)) if c.get("avg_rating") is not None else 3.0
        num_ratings = float(c.get("num_ratings", 0))
        bayes_rating = (num_ratings * raw_rating + 100.0 * 3.2) / (num_ratings + 100.0)
        log_v_boost  = min(1.0, math.log10(num_ratings + 1.0) / 4.5)
        norm_rating  = 0.70 * max(0.0, min(1.0, (bayes_rating - 1.0) / 4.0)) + 0.30 * log_v_boost

        # ── Blending: LTR replaces norm_prof ──
        combined_relevance = (0.75 * norm_rr) + (0.25 * norm_rating)
        base_score = (
            personalization_lambda * combined_relevance
            + (1.0 - personalization_lambda) * ltr_norm
        )

        # ── Quality gate (identical to score_candidates) ──
        if is_obscure_query or raw_rating >= 3.5:
            w_rating = 1.0
        elif raw_rating >= 2.0:
            w_rating = 0.75 + 0.25 * ((raw_rating - 2.0) / 1.5)
        else:
            norm_low = max(0.0, (raw_rating - 0.5) / 1.5)
            w_rating = 0.20 + 0.55 * (norm_low ** 1.5)

        s_final = base_score * w_rating

        cand["final_score"]       = round(s_final, 4)
        cand["ltr_score_raw"]     = round(float(ltr_raw), 4)
        cand["ltr_score_norm"]    = round(ltr_norm, 4)
        cand["norm_rerank_score"] = round(norm_rr, 4)
        scored.append(cand)

    scored.sort(key=lambda x: (-x.get("final_score", -999.0), x.get("movie_id", 0)))
    return scored
