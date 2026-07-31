"""
eval/replay.py — Time-aware UserProfile reconstruction for offline evaluation.

Provides two functions used by the Phase 2 evaluation harness:

  replay_profile(user_id, cutoff_ts, db_path)
      Reconstructs a UserProfile as of ``cutoff_ts`` by replaying the user's
      rating and user_tag events in chronological order through the production
      ``UserProfile.apply_rating_update()`` method.  The result is mathematically
      identical to the profile the production system would have built from the same
      event stream up to that moment in time — crucially, no affinity update logic
      is duplicated here.

  time_split_user(user_id, n_test, db_path)
      Partitions a user's rating history into a chronological train set and a held-
      out test set consisting of the most-recent ``n_test`` interactions.

Design notes
------------
- The ``ratings`` and ``user_tags`` tables both carry Unix timestamps
  (``rated_at`` and ``tagged_at`` respectively), so chronological ordering is
  straightforward.
- The ``reviews.review_date`` column is stored as TEXT; ``_parse_review_date``
  normalises it to a Unix timestamp for any caller that needs to merge review
  events into the same stream (not currently included in replay_profile, since
  the review processor's output is already captured by the associated rating event).
- Tag events from ``user_tags`` are treated as implicit positive signals:
  the movie is added to ``watch_history`` and the tag string is incremented in
  ``tag_affinity`` by +0.3, clamped via the production ``_clamp()`` helper.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from user_profile.schema import UserProfile, _clamp

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_review_date(review_date: str) -> Optional[int]:
    """
    Converts a ``reviews.review_date`` TEXT value to a Unix timestamp (int).

    Supported formats (as written by the ingestion pipeline):
      - ``'YYYY-MM-DD'``
      - ``'YYYY-MM-DDTHH:MM:SS'``
      - ``'YYYY-MM-DD HH:MM:SS'``

    Returns ``None`` if the value is absent or cannot be parsed, so callers can
    silently drop malformed rows rather than crashing the replay.

    Parameters
    ----------
    review_date : str
        Raw value from the ``reviews.review_date`` column.

    Returns
    -------
    int or None
        Unix timestamp, or None on parse failure.
    """
    if not review_date:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(review_date.strip(), fmt)
            # Treat naive datetimes as local time (same assumption as production ingestion)
            return int(dt.replace(tzinfo=timezone.utc).timestamp())
        except ValueError:
            continue
    return None


def _fetch_movie_card(conn: sqlite3.Connection, movie_id: int) -> Dict[str, Any]:
    """
    Builds a minimal movie card dict for ``UserProfile.apply_rating_update()``.

    Fetches title, year, content_rating, actors, directors, and genres from
    SQLite.  Fields absent from the DB schema (original_language, tone, pacing,
    top_tags, production_countries, collection) are omitted from the SELECT;
    ``original_language`` is defaulted to ``""`` so that the language-affinity
    update in ``apply_rating_update()`` (guarded by ``if lang:``) silently
    no-ops — the same pattern already used for all the other absent fields.

    Note: ``original_language`` is not a column in the ``movies`` table of this
    database.  In the live pipeline it is sourced from the Tier A/B/C profile
    card JSONL files loaded by ``nlp/hydrator.py``, not from SQLite.

    Actors and directors are stored in the ``movies`` table as either a JSON
    array string or a comma-separated plain string, depending on the ingestion
    path; both formats are handled.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to cinevault.db.
    movie_id : int
        PK to look up.

    Returns
    -------
    dict
        Minimal movie card ready for ``apply_rating_update()``.
    """
    c = conn.cursor()

    row = c.execute(
        "SELECT title, year, content_rating, actors, directors "
        "FROM movies WHERE movie_id = ?",
        (movie_id,),
    ).fetchone()

    # original_language is not a column in the movies table; it lives in the
    # Tier A/B/C profile card files loaded by nlp/hydrator.py.  Default to ""
    # so apply_rating_update()'s "if lang:" guard no-ops cleanly.
    card: Dict[str, Any] = {"movie_id": movie_id, "original_language": ""}
    if row:
        card["title"]          = row[0]
        card["year"]           = row[1]
        card["content_rating"] = row[2]

        for field_name, raw_val in (("actors", row[3]), ("directors", row[4])):
            if not raw_val:
                card[field_name] = []
                continue
            try:
                parsed = json.loads(raw_val)
                card[field_name] = parsed if isinstance(parsed, list) else [str(parsed)]
            except (json.JSONDecodeError, TypeError):
                card[field_name] = [s.strip() for s in raw_val.split(",") if s.strip()]

    genre_rows = c.execute(
        """
        SELECT g.name
        FROM   genres g
        JOIN   movie_genres mg ON g.genre_id = mg.genre_id
        WHERE  mg.movie_id = ?
        """,
        (movie_id,),
    ).fetchall()
    card["genres"] = [r[0] for r in genre_rows]

    return card


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay_profile(
    user_id: str,
    cutoff_ts: int,
    db_path: Path = DB_PATH,
) -> UserProfile:
    """
    Reconstructs a UserProfile as of a given Unix timestamp.

    All rating events with ``rated_at <= cutoff_ts`` and tag events with
    ``tagged_at <= cutoff_ts`` are merged into a single chronological stream and
    replayed in timestamp order.  Events strictly after ``cutoff_ts`` are excluded,
    making the result exactly the profile the production system would have held at
    time T.

    Rating events are replayed through ``UserProfile.apply_rating_update()``, which
    is the identical code path used in production.  This guarantees that genre,
    tone, tag, pacing, era, language, country, actor, and director affinities, as
    well as confidence counters and watch-state buckets, are computed the same way
    the live system computes them — any future refactor of that method is
    automatically reflected here.

    Tag events from ``user_tags`` are treated as implicit positive signals:
    the movie is added to ``watch_history`` and the tag string is incremented in
    ``tag_affinity`` by a fixed delta of +0.3, clamped to [−3.0, 3.0] via
    ``_clamp()``.

    Parameters
    ----------
    user_id : str
        User to reconstruct.  The ``ratings`` / ``user_tags`` tables store
        ``user_id`` as INTEGER; this function casts to int before querying.
    cutoff_ts : int
        Inclusive upper bound (Unix seconds).
    db_path : Path
        Path to cinevault.db.

    Returns
    -------
    UserProfile
        Freshly constructed profile with all signals up to ``cutoff_ts`` applied.
    """
    profile = UserProfile(user_id=str(user_id))
    conn    = sqlite3.connect(str(db_path))

    # C1 fix: ratings/user_tags.user_id is now TEXT; pass the string directly.
    uid = str(user_id)

    try:
        rating_rows = conn.execute(
            "SELECT movie_id, rating, rated_at FROM ratings "
            "WHERE user_id = ? AND rated_at <= ? ORDER BY rated_at",
            (uid, cutoff_ts),
        ).fetchall()

        rating_events = [
            {
                "type":     "rating",
                "ts":       int(row[2]),
                "movie_id": int(row[0]),
                "rating":   float(row[1]),
            }
            for row in rating_rows
        ]

        tag_rows = conn.execute(
            "SELECT movie_id, tag, tagged_at FROM user_tags "
            "WHERE user_id = ? AND tagged_at <= ? ORDER BY tagged_at",
            (uid, cutoff_ts),
        ).fetchall()

        tag_events = [
            {
                "type":     "tag",
                "ts":       int(row[2]),
                "movie_id": int(row[0]),
                "tag":      str(row[1]),
            }
            for row in tag_rows
        ]

        # Merge and sort: primary key = timestamp, secondary = type so ratings
        # are processed before same-second tags (rating populates watch_history
        # first, which is the more authoritative signal).
        all_events = sorted(
            rating_events + tag_events,
            key=lambda e: (e["ts"], 0 if e["type"] == "rating" else 1),
        )

        _card_cache: Dict[int, Dict[str, Any]] = {}

        for event in all_events:
            mid = event["movie_id"]
            if mid not in _card_cache:
                _card_cache[mid] = _fetch_movie_card(conn, mid)
            card = _card_cache[mid]

            if event["type"] == "rating":
                profile.apply_rating_update(card, star_rating=event["rating"])

            elif event["type"] == "tag":
                profile.watch_history.add(mid)
                tag = event["tag"]
                profile.tag_affinity[tag] = _clamp(
                    profile.tag_affinity.get(tag, 0.0) + 0.3
                )

    finally:
        conn.close()

    return profile


def time_split_user(
    user_id: str,
    n_test: int = 5,
    db_path: Path = DB_PATH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Chronological train/test split for a single user.

    Retrieves all rating interactions for ``user_id`` ordered by ``rated_at``
    ascending.  The most-recent ``n_test`` interactions form the test set;
    everything earlier forms the train set.

    The cutoff timestamp for ``replay_profile()`` should be set to
    ``train[-1]["rated_at"]`` (the last train event's timestamp) so that the
    reconstructed profile contains exactly the training-period signal.

    If the user has fewer than ``n_test + 1`` total interactions the train set
    will be empty; callers should filter such users before computing metrics to
    avoid NDCG / Recall being evaluated with no training signal.

    Parameters
    ----------
    user_id : str
        User to split.
    n_test : int
        Number of most-recent interactions to hold out.
    db_path : Path
        Path to cinevault.db.

    Returns
    -------
    train : list of dict
        Each element has keys: ``user_id``, ``movie_id``, ``rating``, ``rated_at``.
    test : list of dict
        Same structure.  Empty list if user has no interactions.
    """
    conn = sqlite3.connect(str(db_path))

    # C1 fix: user_id is TEXT in the migrated schema.
    uid = str(user_id)

    try:
        rows = conn.execute(
            "SELECT user_id, movie_id, rating, rated_at FROM ratings "
            "WHERE user_id = ? ORDER BY rated_at ASC",
            (uid,),
        ).fetchall()
    finally:
        conn.close()

    interactions = [
        {
            "user_id":  str(row[0]),
            "movie_id": int(row[1]),
            "rating":   float(row[2]),
            "rated_at": int(row[3]),
        }
        for row in rows
    ]

    if len(interactions) <= n_test:
        return [], interactions

    split_idx = len(interactions) - n_test
    return interactions[:split_idx], interactions[split_idx:]
