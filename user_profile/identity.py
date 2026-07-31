"""user_profile/identity.py — User ID validation and anchor movie resolution."""

import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

# safe chars for new user IDs
_SAFE_USER_ID_RE = re.compile(r'^[A-Za-z0-9_\-]{1,64}$')


def validate_user_id(user_id, *, db_path):
    """Validates a user_id. Existing DB IDs always accepted as-is.
    New IDs: stripped, non-empty, alphanumeric/hyphen/underscore, 1-64 chars.
    Exits with code 2 on failure."""
    stripped = user_id.strip()
    if not stripped:
        print(
            "ERROR: --user / -u must not be empty or whitespace-only.\n"
            "Example valid user IDs: alice, user_1, my-profile",
            file=sys.stderr,
        )
        sys.exit(2)

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            existing = conn.execute(
                "SELECT 1 FROM user_profiles WHERE user_id = ? LIMIT 1", (stripped,)
            ).fetchone()
            conn.close()
            if existing:
                return stripped
        except Exception:
            pass

    if not _SAFE_USER_ID_RE.match(stripped):
        print(
            f"ERROR: New user ID {stripped!r} contains invalid characters.\n"
            f"Allowed: letters, digits, hyphens (-) and underscores (_), 1-64 characters.\n"
            f"Example valid user IDs: alice, user_1, my-profile",
            file=sys.stderr,
        )
        sys.exit(2)

    return stripped


def lookup_anchor_by_title(title, db_path):
    """Case-insensitive LIKE search, returns up to 3 (movie_id, title) tuples."""
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT movie_id, title FROM movies "
            "WHERE LOWER(title) LIKE ? LIMIT 3",
            (f"%{title.lower()}%",),
        ).fetchall()
        conn.close()
        return [(int(r[0]), str(r[1])) for r in rows]
    except Exception:
        return []


def resolve_anchor_tokens(raw, db_path):
    """Parses comma-separated anchor input. Digits → movie IDs directly,
    text → title search. Returns (resolved_ids, warnings)."""
    resolved = []
    warnings = []

    for val in raw.split(","):
        val = val.strip()
        if not val:
            continue
        if val.isdigit():
            resolved.append(int(val))
        else:
            matches = lookup_anchor_by_title(val, db_path)
            if not matches:
                warnings.append(f"No movie found matching {val!r} — skipped.")
            elif len(matches) == 1:
                resolved.append(matches[0][0])
            else:
                resolved.append(matches[0][0])
                warnings.append(
                    f"Multiple matches for {val!r}; using {matches[0][1]!r} "
                    f"(ID {matches[0][0]}). Use a movie ID for exact selection."
                )

    return resolved, warnings
