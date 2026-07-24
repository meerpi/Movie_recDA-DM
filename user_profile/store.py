#!/usr/bin/env python3
"""
profile/store.py — Step 10: User Profile Storage Layer

Manages saving, loading, and updating UserProfile instances in SQLite database.
Stores user profile JSON blobs in `db/cinevault.db` table `user_profiles`.

Usage:
    from profile.store import UserProfileStore

    store = UserProfileStore()
    profile = store.load_profile("user_123")
    profile.apply_rating_update(movie_card, star_rating=5.0)
    store.save_profile(profile)
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from user_profile.schema import UserProfile

logger = logging.getLogger("cinevault.profile_store")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"


class UserProfileStore:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Ensures `user_profiles` table exists in database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def load_profile(self, user_id: str) -> UserProfile:
        """Loads UserProfile from database, or returns fresh empty profile if not found."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        row = c.execute(
            "SELECT profile_json FROM user_profiles WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        conn.close()

        if row:
            try:
                data = json.loads(row[0])
                return UserProfile.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to parse JSON for user_id={user_id}: {e}")

        # Default fresh profile
        return UserProfile(user_id=user_id)

    def list_users(self) -> List[str]:
        """Returns list of all saved user IDs in SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        rows = c.execute("SELECT user_id FROM user_profiles ORDER BY updated_at DESC").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def save_profile(self, profile: UserProfile):
        """Saves or updates UserProfile instance in database."""
        data = profile.to_dict()
        profile_json = json.dumps(data)

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO user_profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
        """, (profile.user_id, profile_json))
        conn.commit()
        conn.close()
        logger.info(f"Saved UserProfile for user_id='{profile.user_id}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = UserProfileStore()

    user = store.load_profile("test_user_42")
    user.actor_affinity["Keanu Reeves"] = 0.95
    user.genre_affinity["Sci-Fi"] = 0.90
    user.watch_history.add(603)  # The Matrix

    store.save_profile(user)

    reloaded = store.load_profile("test_user_42")
    print(f"Reloaded user '{reloaded.user_id}' profile:")
    print(f"  Actor affinity: {reloaded.actor_affinity}")
    print(f"  Genre affinity: {reloaded.genre_affinity}")
    print(f"  Watch history : {reloaded.watch_history}")
