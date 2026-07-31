"""user_profile/store.py — SQLite profile persistence with optimistic locking.

Migrations (applied on first init):
  M1: ratings.user_id INTEGER → TEXT
  M2: user_tags.user_id INTEGER → TEXT
  M3: reviews.user_id TEXT column added
  M4: user_profiles.version column (optimistic concurrency)
  M5: user_profiles_recovery table
  M6: user_profile_snapshots table
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List

from user_profile.schema import UserProfile

logger = logging.getLogger("cinevault.profile_store")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "db" / "cinevault.db"


class ProfileConflictError(Exception):
    pass


class UserProfileStore:

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")

        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id      TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                version      INTEGER NOT NULL DEFAULT 1
            )
        """)

        # M4 — version column for existing DBs
        existing_cols = {
            row[1] for row in c.execute("PRAGMA table_info(user_profiles)").fetchall()
        }
        if "version" not in existing_cols:
            c.execute(
                "ALTER TABLE user_profiles ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
            logger.info("Migration M4: added version column to user_profiles")

        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles_recovery (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                corrupt_json TEXT NOT NULL,
                backed_up_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS user_profile_snapshots (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                snapshot_ts  INTEGER NOT NULL
            )
        """)
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_user_ts "
            "ON user_profile_snapshots(user_id, snapshot_ts)"
        )

        conn.commit()

        self._migrate_catalog_user_id_to_text(conn)
        self._migrate_reviews_add_user_id(conn)

        conn.close()

    def _migrate_catalog_user_id_to_text(self, conn):
        """M1+M2: ratings/user_tags user_id INTEGER → TEXT. Idempotent — checks
        column type before running. Existing int values become string repr."""
        c = conn.cursor()
        for table in ("ratings", "user_tags"):
            exists = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if not exists:
                continue

            col_info = {
                row[1]: row[2]
                for row in c.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if col_info.get("user_id", "").upper() == "TEXT":
                continue

            logger.info(f"Migration M1/M2: converting {table}.user_id INTEGER → TEXT")

            c.execute(f"DROP TABLE IF EXISTS {table}_migrated")

            if table == "ratings":
                c.execute("""
                    CREATE TABLE ratings_migrated (
                        user_id  TEXT NOT NULL,
                        movie_id INTEGER,
                        rating   REAL,
                        rated_at INTEGER,
                        PRIMARY KEY (user_id, movie_id)
                    )
                """)
                c.execute("""
                    INSERT OR IGNORE INTO ratings_migrated (user_id, movie_id, rating, rated_at)
                    SELECT CAST(user_id AS TEXT), movie_id, rating, rated_at
                    FROM ratings
                """)
            else:
                c.execute("""
                    CREATE TABLE user_tags_migrated (
                        user_id   TEXT NOT NULL,
                        movie_id  INTEGER,
                        tag       TEXT,
                        tagged_at INTEGER
                    )
                """)
                c.execute("""
                    INSERT INTO user_tags_migrated (user_id, movie_id, tag, tagged_at)
                    SELECT CAST(user_id AS TEXT), movie_id, tag, tagged_at
                    FROM user_tags
                """)

            c.execute(f"DROP TABLE {table}")
            c.execute(f"ALTER TABLE {table}_migrated RENAME TO {table}")

            if table == "user_tags":
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_tags_user "
                    "ON user_tags(user_id)"
                )
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_tags_movie "
                    "ON user_tags(movie_id)"
                )
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_tags_triple "
                    "ON user_tags(user_id, movie_id, tag)"
                )

            conn.commit()
            logger.info(f"Migration M1/M2: {table}.user_id → TEXT complete")

    def _migrate_reviews_add_user_id(self, conn):
        c = conn.cursor()
        exists = c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'"
        ).fetchone()
        if not exists:
            return
        col_names = {
            row[1] for row in c.execute("PRAGMA table_info(reviews)").fetchall()
        }
        if "user_id" not in col_names:
            c.execute("ALTER TABLE reviews ADD COLUMN user_id TEXT")
            conn.commit()
            logger.info("Migration M3: added user_id column to reviews")

    def user_exists(self, user_id):
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT 1 FROM user_profiles WHERE user_id = ? LIMIT 1", (user_id,)
        ).fetchone()
        conn.close()
        return row is not None

    def load_profile(self, user_id):
        """Loads profile from DB. If the JSON blob is corrupt, backs it up to
        the recovery table and returns a fresh default profile."""
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT profile_json, version FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()

        if row is None:
            return UserProfile(user_id=user_id)

        raw_json, db_version = row[0], (row[1] if row[1] is not None else 1)

        try:
            data = json.loads(raw_json)
            profile = UserProfile.from_dict(data)
            profile._db_version = db_version
            return profile
        except Exception as exc:
            logger.error(
                f"Profile JSON for {user_id!r} corrupt: {exc}. "
                f"Backed up to recovery table, returning defaults."
            )
            self._backup_corrupt_profile(user_id, raw_json)
            fresh = UserProfile(user_id=user_id)
            fresh._db_version = db_version
            return fresh

    def _backup_corrupt_profile(self, user_id, corrupt_json):
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT INTO user_profiles_recovery (user_id, corrupt_json) VALUES (?, ?)",
                (user_id, corrupt_json),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Could not write to recovery table: {e}")

    def save_profile(self, profile, *, snapshot=False):
        """Optimistic concurrency save. Raises ProfileConflictError if the row
        version doesn't match (another session wrote in between)."""
        import time as _time

        data = profile.to_dict()
        profile_json = json.dumps(data)
        expected_ver = getattr(profile, "_db_version", 0)
        new_ver = expected_ver + 1

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        if expected_ver == 0:
            c.execute(
                """
                INSERT INTO user_profiles (user_id, profile_json, updated_at, version)
                VALUES (?, ?, CURRENT_TIMESTAMP, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = CURRENT_TIMESTAMP,
                    version      = user_profiles.version + 1
                """,
                (profile.user_id, profile_json),
            )
        else:
            c.execute(
                """
                UPDATE user_profiles
                SET profile_json = ?,
                    updated_at   = CURRENT_TIMESTAMP,
                    version      = ?
                WHERE user_id = ? AND version = ?
                """,
                (profile_json, new_ver, profile.user_id, expected_ver),
            )
            if c.rowcount == 0:
                conn.close()
                raise ProfileConflictError(
                    f"Profile for {profile.user_id!r} modified by another session "
                    f"(expected version {expected_ver}). Reload and retry."
                )

        if snapshot:
            c.execute(
                "INSERT INTO user_profile_snapshots (user_id, profile_json, snapshot_ts) "
                "VALUES (?, ?, ?)",
                (profile.user_id, profile_json, int(_time.time())),
            )

        conn.commit()
        conn.close()
        profile._db_version = new_ver

    def list_users(self):
        conn = sqlite3.connect(str(self.db_path))
        rows = conn.execute(
            "SELECT user_id FROM user_profiles ORDER BY updated_at DESC"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
