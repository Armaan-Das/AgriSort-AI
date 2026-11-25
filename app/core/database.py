"""SQLite persistence for AgriSort AI.

SQLite stores the whole database in a single file (data/agrisort.db),
so there is no server to install. The table is created automatically
the first time the app runs.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    width            INTEGER NOT NULL,
    height           INTEGER NOT NULL,
    brightness       REAL    NOT NULL,
    contrast         REAL    NOT NULL,
    sharpness        REAL    NOT NULL,
    predicted_class  TEXT,             -- NULL until a trained model exists
    confidence       REAL,             -- NULL until a trained model exists
    model_version    TEXT              -- NULL until a trained model exists
);
"""


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database file and table if they do not already exist."""
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def save_analysis(
    filename: str,
    stats: dict,
    predicted_class: str | None = None,
    confidence: float | None = None,
    model_version: str | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Insert one analysis record and return its new id."""
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (created_at, filename, width, height, brightness,
                                  contrast, sharpness, predicted_class, confidence,
                                  model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                filename,
                stats["width"],
                stats["height"],
                stats["brightness"],
                stats["contrast"],
                stats["sharpness"],
                predicted_class,
                confidence,
                model_version,
            ),
        )
        return int(cur.lastrowid)


def load_analyses(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all saved analyses, newest first, as a pandas DataFrame."""
    with closing(_connect(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM analyses ORDER BY id DESC", conn)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def clear_analyses(db_path: Path = DB_PATH) -> None:
    """Delete every analysis record (the table itself is kept)."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM analyses")
