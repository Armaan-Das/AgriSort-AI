"""SQLite persistence for AgriSort AI.

SQLite stores the whole database in a single file (data/agrisort.db),
so there is no server to install. The table is created automatically the first
time the app runs, and older databases are upgraded in place (new columns are
added; existing rows are kept).
"""

from __future__ import annotations

import json
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
    predicted_class  TEXT,             -- NULL for records saved before the model existed
    confidence       REAL,
    model_version    TEXT,
    probabilities    TEXT,             -- JSON, e.g. {"Good": 0.93, "Rotten": 0.07}
    inference_ms     REAL
);
"""

# Columns added after the first release; added to older databases by init_db().
_ADDED_COLUMNS = {"probabilities": "TEXT", "inference_ms": "REAL"}


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the database and table if needed, and add any missing columns."""
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(analyses)")}
        for column, sql_type in _ADDED_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE analyses ADD COLUMN {column} {sql_type}")


def save_analysis(
    filename: str,
    stats: dict,
    predicted_class: str | None = None,
    confidence: float | None = None,
    model_version: str | None = None,
    probabilities: dict | None = None,
    inference_ms: float | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Insert one analysis record and return its new id."""
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            """
            INSERT INTO analyses (created_at, filename, width, height, brightness, contrast,
                                  sharpness, predicted_class, confidence, model_version,
                                  probabilities, inference_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(probabilities) if probabilities is not None else None,
                inference_ms,
            ),
        )
        return int(cur.lastrowid)


def save_prediction(filename: str, stats: dict, prediction: dict, db_path: Path = DB_PATH) -> int:
    """Save an image's measurements together with a genuine model prediction."""
    return save_analysis(
        filename, stats,
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        model_version=prediction["model_version"],
        probabilities=prediction["probabilities"],
        inference_ms=prediction["inference_ms"],
        db_path=db_path,
    )


def load_analyses(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Return all saved analyses, newest first, as a pandas DataFrame.

    Adds one column per class probability (e.g. prob_Good, prob_Rotten) parsed from JSON.
    """
    with closing(_connect(db_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM analyses ORDER BY id DESC", conn)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    if "probabilities" in df:
        parsed = df["probabilities"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})
        for cls in sorted({k for d in parsed for k in d}):
            df[f"prob_{cls}"] = parsed.apply(lambda d, c=cls: d.get(c))
    return df


def clear_analyses(db_path: Path = DB_PATH) -> None:
    """Delete every analysis record (the table itself is kept)."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM analyses")
