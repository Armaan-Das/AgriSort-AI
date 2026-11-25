"""Tests for the core logic: image utilities, database and the trained classifier."""

import csv
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from core import classifier, config, database, image_utils

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "fruitnet_cls"
PREDICTIONS_CSV = config.MODEL_REPORT_DIR / "test_evaluation" / "predictions.csv"
needs_model = pytest.mark.skipif(not config.MODEL_WEIGHTS_PATH.is_file(), reason="trained weights not present")


def _sample_png() -> bytes:
    """Synthetic test image (a red circle) - only used to test image plumbing."""
    img = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.circle(img, (80, 60), 40, (30, 30, 200), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_and_stats():
    image = image_utils.decode_image(_sample_png())
    stats = image_utils.compute_image_stats(image)
    assert (stats["width"], stats["height"]) == (160, 120)
    assert 0 <= stats["brightness"] <= 255
    assert stats["sharpness"] > 0


def test_decode_rejects_non_image():
    with pytest.raises(ValueError):
        image_utils.decode_image(b"not an image")


def test_database_roundtrip(tmp_path):
    db = tmp_path / "test.db"
    database.init_db(db)
    stats = {"width": 10, "height": 20, "brightness": 1.0, "contrast": 2.0, "sharpness": 3.0}
    database.save_analysis("apple.png", stats, db_path=db)
    pred = {"predicted_class": "Rotten", "confidence": 0.9, "model_version": "v-test",
            "probabilities": {"Good": 0.1, "Rotten": 0.9}, "inference_ms": 4.2}
    database.save_prediction("rotten.png", stats, pred, db_path=db)
    df = database.load_analyses(db)
    assert len(df) == 2
    newest = df.iloc[0]
    assert newest["predicted_class"] == "Rotten" and newest["prob_Rotten"] == 0.9
    assert newest["inference_ms"] == 4.2
    assert pd.isna(df.iloc[1]["predicted_class"])
    database.clear_analyses(db)
    assert database.load_analyses(db).empty


def test_database_upgrade_keeps_old_rows(tmp_path):
    """A database created by version 0.1 (no probabilities / inference_ms) is upgraded in place."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            filename TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, brightness REAL NOT NULL,
            contrast REAL NOT NULL, sharpness REAL NOT NULL, predicted_class TEXT, confidence REAL,
            model_version TEXT)""")
        conn.execute("INSERT INTO analyses (created_at, filename, width, height, brightness, contrast, sharpness)"
                     " VALUES ('2026-11-25T15:00:00', 'old.jpg', 1, 1, 1, 1, 1)")
    database.init_db(db)
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(analyses)")}
    assert {"probabilities", "inference_ms"} <= cols
    df = database.load_analyses(db)
    assert len(df) == 1 and df.iloc[0]["filename"] == "old.jpg"


def test_missing_weights_raise_and_never_fake(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_WEIGHTS_PATH", tmp_path / "missing.pt")
    assert not classifier.is_model_available()
    with pytest.raises(classifier.ModelUnavailableError):
        classifier.predict(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize("bad", [None, np.zeros((64, 64), np.uint8), np.zeros((4, 4, 3), np.uint8)])
def test_invalid_input_raises(bad):
    with pytest.raises(classifier.InferenceError):
        classifier.predict(bad)


@needs_model
def test_real_prediction_is_valid():
    img = image_utils.decode_image(_sample_png())
    r = classifier.predict(img)
    assert r["predicted_class"] in config.MODEL_CLASSES
    assert set(r["probabilities"]) == set(config.MODEL_CLASSES)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-4
    assert r["confidence"] == max(r["probabilities"].values())
    assert r["inference_ms"] > 0 and r["model_version"] == config.MODEL_NAME


@needs_model
def test_model_is_cached():
    assert classifier.load_model() is classifier.load_model()


@needs_model
@pytest.mark.skipif(not (PREDICTIONS_CSV.is_file() and PROCESSED.is_dir()), reason="evaluation files not present")
def test_app_predictions_match_offline_evaluation():
    """The website's decode + predict path must reproduce the recorded test-set predictions."""
    rows = list(csv.DictReader(open(PREDICTIONS_CSV)))
    for row in rows:
        img = image_utils.decode_image((PROCESSED / row["path"]).read_bytes())
        r = classifier.predict(img)
        assert r["predicted_class"] == row["pred"], row["path"]
        assert abs(r["confidence"] - float(row["confidence"])) < 1e-3, row["path"]
