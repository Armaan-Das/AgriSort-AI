"""Basic tests for the core logic (no ML involved)."""

import cv2
import numpy as np
import pytest

from core import classifier, database, image_utils


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
    df = database.load_analyses(db)
    assert len(df) == 1
    assert df.iloc[0]["filename"] == "apple.png"
    assert df.iloc[0]["predicted_class"] is None
    database.clear_analyses(db)
    assert database.load_analyses(db).empty


def test_classifier_never_fakes_predictions():
    with pytest.raises(NotImplementedError):
        classifier.predict(np.zeros((10, 10, 3), dtype=np.uint8))
