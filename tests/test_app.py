"""Smoke tests: every page of the Streamlit app renders without exceptions.

All tests use a temporary database (see conftest.py).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import database

MAIN = str(Path(__file__).resolve().parents[1] / "app" / "main.py")
PAGES = ["views/home.py", "views/analyse.py", "views/dashboard.py",
         "views/history.py", "views/about.py"]


def _open(page: str) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=30)
    at.run()
    at.switch_page(page)
    at.run()
    return at


def _sample_png() -> bytes:
    img = np.full((240, 320, 3), 235, dtype=np.uint8)
    cv2.circle(img, (160, 120), 80, (40, 40, 190), -1)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(autouse=True)
def empty_db():
    database.init_db()
    database.clear_analyses()
    yield
    database.clear_analyses()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_with_empty_db(page):
    at = _open(page)
    assert not at.exception, at.exception


def test_upload_analyse_and_save():
    at = _open("views/analyse.py")
    at.file_uploader[0].upload("apple.png", _sample_png(), "image/png").run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4
    at.button[0].click().run()
    assert not at.exception, at.exception
    df = database.load_analyses()
    assert len(df) == 1 and df.iloc[0]["predicted_class"] is None


@pytest.mark.parametrize("page", ["views/dashboard.py", "views/history.py"])
def test_pages_render_with_data(page):
    stats = {"width": 320, "height": 240, "brightness": 150.0, "contrast": 40.0, "sharpness": 500.0}
    database.save_analysis("unclassified.png", stats)
    # Test-only record to exercise the class-distribution chart code path.
    database.save_analysis("test_only.png", stats, predicted_class="Good",
                           confidence=0.5, model_version="test")
    at = _open(page)
    assert not at.exception, at.exception
