"""Streamlit app tests: every page renders, and the full workflow
upload -> genuine prediction -> save -> History -> Dashboard works.

All tests use a temporary database (see conftest.py).
"""

from pathlib import Path

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from core import config, database

ROOT = Path(__file__).resolve().parents[1]
MAIN = str(ROOT / "app" / "main.py")
PAGES = ["views/home.py", "views/analyse.py", "views/dashboard.py",
         "views/history.py", "views/model_info.py", "views/about.py"]
TEST_DIR = ROOT / "data" / "processed" / "fruitnet_cls" / "test"
needs_model = pytest.mark.skipif(not config.MODEL_WEIGHTS_PATH.is_file(), reason="trained weights not present")


def _open(page: str) -> AppTest:
    at = AppTest.from_file(MAIN, default_timeout=60)
    at.run()
    at.switch_page(page)
    at.run()
    return at


def _sample_png() -> bytes:
    img = np.full((240, 320, 3), 235, dtype=np.uint8)
    cv2.circle(img, (160, 120), 80, (40, 40, 190), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def _real_test_image(cls: str) -> tuple[str, bytes]:
    files = sorted((TEST_DIR / cls).glob("*"))
    if not files:
        pytest.skip("prepared test images not present")
    return files[0].name, files[0].read_bytes()


def _all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.info] + [m.value for m in at.warning] \
        + [m.value for m in at.error] + [m.value for m in at.success] + [m.value for m in at.caption]
    return "\n".join(str(p) for p in parts)


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


@needs_model
@pytest.mark.parametrize("cls", ["Good", "Rotten"])
def test_full_workflow_upload_predict_save_history_dashboard(cls):
    name, data = _real_test_image(cls)
    at = _open("views/analyse.py")
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception, at.exception
    text = _all_text(at)
    assert "PREDICTED QUALITY" in text
    assert "not a guarantee" in text and "food-safety" in text
    assert "Damaged" in text  # scope notice
    assert at.metric[0].label == "Inference time"

    save = [b for b in at.button if b.label == "Save inspection to history"][0]
    save.click().run()
    assert not at.exception, at.exception
    df = database.load_analyses()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["predicted_class"] in config.MODEL_CLASSES
    assert row["model_version"] == config.MODEL_NAME
    assert abs(row["prob_Good"] + row["prob_Rotten"] - 1) < 1e-4
    assert row["confidence"] == max(row["prob_Good"], row["prob_Rotten"])
    assert row["inference_ms"] > 0
    assert row["predicted_class"] in text  # the saved class is the one that was displayed

    # saving the same upload twice is blocked
    save = [b for b in at.button if b.label == "Save inspection to history"][0]
    assert save.disabled

    hist = _open("views/history.py")
    assert not hist.exception, hist.exception
    shown = hist.dataframe[0].value
    assert len(shown) == 1 and shown.iloc[0]["predicted_class"] == row["predicted_class"]

    dash = _open("views/dashboard.py")
    assert not dash.exception, dash.exception
    metrics = {m.label: m.value for m in dash.metric}
    assert metrics["Inspections saved"] == "1"
    assert metrics["Classified by the model"] == "1"
    assert metrics["Predicted Rotten"] == ("100%" if row["predicted_class"] == "Rotten" else "0%")


@needs_model
def test_synthetic_upload_gets_genuine_prediction():
    at = _open("views/analyse.py")
    at.file_uploader[0].upload("circle.png", _sample_png(), "image/png").run()
    assert not at.exception, at.exception
    assert "PREDICTED QUALITY" in _all_text(at)


def test_invalid_file_shows_error_and_no_prediction():
    at = _open("views/analyse.py")
    at.file_uploader[0].upload("fake.jpg", b"this is not an image", "image/jpeg").run()
    assert not at.exception, at.exception
    assert any("could not be read" in e.value for e in at.error)
    assert "PREDICTED QUALITY" not in _all_text(at)


def test_missing_weights_shows_error_and_disables_saving(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MODEL_WEIGHTS_PATH", tmp_path / "missing.pt")
    at = _open("views/analyse.py")
    at.file_uploader[0].upload("circle.png", _sample_png(), "image/png").run()
    assert not at.exception, at.exception
    assert any("model is unavailable" in e.value for e in at.error)
    assert "PREDICTED QUALITY" not in _all_text(at)
    save = [b for b in at.button if b.label == "Save inspection to history"][0]
    assert save.disabled
    assert database.load_analyses().empty


def test_dashboard_and_history_with_old_unclassified_rows():
    stats = {"width": 320, "height": 240, "brightness": 150.0, "contrast": 40.0, "sharpness": 500.0}
    database.save_analysis("old_record.png", stats)  # saved before the model existed
    for page in ("views/dashboard.py", "views/history.py"):
        at = _open(page)
        assert not at.exception, at.exception


def test_model_info_page_shows_genuine_report():
    at = _open("views/model_info.py")
    assert not at.exception, at.exception
    from core.model_info import load_report
    test = load_report()["test"]
    if test is None:
        pytest.skip("report not present")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Accuracy"] == f"{test['accuracy']:.1%}"
    assert metrics["Balanced accuracy"] == f"{test['balanced_accuracy']:.1%}"
