"""End-to-end test of the dataset pipeline on a small FAKE FruitNet-style zip.

The fake images are random noise - they only exercise the code paths
(extraction, audit, labelling tool, merge, split). Nothing here touches data/.
"""

import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _img(seed: int, brighten: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    img = cv2.resize(small, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = cv2.add(img, np.full_like(img, brighten))
    return cv2.imencode(".jpg", img)[1].tobytes()


@pytest.fixture()
def data_dir(tmp_path):
    zpath = tmp_path / "downloads" / "FruitNetDataset.zip"
    zpath.parent.mkdir(parents=True)
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(40):
            z.writestr(f"FruitNet/Good Quality_Fruits/Apple_Good/IMG_{i}.jpg", _img(i))
        for i in range(40):
            z.writestr(f"FruitNet/Bad Quality_Fruits/Apple_Bad/IMG_{i}.jpg", _img(100 + i))
        for i in range(3):
            z.writestr(f"FruitNet/Mixed Quality_Fruits/Apple/IMG_{i}.jpg", _img(200 + i))
        z.writestr("FruitNet/Bad Quality_Fruits/Banana_Bad/IMG_0.jpg", _img(300))  # other fruit
        z.writestr("FruitNet/Bad Quality_Fruits/Apple_Bad/broken.jpg", b"\xff\xd8 not a jpeg")
        z.writestr("FruitNet/Bad Quality_Fruits/Apple_Bad/dup_of_good0.jpg", _img(0))  # exact dup
        z.writestr("FruitNet/Bad Quality_Fruits/Apple_Bad/near_bad1.jpg", _img(101, brighten=3))
    return tmp_path


def run(script, data_dir, *args):
    env = {**os.environ, "AGRISORT_DATA_DIR": str(data_dir)}
    res = subprocess.run([PY, str(ROOT / "scripts" / script), *args], env=env,
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def test_full_pipeline(data_dir, monkeypatch):
    out = run("fruitnet_extract.py", data_dir, "--skip-checksum")
    raw = data_dir / "raw" / "fruitnet"
    assert not list(raw.rglob("*Banana*")), "other fruits must not be extracted"
    assert len(list((raw / "Good").glob("*.jpg"))) == 40
    assert len(list((raw / "Mixed").glob("*.jpg"))) == 3
    assert "Extracted 86 apple images" in out

    run("fruitnet_audit.py", data_dir)
    summary = json.loads((data_dir / "interim" / "fruitnet_audit" / "summary.json").read_text())
    assert summary["unreadable_images"] == 1
    assert summary["exact_duplicate_sets"] == 1
    assert summary["cross_folder_conflict_pairs"] >= 1
    assert summary["near_duplicate_pairs"] >= 2
    assert sum(summary["batch_sizes"].values()) == 86

    # --- drive the labelling tool like a user would ---------------------
    monkeypatch.setenv("AGRISORT_DATA_DIR", str(data_dir))
    for mod in ("dataset_common",):
        sys.modules.pop(mod, None)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "labelling" / "label_tool.py"), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    at.button(key="btn_Damaged").click().run()          # no annotator yet -> refused
    assert not (data_dir / "labels" / "fruitnet_labels_all.csv").exists()
    at.text_input(key="annotator").input("T").run()
    for name in ["Damaged", "Rotten"] * 12 + ["Exclude"]:
        at.button(key=f"btn_{name}").click().run()
        assert not at.exception, at.exception
    at.radio(key="queue").set_value("Good").run()
    for _ in range(24):
        at.button(key="btn_Good").click().run()
    with open(data_dir / "labels" / "fruitnet_labels_all.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 49 and len({r["image_id"] for r in rows}) == 49

    # resume: a fresh session starts at an unlabelled image
    at2 = AppTest.from_file(str(ROOT / "labelling" / "label_tool.py"), default_timeout=30)
    at2.run()
    assert "Not labelled yet" in "".join(m.value for m in at2.markdown)

    # batch file + conflicting label is detected by merge
    batch_file = data_dir / "labels" / "fruitnet_labels_batch1.csv"
    first = rows[0]
    with open(batch_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(first.keys()))
        w.writeheader()
        w.writerow({**first, "label": "Good" if first["label"] != "Good" else "Rotten"})
    out = run("merge_labels.py", data_dir)
    assert "Conflicts        : 1" in out

    run("build_yolo_dataset.py", data_dir, "--balance", "--min-per-class", "5")
    proc = data_dir / "processed" / "fruitnet_cls"
    s = json.loads((proc / "summary.json").read_text())
    assert s["groups_split_across_sets"] == 0
    assert s["unverified_images"] == 0
    totals = s["totals_per_class"]
    assert max(totals.values()) - min(totals.values()) <= 1
    for split in ("train", "val", "test"):
        for c in ("Good", "Damaged", "Rotten"):
            assert any((proc / split / c).iterdir()), f"{split}/{c} empty"
