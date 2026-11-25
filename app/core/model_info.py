"""Read the genuine training / evaluation results of the current model.

All numbers shown on the Model Information page come from these files, which were
written by scripts/train_classifier.py and scripts/evaluate_classifier.py.
Nothing here is hard-coded; missing files are reported as missing.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from core.config import MODEL_REPORT_DIR

FILES = {
    "training": "training_summary.json",
    "test": "test_evaluation/metrics.json",
    "dataset": "dataset_summary.json",
    "verification": "dataset_verification.json",
    "epochs": "epoch_metrics.csv",
}

LIMITATIONS = [
    ("Small, clustered test set",
     "80 test images from only 17 independent groups; the 95% interval for balanced accuracy is wide."),
    ("Camera bias",
     "217 of the 362 Rotten images in the dataset come from one phone (ZUK Z2 Plus) that took no "
     "Good images."),
    ("Background and composition bias",
     "Rotten apples are mostly single, hand-held, against walls; 6 of the 10 missed rotten test images "
     "show two rotten apples on a plain surface."),
    ("Class imbalance (1 : 2.1)",
     "Handled with verified class-weighted loss; precision and recall still differ between classes."),
    ("Approximate grouping",
     "The same apple photographed more than 20 s apart from another angle cannot be detected, "
     "so some hidden leakage (optimistic results) is possible."),
    ("Single annotator", "No measure of agreement between annotators yet."),
    ("One dataset only",
     "FruitNet phone photos of Indian apple varieties; not yet tested on new photos, varieties or lighting."),
    ("Two classes only",
     "Damaged apples are forced into Good or Rotten; images that are not apples still receive a "
     "Good/Rotten answer; multi-apple images are classified as a whole."),
]


@lru_cache(maxsize=4)
def _read(report_dir: str, mtime_key: tuple) -> dict:
    base = Path(report_dir)
    out: dict = {"missing": []}
    for key, rel in FILES.items():
        path = base / rel
        if not path.is_file():
            out["missing"].append(rel)
            out[key] = None
        elif path.suffix == ".json":
            out[key] = json.loads(path.read_text())
        else:
            with open(path, newline="") as f:
                out[key] = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return out


def load_report(report_dir: Path = MODEL_REPORT_DIR) -> dict:
    mtimes = tuple((report_dir / rel).stat().st_mtime if (report_dir / rel).is_file() else 0
                   for rel in FILES.values())
    return _read(str(report_dir), mtimes)
