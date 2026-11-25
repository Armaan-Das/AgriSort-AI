"""Shared paths and helpers for the FruitNet dataset pipeline.

Used by the scripts in scripts/ and by the standalone labelling tool in labelling/.
Nothing here is used by the AgriSort AI website.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# AGRISORT_DATA_DIR lets tests run the pipeline on a throwaway folder.
DATA_DIR = Path(os.environ.get("AGRISORT_DATA_DIR", PROJECT_ROOT / "data"))

# --- Source (FruitNet, Mendeley Data, CC BY 4.0) ---------------------------
FRUITNET_DOI = "10.17632/b6fftwbr2v.3"
FRUITNET_URL = "https://data.mendeley.com/datasets/b6fftwbr2v/3"
FRUITNET_ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/b6fftwbr2v/files/"
    "532e56bf-d855-4afd-86fd-2c8c425eb32c/file_downloaded"
)
FRUITNET_ZIP_SHA256 = "b84851e0c9101156bc537a7adfc4528289fa063d8a9a029ea4a6dd713cf5e53d"
FRUITNET_ZIP_SIZE = 3_253_463_240

ZIP_PATH = DATA_DIR / "downloads" / "FruitNetDataset.zip"
RAW_DIR = DATA_DIR / "raw" / "fruitnet"                 # original apple images
AUDIT_DIR = DATA_DIR / "interim" / "fruitnet_audit"     # audit outputs
IMAGES_CSV = AUDIT_DIR / "images.csv"                   # one row per raw image
LABELS_DIR = DATA_DIR / "labels"                        # human labels (keep in git)
MERGED_LABELS_CSV = LABELS_DIR / "fruitnet_labels_merged.csv"
PROCESSED_DIR = DATA_DIR / "processed" / "fruitnet_cls"  # YOLO classification layout

# --- Labels ------------------------------------------------------------------
SOURCE_QUALITIES = ["Good", "Bad", "Mixed"]       # FruitNet's original folders
TARGET_CLASSES = ["Good", "Damaged", "Rotten"]    # AgriSort AI classes
EXCLUDE = "Exclude"
ALL_LABELS = TARGET_CLASSES + [EXCLUDE]
NUM_BATCHES = 6

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_FIELDS = ["image_id", "label", "source_quality", "annotator", "labelled_at"]


# --- Helpers -----------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def perceptual_hash(image_bgr: np.ndarray) -> int:
    """64-bit DCT perceptual hash (pHash). Similar images -> small Hamming distance."""
    grey = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8]
    median = np.median(low.flatten()[1:])  # ignore the DC term
    bits = (low > median).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write to a temp file then rename, so a crash can never leave a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def label_file_for(batch: str) -> Path:
    """batch is 'all' or '1'..'6'."""
    name = "fruitnet_labels_all.csv" if batch == "all" else f"fruitnet_labels_batch{batch}.csv"
    return LABELS_DIR / name
