"""Central configuration for AgriSort AI.

Every path, class name and colour lives here so the rest of the code
never hard-codes them. Change something once here and it updates everywhere.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file, so the app works from any directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
# The AGRISORT_DB_PATH environment variable can override this (used by tests,
# so they never touch the real database).
DB_PATH = Path(os.environ.get("AGRISORT_DB_PATH", DATA_DIR / "agrisort.db"))

# Where the trained YOLO classification weights will be saved after training.
# The file does NOT exist yet - the model has not been trained.
MODEL_WEIGHTS_PATH = MODELS_DIR / "agrisort_yolo_cls.pt"

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "AgriSort AI"
APP_TAGLINE = "AI-powered visual quality grading for apples"
APP_VERSION = "0.1.0 (prototype)"

# ---------------------------------------------------------------------------
# Quality classes
# ---------------------------------------------------------------------------
QUALITY_CLASSES = ["Good", "Damaged", "Rotten"]

CLASS_COLOURS = {
    "Good": "#2E7D32",     # green
    "Damaged": "#F9A825",  # amber
    "Rotten": "#C62828",   # red
}

CLASS_DESCRIPTIONS = {
    "Good": "Intact skin, uniform colour, no visible bruising or blemishes.",
    "Damaged": "Visible bruises, cuts, cracks or surface defects, but no decay.",
    "Rotten": "Signs of decay: soft brown patches, mould or fungal growth.",
}

# Upload constraints
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_MB = 10
