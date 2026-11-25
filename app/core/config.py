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

# Trained model v1: YOLO11n-cls, Good vs Rotten (see reports/agrisort_good_rotten_v1/REPORT.md).
# AGRISORT_MODEL_PATH can override the weights file (used by tests).
MODEL_NAME = "agrisort_good_rotten_v1"
MODEL_WEIGHTS_PATH = Path(os.environ.get("AGRISORT_MODEL_PATH", MODELS_DIR / f"{MODEL_NAME}.pt"))
MODEL_REPORT_DIR = PROJECT_ROOT / "reports" / MODEL_NAME
MODEL_CLASSES = ["Good", "Rotten"]      # classes the current model can predict
PLANNED_CLASSES = ["Damaged"]           # planned for the final model
MODEL_IMGSZ = 224
# Single-image inference measured 3.6 ms on CPU vs 4.1 ms on MPS (see report), and CPU
# avoids GPU threading issues inside the Streamlit server, so the app uses the CPU.
MODEL_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Application metadata
# ---------------------------------------------------------------------------
APP_NAME = "AgriSort AI"
APP_TAGLINE = "AI-powered visual quality grading for apples"
APP_VERSION = "0.2.0 (prototype, model v1)"

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
    "Good": "Intact, firm-looking skin with no visible decay, bruising, cuts or cracks. Natural "
            "speckles, russeting and uneven colouring alone do not indicate damage or decay.",
    "Damaged": "Visible bruises, cuts, cracks or surface defects, but no decay.",
    "Rotten": "Signs of decay: soft brown patches, mould or fungal growth.",
}

# Upload constraints
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_UPLOAD_MB = 10
