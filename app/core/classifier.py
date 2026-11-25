"""Apple quality classifier - placeholder.

The YOLO classification model has NOT been trained yet. This module only
reports whether trained weights exist. It never returns a fake prediction.

After training, `predict()` will load MODEL_WEIGHTS_PATH with Ultralytics
and return the real class + confidence.
"""

from __future__ import annotations

import numpy as np

from core.config import MODEL_WEIGHTS_PATH


def is_model_available() -> bool:
    """True only if a trained weights file exists on disk."""
    return MODEL_WEIGHTS_PATH.is_file()


def model_status() -> str:
    if is_model_available():
        return "Trained weights found (not yet connected to the app)"
    return "Not trained yet"


def predict(image: np.ndarray) -> dict:
    """Classify an apple image.

    Deliberately raises an error until a real model is trained and integrated,
    so the app can never display a made-up result.
    """
    raise NotImplementedError(
        "The AgriSort AI model has not been trained yet. "
        "Predictions will be available after training and integration."
    )
