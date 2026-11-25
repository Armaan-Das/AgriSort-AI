"""Apple quality classifier - trained YOLO11n-cls model v1 (Good vs Rotten).

Training and evaluation: reports/agrisort_good_rotten_v1/REPORT.md
This module never fabricates a result: if the weights are missing, the input is
invalid or inference fails, it raises an error that the UI shows to the user.
"""

from __future__ import annotations

import threading
import time
from functools import lru_cache

import numpy as np

from core import config


class ModelUnavailableError(RuntimeError):
    """The model weights are missing or could not be loaded."""


class InferenceError(RuntimeError):
    """The model could not classify this input."""


_predict_lock = threading.Lock()


def is_model_available() -> bool:
    """True if the trained weights file exists on disk."""
    return config.MODEL_WEIGHTS_PATH.is_file()


def model_status() -> str:
    if is_model_available():
        return "Model v1 ready (Good / Rotten)"
    return "Model weights not found"


@lru_cache(maxsize=2)
def _load(path: str, mtime: float):
    """Load once per weights file (cache key includes modification time)."""
    from ultralytics import YOLO  # imported lazily: it is slow to import

    model = YOLO(path, task="classify")
    names = [model.names[i] for i in sorted(model.names)]
    if names != config.MODEL_CLASSES:
        raise ModelUnavailableError(f"Unexpected model classes {names}; expected {config.MODEL_CLASSES}.")
    # warm-up so the first real prediction is not slowed down by initialisation
    model.predict(np.zeros((config.MODEL_IMGSZ, config.MODEL_IMGSZ, 3), np.uint8),
                  device=config.MODEL_DEVICE, imgsz=config.MODEL_IMGSZ, verbose=False)
    return model


def load_model():
    path = config.MODEL_WEIGHTS_PATH
    if not path.is_file():
        raise ModelUnavailableError(f"Trained model weights not found at {path.name}.")
    try:
        return _load(str(path), path.stat().st_mtime)
    except ModelUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(f"The model could not be loaded ({type(exc).__name__}: {exc}).") from exc


def predict(image_bgr: np.ndarray) -> dict:
    """Classify one apple image (OpenCV BGR array, as returned by image_utils.decode_image).

    Returns predicted_class, confidence, probabilities (per class), inference_ms,
    model_version and device. Raises ModelUnavailableError or InferenceError.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 \
            or min(image_bgr.shape[:2]) < 8:
        raise InferenceError("Expected a colour image of at least 8x8 pixels.")

    model = load_model()
    try:
        with _predict_lock:
            start = time.perf_counter()
            result = model.predict(image_bgr, device=config.MODEL_DEVICE, imgsz=config.MODEL_IMGSZ,
                                   verbose=False)[0]
            inference_ms = (time.perf_counter() - start) * 1000
        probs = result.probs.data.cpu().numpy().astype(float)
    except Exception as exc:  # noqa: BLE001
        raise InferenceError(f"The model failed to classify this image ({type(exc).__name__}: {exc}).") from exc

    if probs.shape != (len(config.MODEL_CLASSES),) or not np.all(np.isfinite(probs)):
        raise InferenceError("The model returned an invalid output.")

    names = [model.names[i] for i in range(len(probs))]
    best = int(probs.argmax())
    return {
        "predicted_class": names[best],
        "confidence": float(probs[best]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "inference_ms": round(inference_ms, 2),
        "model_version": config.MODEL_NAME,
        "device": config.MODEL_DEVICE,
    }
