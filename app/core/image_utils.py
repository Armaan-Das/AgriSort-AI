"""OpenCV image helpers.

These functions compute REAL, measurable properties of an uploaded image.
They do not classify quality - that is the job of the ML model (not yet trained).
"""

from __future__ import annotations

import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Turn uploaded file bytes into an OpenCV image (BGR channel order).

    Raises ValueError if the bytes are not a readable image.
    """
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file could not be read as an image.")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """OpenCV uses BGR order; browsers and Streamlit expect RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_image_stats(image: np.ndarray) -> dict:
    """Measure basic image-quality properties.

    - brightness: mean grey level (0 = black, 255 = white)
    - contrast:   standard deviation of grey levels
    - sharpness:  variance of the Laplacian; low values suggest a blurry photo
    """
    height, width = image.shape[:2]
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "width": int(width),
        "height": int(height),
        "brightness": round(float(grey.mean()), 2),
        "contrast": round(float(grey.std()), 2),
        "sharpness": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 2),
    }


def channel_means(image: np.ndarray) -> dict:
    """Average Red, Green and Blue intensity (0-255)."""
    b, g, r = (float(c) for c in cv2.mean(image)[:3])
    return {"Red": round(r, 1), "Green": round(g, 1), "Blue": round(b, 1)}


def edge_map(image: np.ndarray) -> np.ndarray:
    """Canny edge detection - highlights outlines, cuts and texture changes."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    return cv2.Canny(blurred, 50, 150)


def resize_for_display(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Shrink very large images so the page stays fast."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
