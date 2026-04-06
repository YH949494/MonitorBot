"""Load template images for OpenCV matching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import resolve_asset


def load_template_grayscale(path_str: str) -> np.ndarray | None:
    """Load image as single-channel uint8, or None if missing."""
    p = resolve_asset(path_str)
    if not p.is_file():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img


def load_template_bgr(path_str: str) -> np.ndarray | None:
    p = resolve_asset(path_str)
    if not p.is_file():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img
