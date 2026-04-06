"""Template matching, motion scoring, and region differencing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .models import DetectorResult, DetectorKind, Region


def crop_region(image_bgr_or_gray: np.ndarray, region: Region) -> np.ndarray:
    """Crop ``region`` from image; assumes region fits."""
    h, w = image_bgr_or_gray.shape[:2]
    x1 = max(0, region.left)
    y1 = max(0, region.top)
    x2 = min(w, region.left + region.width)
    y2 = min(h, region.top + region.height)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    return image_bgr_or_gray[y1:y2, x1:x2]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def template_match_best(
    scene_gray: np.ndarray,
    tmpl_gray: np.ndarray,
) -> tuple[float, tuple[int, int]]:
    """
    Return (best_score, (x, y)) where score is normalized correlation max in [0,1].
    Uses TM_CCOEFF_NORMED.
    """
    if tmpl_gray.size == 0 or scene_gray.size == 0:
        return 0.0, (0, 0)
    if tmpl_gray.shape[0] > scene_gray.shape[0] or tmpl_gray.shape[1] > scene_gray.shape[1]:
        return 0.0, (0, 0)
    res = cv2.matchTemplate(scene_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    score = float(max(0.0, min(1.0, max_val)))
    return score, (int(max_loc[0]), int(max_loc[1]))


def motion_score(prev_gray: np.ndarray | None, curr_gray: np.ndarray) -> float:
    """
    Mean absolute difference between two same-sized grayscale crops.
    Returns 0 if prev is None or sizes mismatch.
    """
    if prev_gray is None:
        return 0.0
    if prev_gray.shape != curr_gray.shape:
        return 0.0
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(np.mean(diff))


def rolling_motion_score(
    history: list[np.ndarray],
    curr_gray: np.ndarray,
    history_frames: int,
) -> float:
    """Average motion vs last N-1 frames in history (including curr appended)."""
    if not history:
        return 0.0
    scores: list[float] = []
    for prev in history[-(history_frames - 1) :]:
        scores.append(motion_score(prev, curr_gray))
    if not scores:
        return motion_score(history[-1], curr_gray) if history else 0.0
    return float(np.mean(scores))


def region_mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(a, b)))


def ocr_region_text(image_bgr: np.ndarray, region: Region, lang: str = "eng") -> str:
    """Optional OCR via pytesseract; returns empty string if unavailable."""
    try:
        import pytesseract
    except ImportError:
        return ""
    crop = crop_region(image_bgr, region)
    if crop.size == 0:
        return ""
    try:
        cfg = f"--psm 7 -l {lang}"
        return pytesseract.image_to_string(crop, config=cfg).strip()
    except Exception:
        return ""


def detector_from_template(
    name: str,
    scene_gray: np.ndarray,
    tmpl_gray: np.ndarray | None,
    threshold: float,
) -> DetectorResult:
    if tmpl_gray is None:
        return DetectorResult(
            name=name,
            kind=DetectorKind.TEMPLATE,
            active=False,
            confidence=0.0,
            detail={"reason": "missing_template"},
        )
    score, loc = template_match_best(scene_gray, tmpl_gray)
    active = score >= threshold
    return DetectorResult(
        name=name,
        kind=DetectorKind.TEMPLATE,
        active=active,
        confidence=score,
        detail={"loc": loc, "threshold": threshold},
    )
