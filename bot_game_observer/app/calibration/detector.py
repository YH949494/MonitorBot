from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class DetectionResult:
    box: dict[str, int]
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "box": self.box,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "reason": self.reason,
        }


def _empty_result(reason: str) -> DetectionResult:
    return DetectionResult(
        box={"left": 0, "top": 0, "width": 0, "height": 0},
        confidence=0.0,
        reason=reason,
    )


def _box_from_contour(contour: np.ndarray) -> dict[str, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}


def detect_reels(frame: np.ndarray) -> dict[str, Any]:
    if frame is None or frame.size == 0:
        return _empty_result("invalid_frame").as_dict()
    if cv2 is None:
        return _empty_result("opencv_unavailable").as_dict()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = w // 2, h // 2
    best = None
    best_score = 0.0
    for contour in contours:
        box = _box_from_contour(contour)
        bw, bh = box["width"], box["height"]
        if bw < w * 0.2 or bh < h * 0.2:
            continue
        width_ratio = bw / w
        height_ratio = bh / h
        if not (0.35 <= width_ratio <= 0.85 and 0.35 <= height_ratio <= 0.85):
            continue
        bx = box["left"] + bw // 2
        by = box["top"] + bh // 2
        center_distance = np.hypot(bx - cx, by - cy)
        center_score = max(0.0, 1.0 - (center_distance / np.hypot(cx, cy)))
        area_score = min(1.0, (bw * bh) / (w * h * 0.65))
        score = 0.55 * center_score + 0.45 * area_score
        if score > best_score:
            best_score, best = score, box
    if best is None:
        return _empty_result("no_candidate_rect_found").as_dict()
    return DetectionResult(best, float(best_score), "largest_central_rect").as_dict()


def detect_spin_button(frame: np.ndarray) -> dict[str, Any]:
    if frame is None or frame.size == 0:
        return _empty_result("invalid_frame").as_dict()
    if cv2 is None:
        return _empty_result("opencv_unavailable").as_dict()
    h, w = frame.shape[:2]
    roi = frame[h // 2 :, w // 2 :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 30, param1=100, param2=22, minRadius=10, maxRadius=min(120, min(roi.shape[:2]) // 3))
    if circles is not None and len(circles) > 0:
        c = circles[0][0]
        x, y, r = int(c[0]), int(c[1]), int(c[2])
        box = {"left": w // 2 + x - r, "top": h // 2 + y - r, "width": 2 * r, "height": 2 * r}
        return DetectionResult(box, 0.82, "circle_in_bottom_right").as_dict()
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = bw * bh
        if area < 400:
            continue
        if area > best_area:
            best_area = area
            best = {"left": w // 2 + x, "top": h // 2 + y, "width": bw, "height": bh}
    if best is None:
        return _empty_result("no_bottom_right_candidate").as_dict()
    conf = min(0.74, max(0.45, best_area / (w * h * 0.10)))
    return DetectionResult(best, float(conf), "bottom_right_contour_fallback").as_dict()


def detect_popup_close(frame: np.ndarray) -> dict[str, Any]:
    if frame is None or frame.size == 0:
        return _empty_result("invalid_frame").as_dict()
    if cv2 is None:
        return _empty_result("opencv_unavailable").as_dict()
    h, w = frame.shape[:2]
    roi = frame[: h // 3, (2 * w) // 3 :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, bw, bh = cv2.boundingRect(contour)
        if 12 <= bw <= 120 and 12 <= bh <= 120:
            aspect = bw / max(bh, 1)
            if 0.7 <= aspect <= 1.3:
                box = {"left": (2 * w) // 3 + x, "top": y, "width": bw, "height": bh}
                return DetectionResult(box, 0.58, "top_right_small_square").as_dict()
    return _empty_result("no_popup_candidate").as_dict()


def detect_grid(reels_box: dict[str, int]) -> dict[str, Any]:
    bw, bh = reels_box.get("width", 0), reels_box.get("height", 0)
    if bw <= 0 or bh <= 0:
        return {"rows": None, "cols": None, "confidence": 0.0, "reason": "invalid_reels_box"}
    aspect = bw / bh
    candidates = [(5, 3, 1.67), (6, 4, 1.5), (5, 4, 1.25), (6, 5, 1.2)]
    best = min(candidates, key=lambda c: abs(aspect - c[2]))
    delta = abs(aspect - best[2])
    conf = max(0.2, min(0.88, 1.0 - delta))
    if conf < 0.45:
        return {"rows": None, "cols": None, "confidence": float(conf), "reason": "low_grid_confidence"}
    return {"rows": best[1], "cols": best[0], "confidence": float(conf), "reason": "aspect_ratio_match"}


def run_auto_detection(frame: np.ndarray) -> dict[str, Any]:
    reels = detect_reels(frame)
    spin = detect_spin_button(frame)
    popup = detect_popup_close(frame)
    grid = detect_grid(reels["box"])
    return {"reels": reels, "spin_button": spin, "popup_close": popup, "grid": grid}
