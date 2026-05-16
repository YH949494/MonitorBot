from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

COLORS: dict[str, tuple[int, int, int]] = {
    "reels": (0, 255, 0),
    "spin_button": (0, 165, 255),
    "popup_close": (255, 0, 0),
}


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for overlay drawing")


def draw_grid_lines(image: np.ndarray, reels_box: dict[str, int], rows: int | None, cols: int | None) -> None:
    _require_cv2()
    if not reels_box or rows is None or cols is None or rows <= 0 or cols <= 0:
        return
    left = int(reels_box.get("left", 0))
    top = int(reels_box.get("top", 0))
    width = int(reels_box.get("width", 0))
    height = int(reels_box.get("height", 0))
    if width <= 0 or height <= 0:
        return
    for col in range(1, cols):
        x = left + int((width * col) / cols)
        cv2.line(image, (x, top), (x, top + height), (255, 255, 0), 1)
    for row in range(1, rows):
        y = top + int((height * row) / rows)
        cv2.line(image, (left, y), (left + width, y), (255, 255, 0), 1)


def draw_detection_overlay(frame: np.ndarray, detections: dict[str, Any], grid: dict[str, Any] | None = None) -> np.ndarray:
    _require_cv2()
    image = frame.copy()
    for name in ("reels", "spin_button", "popup_close"):
        detection = detections.get(name, {}) if detections else {}
        box = detection.get("box", {})
        if box.get("width", 0) <= 0 or box.get("height", 0) <= 0:
            continue
        color = COLORS.get(name, (200, 200, 200))
        p1 = (int(box.get("left", 0)), int(box.get("top", 0)))
        p2 = (p1[0] + int(box.get("width", 0)), p1[1] + int(box.get("height", 0)))
        cv2.rectangle(image, p1, p2, color, 2)
        conf = float(detection.get("confidence", 0.0))
        cv2.putText(image, f"{name}:{conf:.2f}", (p1[0], max(20, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    grid_data = grid or (detections.get("grid") if detections else None) or {}
    draw_grid_lines(image, detections.get("reels", {}).get("box", {}), grid_data.get("rows"), grid_data.get("cols"))
    return image
