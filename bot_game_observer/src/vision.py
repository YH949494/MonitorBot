"""Template matching, motion scoring, and region differencing."""

from __future__ import annotations

import os
import re
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
    if hasattr(cv2, "cvtColor") and hasattr(cv2, "COLOR_BGR2GRAY"):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image[..., 0]


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
    if float(np.std(tmpl_gray)) == 0.0:
        th, tw = tmpl_gray.shape[:2]
        target = int(tmpl_gray[0, 0])
        for y in range(0, scene_gray.shape[0] - th + 1):
            for x in range(0, scene_gray.shape[1] - tw + 1):
                patch = scene_gray[y : y + th, x : x + tw]
                if patch.shape == tmpl_gray.shape and int(patch[0, 0]) == target and np.all(patch == tmpl_gray):
                    return 1.0, (x, y)
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
        pytesseract.pytesseract.tesseract_cmd = r"C:\Users\user1\Downloads\MonitorBot-main (1)\MonitorBot-main\bot_game_observer\tesseract-main\tesseract-main\tesseract.exe"
        os.environ["TESSDATA_PREFIX"] = r"C:\Users\user1\Downloads\MonitorBot-main (1)\MonitorBot-main\bot_game_observer\tesseract-main\tesseract-main\tessdata"
    except ImportError:
        return ""
    crop = crop_region(image_bgr, region)
    if crop.size == 0:
        return ""
    gray = to_gray(crop)
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _th, processed = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        cfg = f"--psm 7 -l {lang}"
        return pytesseract.image_to_string(processed, config=cfg).strip()
    except Exception as exc:
        print(f"OCR failed: {exc}")
        return ""


def parse_numeric_amount(text: str, hint: str | None = None) -> tuple[float | None, float]:
    cleaned = text.strip()
    if not cleaned:
        return None, 0.0
    compact = cleaned.replace(" ", "")
    token_matches = list(re.finditer(r"\d[\d°'`.,-]*", compact))
    if not token_matches:
        return None, 0.0

    def _normalize_numeric_token(token: str) -> str | None:
        normalized = token.replace("°", "").replace("'", "").replace("`", "")
        normalized = re.sub(r"[^0-9.,-]", "", normalized)
        if not re.search(r"\d", normalized):
            return None
        seps = [i for i, ch in enumerate(normalized) if ch in ".,-"]
        if not seps:
            return normalized
        if len(seps) == 1:
            sep_idx = seps[0]
            sep_char = normalized[sep_idx]
            whole = re.sub(r"[^0-9]", "", normalized[:sep_idx])
            frac = re.sub(r"[^0-9]", "", normalized[sep_idx + 1 :])
            if not whole:
                return None
            if sep_char in ",." and len(frac) == 3:
                return f"{whole}{frac}"
            if frac:
                return f"{whole}.{frac}"
            return whole
        last_sep = seps[-1]
        whole = re.sub(r"[^0-9]", "", normalized[:last_sep])
        frac = re.sub(r"[^0-9]", "", normalized[last_sep + 1 :])
        if not whole:
            return None
        if frac:
            return f"{whole}.{frac}"
        return whole

    candidates: list[tuple[int, str, float]] = []
    for m in token_matches:
        raw_token = m.group(0)
        normalized = _normalize_numeric_token(raw_token)
        if not normalized:
            continue
        try:
            candidate_value = float(normalized)
        except ValueError:
            continue
        candidates.append((m.start(), normalized, candidate_value))
    if not candidates:
        return None, 0.0

    selected = candidates[-1]
    if hint:
        hint_key = hint.strip().lower()
        keywords_by_hint = {
            "credit": ["CREDIT", "BALANCE"],
            "balance": ["BALANCE", "CREDIT"],
            "bet": ["BET"],
            "win": ["WIN"],
        }
        keywords = keywords_by_hint.get(hint_key, [])
        if keywords:
            upper_text = compact.upper()
            keyword_positions: list[tuple[int, int]] = []
            for keyword in keywords:
                for m in re.finditer(re.escape(keyword), upper_text):
                    keyword_positions.append((m.start(), m.end()))
            if keyword_positions:
                ranked: list[tuple[int, int, int, float, int]] = []
                for idx, (start, normalized, candidate_value) in enumerate(candidates):
                    best_after = 0
                    best_distance = len(upper_text) + 1
                    for kw_start, kw_end in keyword_positions:
                        distance = abs(start - kw_end)
                        if distance < best_distance:
                            best_distance = distance
                        if start >= kw_end:
                            best_after = 1
                    digit_len = len(normalized.replace(".", ""))
                    ranked.append((best_after, -best_distance, digit_len, candidate_value, idx))
                selected = candidates[max(ranked)[4]]
    raw = selected[1]
    try:
        value = float(raw)
    except ValueError:
        return None, 0.0
    confidence = min(1.0, len(raw) / max(1, len(cleaned)))
    return value, confidence


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


def template_match_locations(
    scene_gray: np.ndarray,
    tmpl_gray: np.ndarray | None,
    threshold: float,
    *,
    max_matches: int = 12,
    iou_threshold: float = 0.3,
    center_merge_px: int = 0,
) -> tuple[bool, list[dict[str, int]], list[float]]:
    """Return deduplicated template matches above threshold as compact boxes/scores."""
    if tmpl_gray is None or tmpl_gray.size == 0 or scene_gray.size == 0:
        return False, [], []
    th, tw = tmpl_gray.shape[:2]
    sh, sw = scene_gray.shape[:2]
    if th > sh or tw > sw:
        return False, [], []

    res = cv2.matchTemplate(scene_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= float(threshold))
    if len(xs) == 0:
        return True, [], []

    candidates: list[tuple[float, int, int, int, int]] = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        score = float(res[y, x])
        candidates.append((score, int(x), int(y), int(tw), int(th)))
    candidates.sort(key=lambda c: c[0], reverse=True)

    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return 0.0 if union <= 0 else float(inter / union)

    picked_boxes: list[dict[str, int]] = []
    picked_scores: list[float] = []
    picked_rects: list[tuple[int, int, int, int]] = []
    picked_centers_x: list[float] = []
    for score, x, y, w, h in candidates:
        rect = (x, y, w, h)
        if any(_iou(rect, p) > iou_threshold for p in picked_rects):
            continue
        cx = x + (w / 2.0)
        if center_merge_px > 0 and any(abs(cx - existing) <= center_merge_px for existing in picked_centers_x):
            continue
        picked_rects.append(rect)
        picked_centers_x.append(cx)
        picked_boxes.append({"x": x, "y": y, "w": w, "h": h})
        picked_scores.append(round(score, 4))
        if len(picked_boxes) >= max_matches:
            break
    return True, picked_boxes, picked_scores
