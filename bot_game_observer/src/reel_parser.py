from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import Region
from .multi_slot import GameProfile


def _crop_region(image: np.ndarray, region: Region) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, region.left)
    y1 = max(0, region.top)
    x2 = min(w, region.left + region.width)
    y2 = min(h, region.top + region.height)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return image[..., 0]

@dataclass
class ParsedReelGrid:
    game_id: str
    frame_index: int
    grid: list[list[str]]
    cell_confidences: list[list[float]]
    min_confidence: float
    avg_confidence: float
    unknown_count: int
    parser_status: str
    reason: str


def _split_reel_cells(reels_crop: np.ndarray, reel_count: int, row_count: int) -> list[list[np.ndarray]]:
    h, w = reels_crop.shape[:2]
    grid: list[list[np.ndarray]] = []
    for c in range(reel_count):
        x0 = int(round((c * w) / reel_count))
        x1 = int(round(((c + 1) * w) / reel_count))
        col: list[np.ndarray] = []
        for r in range(row_count):
            y0 = int(round((r * h) / row_count))
            y1 = int(round(((r + 1) * h) / row_count))
            col.append(reels_crop[y0:y1, x0:x1])
        grid.append(col)
    return grid


def _load_templates(profile: GameProfile) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for symbol, rel_path in (profile.symbol_templates or {}).items():
        p = Path(rel_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            continue
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
        else:
            arr = np.array(Image.open(p).convert("L"))
        if arr is not None and getattr(arr, "size", 0) > 0:
            loaded[symbol] = arr
    return loaded


def _match_symbol(cell_img: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    if not templates:
        return "UNKNOWN", 0.0
    gray = _to_gray(cell_img)
    best_sym = "UNKNOWN"
    best = 0.0
    for sym, tmpl in templates.items():
        resized = np.array(Image.fromarray(gray).resize((tmpl.shape[1], tmpl.shape[0])))
        err = float(np.mean(np.abs(resized.astype(np.float32) - tmpl.astype(np.float32))))
        score = max(0.0, 1.0 - (err / 255.0))
        if score > best:
            best, best_sym = score, sym
    return (best_sym if best > 0.7 else "UNKNOWN"), best


def parse_frame_to_spin_grid(
    frame: np.ndarray,
    game_profile: GameProfile,
    regions_or_settings: Any,
    *,
    frame_index: int = 0,
    low_confidence_threshold: float = 0.7,
) -> ParsedReelGrid:
    reels_region = regions_or_settings.reels if hasattr(regions_or_settings, "reels") else regions_or_settings["reels"]
    if isinstance(reels_region, dict):
        reels_region = Region(**reels_region)
    crop = _crop_region(frame, reels_region)
    if crop.size == 0:
        return ParsedReelGrid(game_id=game_profile.game_id, frame_index=frame_index, grid=[], cell_confidences=[], min_confidence=0.0, avg_confidence=0.0, unknown_count=0, parser_status="failed", reason="empty_reels_crop")

    templates = _load_templates(game_profile)
    cells = _split_reel_cells(crop, game_profile.reel_count, game_profile.row_count)
    grid: list[list[str]] = []
    confs: list[list[float]] = []
    unknown = 0
    for col_cells in cells:
        out_col: list[str] = []
        out_conf: list[float] = []
        for cell in col_cells:
            sym, conf = _match_symbol(cell, templates)
            if sym == "UNKNOWN":
                unknown += 1
            out_col.append(sym)
            out_conf.append(conf)
        grid.append(out_col)
        confs.append(out_conf)

    flat = [c for col in confs for c in col]
    min_conf = min(flat) if flat else 0.0
    avg_conf = float(sum(flat) / len(flat)) if flat else 0.0
    status = "ok"
    reason = "parsed"
    if not templates:
        status = "failed"
        reason = "no_templates_loaded"
    elif min_conf < low_confidence_threshold:
        status = "low_confidence"
        reason = "one_or_more_cells_below_threshold"

    return ParsedReelGrid(
        game_id=game_profile.game_id,
        frame_index=frame_index,
        grid=grid,
        cell_confidences=confs,
        min_confidence=min_conf,
        avg_confidence=avg_conf,
        unknown_count=unknown,
        parser_status=status,
        reason=reason,
    )
