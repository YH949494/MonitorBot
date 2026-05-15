from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _symbol_value(symbol: str) -> int:
    return 30 + (sum(symbol.encode("utf-8")) % 200)


def _is_synthetic(notes: str) -> bool:
    lowered = (notes or "").lower()
    return "synthetic" in lowered or "placeholder" in lowered


def generate_missing_synthetic_images(samples_root: str | Path, *, overwrite: bool = False) -> list[Path]:
    root = Path(samples_root)
    generated: list[Path] = []
    for label_path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not _is_synthetic(data.get("notes", "")):
            continue
        image_name = data.get("image")
        if not image_name:
            continue
        image_path = label_path.parent / image_name
        if image_path.exists() and not overwrite:
            continue

        region = data.get("reels_region", {})
        expected_grid = data.get("expected_grid", [])
        reel_count = len(expected_grid)
        row_count = len(expected_grid[0]) if reel_count else 0
        if reel_count == 0 or row_count == 0:
            continue

        left = int(region.get("left", 0))
        top = int(region.get("top", 0))
        width = int(region.get("width", 0))
        height = int(region.get("height", 0))
        canvas_h = max(top + height + 10, 64)
        canvas_w = max(left + width + 10, 64)
        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for c, col in enumerate(expected_grid):
            x0 = left + int(round((c * width) / reel_count))
            x1 = left + int(round(((c + 1) * width) / reel_count))
            for r, sym in enumerate(col):
                y0 = top + int(round((r * height) / row_count))
                y1 = top + int(round(((r + 1) * height) / row_count))
                v = _symbol_value(str(sym))
                frame[y0:y1, x0:x1] = (v, v, v)

        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(image_path)
        generated.append(image_path)

    return generated
