from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .models import Region
from .multi_slot import MultiSlotEngine
from .reel_parser import parse_frame_to_spin_grid
from .validation_fixture_generator import generate_missing_synthetic_images


@dataclass
class CalibrationLabel:
    game_id: str
    sample_id: str
    screenshot: str
    frame_index: int
    reels_region: Region
    expected_grid: list[list[str]]
    bet_amount: float
    payout_amount: float
    label_quality: str
    notes: str
    label_path: Path


def load_calibration_labels(root: str | Path) -> list[CalibrationLabel]:
    labels: list[CalibrationLabel] = []
    for label_path in sorted(Path(root).glob("games/*/labels/*.json")):
        data = json.loads(label_path.read_text(encoding="utf-8"))
        labels.append(
            CalibrationLabel(
                game_id=data["game_id"],
                sample_id=data["sample_id"],
                screenshot=data["screenshot"],
                frame_index=int(data.get("frame_index", 0)),
                reels_region=Region(**data["reels_region"]),
                expected_grid=data["expected_grid"],
                bet_amount=float(data.get("bet_amount", 0.0)),
                payout_amount=float(data.get("payout_amount", 0.0)),
                label_quality=data.get("label_quality", ""),
                notes=data.get("notes", ""),
                label_path=label_path,
            )
        )
    return labels


def _split_cells(crop: np.ndarray, reel_count: int, row_count: int) -> list[list[np.ndarray]]:
    h, w = crop.shape[:2]
    out: list[list[np.ndarray]] = []
    for c in range(reel_count):
        x0 = int(round((c * w) / reel_count))
        x1 = int(round(((c + 1) * w) / reel_count))
        col: list[np.ndarray] = []
        for r in range(row_count):
            y0 = int(round((r * h) / row_count))
            y1 = int(round(((r + 1) * h) / row_count))
            col.append(crop[y0:y1, x0:x1])
        out.append(col)
    return out


def _resolve_screenshot(label: CalibrationLabel) -> Path:
    return (label.label_path.parent / label.screenshot).resolve()


def build_templates_for_game(labels: list[CalibrationLabel], engine: MultiSlotEngine, overwrite: bool = False) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not labels:
        return {}, errors
    game_id = labels[0].game_id
    profile = engine.profiles.get(game_id)
    if profile is None:
        return {}, [f"unknown game_id: {game_id}"]
    templates_dir = Path("calibration") / "games" / game_id / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    symbols: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    allow_overwrite = overwrite or os.getenv("CALIBRATION_OVERWRITE", "0") == "1"
    for label in labels:
        img_path = _resolve_screenshot(label)
        if not img_path.exists():
            errors.append(f"missing screenshot for {label.sample_id}: {img_path}")
            continue
        frame = np.array(Image.open(img_path).convert("RGB"))
        rr = label.reels_region
        crop = frame[rr.top:rr.top + rr.height, rr.left:rr.left + rr.width]
        if crop.size == 0:
            errors.append(f"empty reels_region for {label.sample_id}")
            continue
        cells = _split_cells(crop, profile.reel_count, profile.row_count)
        sources.append(label.sample_id)
        for c, col in enumerate(label.expected_grid):
            for r, sym in enumerate(col):
                cell = cells[c][r]
                if cell.size == 0:
                    continue
                gray = cell[..., 0] if cell.ndim == 3 else cell
                slot = symbols.setdefault(sym, {"count": 0, "template_files": []})
                idx = slot["count"] + 1
                out_name = f"{sym}_{idx:03d}.npy"
                out_path = templates_dir / out_name
                if out_path.exists() and not allow_overwrite:
                    errors.append(f"template exists (set CALIBRATION_OVERWRITE=1 to replace): {out_path}")
                else:
                    np.save(out_path, gray)
                slot["count"] += 1
                if out_name not in slot["template_files"]:
                    slot["template_files"].append(out_name)

    manifest = {
        "game_id": game_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_samples": sorted(set(sources)),
        "symbols": {k: {**v, "recommended_template": (v["template_files"][0] if v["template_files"] else "")} for k, v in symbols.items()},
    }
    (templates_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest, errors


def suggested_mapping_text(game_id: str, manifest: dict[str, Any]) -> str:
    mapping = {sym: f"calibration/games/{game_id}/templates/{spec['recommended_template']}" for sym, spec in manifest.get("symbols", {}).items() if spec.get("recommended_template")}
    return (
        f"Suggested symbol_templates for config/slot_profiles/{game_id}.json:\n\n"
        + json.dumps({"symbol_templates": mapping}, indent=2)
    )


def validate_calibration_labels(labels: list[CalibrationLabel], engine: MultiSlotEngine) -> dict[str, float]:
    total = 0
    matched = 0
    exact = 0
    unknown = 0
    sample_count = 0
    for label in labels:
        img = _resolve_screenshot(label)
        if not img.exists():
            continue
        sample_count += 1
        frame = np.array(Image.open(img).convert("RGB"))
        parsed = parse_frame_to_spin_grid(frame, engine.profiles[label.game_id], {"reels": label.reels_region}, frame_index=label.frame_index)
        this_total = sum(len(col) for col in label.expected_grid)
        this_matched = sum(1 for c, col in enumerate(label.expected_grid) for r, sym in enumerate(col) if c < len(parsed.grid) and r < len(parsed.grid[c]) and parsed.grid[c][r] == sym)
        total += this_total
        matched += this_matched
        unknown += parsed.unknown_count
        if this_total and this_total == this_matched:
            exact += 1
    return {
        "sample_count": float(sample_count),
        "cell_accuracy": (matched / total) if total else 0.0,
        "exact_match_rate": (exact / sample_count) if sample_count else 0.0,
        "unknown_rate": (unknown / total) if total else 0.0,
    }


def generate_calibration_synthetic_if_needed() -> list[Path]:
    return generate_missing_synthetic_images(Path("calibration") / "games")
