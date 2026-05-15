from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from .models import Region
from .multi_slot import MultiSlotEngine
from .reel_parser import parse_frame_to_spin_grid


@dataclass
class ReelValidationSample:
    game_id: str
    sample_id: str
    image: str
    frame_index: int
    reels_region: Region
    expected_grid: list[list[str]]
    expected_symbols: dict[str, int]
    bet_amount: float
    payout_amount: float
    notes: str = ""
    sample_dir: Path = Path(".")


@dataclass
class ReelValidationResult:
    game_id: str
    sample_id: str
    parser_status: str
    expected_grid: list[list[str]]
    parsed_grid: list[list[str]]
    exact_grid_match: bool
    cell_accuracy: float
    matched_cells: int
    total_cells: int
    unknown_count: int
    min_confidence: float
    avg_confidence: float
    wild_count_match: bool
    scatter_count_match: bool
    bonus_count_match: bool
    ingestion_success: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class ReelValidationReport:
    total_samples: int
    passed_samples: int
    failed_samples: int
    average_cell_accuracy: float
    exact_grid_match_rate: float
    average_parser_confidence: float
    total_unknown_cells: int
    unknown_cell_rate: float
    per_game_accuracy: dict[str, float]
    failing_sample_ids: list[str]
    failure_reasons: dict[str, list[str]]


def load_validation_samples(samples_root: str | Path) -> list[ReelValidationSample]:
    root = Path(samples_root)
    samples: list[ReelValidationSample] = []
    for label_path in sorted(root.glob("*/*.json")):
        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
            region = Region(**data["reels_region"])
            samples.append(
                ReelValidationSample(
                    game_id=data["game_id"],
                    sample_id=data["sample_id"],
                    image=data["image"],
                    frame_index=int(data.get("frame_index", 0)),
                    reels_region=region,
                    expected_grid=data["expected_grid"],
                    expected_symbols=data["expected_symbols"],
                    bet_amount=float(data.get("bet_amount", 1.0)),
                    payout_amount=float(data.get("payout_amount", 0.0)),
                    notes=data.get("notes", ""),
                    sample_dir=label_path.parent,
                )
            )
        except Exception as exc:
            sid = label_path.stem
            game_id = label_path.parent.name
            samples.append(
                ReelValidationSample(
                    game_id=game_id,
                    sample_id=sid,
                    image="",
                    frame_index=0,
                    reels_region=Region(left=0, top=0, width=1, height=1),
                    expected_grid=[],
                    expected_symbols={},
                    bet_amount=0.0,
                    payout_amount=0.0,
                    notes=f"malformed label: {exc}",
                    sample_dir=label_path.parent,
                )
            )
    return samples


def _build_bootstrap_templates(sample: ReelValidationSample, frame: np.ndarray, tmpdir: Path) -> dict[str, str]:
    templates: dict[str, str] = {}
    crop = frame[sample.reels_region.top:sample.reels_region.top + sample.reels_region.height, sample.reels_region.left:sample.reels_region.left + sample.reels_region.width]
    if crop.size == 0 or not sample.expected_grid:
        return templates
    reel_count = len(sample.expected_grid)
    row_count = len(sample.expected_grid[0])
    for c, col in enumerate(sample.expected_grid):
        for r, sym in enumerate(col):
            if sym in templates:
                continue
            x0 = int(round((c * crop.shape[1]) / reel_count))
            x1 = int(round(((c + 1) * crop.shape[1]) / reel_count))
            y0 = int(round((r * crop.shape[0]) / row_count))
            y1 = int(round(((r + 1) * crop.shape[0]) / row_count))
            cell = crop[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            gray = cell[..., 0] if cell.ndim == 3 else cell
            out = tmpdir / f"{sample.game_id}_{sample.sample_id}_{sym}.npy"
            np.save(out, gray)
            templates[sym] = str(out)
    return templates


def _compare_grids(expected: list[list[str]], parsed: list[list[str]]) -> tuple[int, int, float, bool]:
    total = sum(len(col) for col in expected)
    matched = 0
    for c, col in enumerate(expected):
        for r, sym in enumerate(col):
            if c < len(parsed) and r < len(parsed[c]) and parsed[c][r] == sym:
                matched += 1
    accuracy = (matched / total) if total else 0.0
    return matched, total, accuracy, (matched == total and total > 0)


def validate_sample(sample: ReelValidationSample, engine: MultiSlotEngine) -> ReelValidationResult:
    errors: list[str] = []
    if not sample.image:
        return ReelValidationResult(sample.game_id, sample.sample_id, "failed", sample.expected_grid, [], False, 0.0, 0, 0, 0, 0.0, 0.0, False, False, False, False, [sample.notes or "missing_image_reference"])

    image_path = sample.sample_dir / sample.image
    if not image_path.exists():
        return ReelValidationResult(sample.game_id, sample.sample_id, "failed", sample.expected_grid, [], False, 0.0, 0, 0, 0, 0.0, 0.0, False, False, False, False, [f"missing image: {image_path}"])
    if sample.game_id not in engine.profiles:
        return ReelValidationResult(sample.game_id, sample.sample_id, "failed", sample.expected_grid, [], False, 0.0, 0, 0, 0, 0.0, 0.0, False, False, False, False, [f"unknown game_id: {sample.game_id}"])

    frame = np.array(Image.open(image_path).convert("RGB"))
    profile = engine.profiles[sample.game_id]
    with TemporaryDirectory(prefix="reel_validation_") as td:
        if not profile.symbol_templates:
            profile.symbol_templates = _build_bootstrap_templates(sample, frame, Path(td))

        parsed = parse_frame_to_spin_grid(frame, profile, SimpleNamespace(reels=sample.reels_region), frame_index=sample.frame_index)
        matched, total, accuracy, exact = _compare_grids(sample.expected_grid, parsed.grid)

        ingest_ok = False
        wild_match = False
        scatter_match = False
        bonus_match = False
        try:
            event = engine.ingest_spin(
                {
                    "game_id": sample.game_id,
                    "session_id": f"validation_{sample.sample_id}",
                    "bet_amount": sample.bet_amount,
                    "payout_amount": sample.payout_amount,
                    "grid": parsed.grid,
                    "confidence": parsed.avg_confidence,
                }
            )
            ingest_ok = True
            wild_match = int(event["wild_count"]) == int(sample.expected_symbols.get("wild_count", -1))
            scatter_match = int(event["scatter_count"]) == int(sample.expected_symbols.get("scatter_count", -1))
            bonus_match = int(event["bonus_count"]) == int(sample.expected_symbols.get("bonus_count", -1))
        except Exception as exc:
            errors.append(f"ingestion_error: {exc}")

        return ReelValidationResult(
            game_id=sample.game_id,
            sample_id=sample.sample_id,
            parser_status=parsed.parser_status,
            expected_grid=sample.expected_grid,
            parsed_grid=parsed.grid,
            exact_grid_match=exact,
            cell_accuracy=accuracy,
            matched_cells=matched,
            total_cells=total,
            unknown_count=parsed.unknown_count,
            min_confidence=parsed.min_confidence,
            avg_confidence=parsed.avg_confidence,
            wild_count_match=wild_match,
            scatter_count_match=scatter_match,
            bonus_count_match=bonus_match,
            ingestion_success=ingest_ok,
            errors=errors,
        )


def aggregate_results(results: list[ReelValidationResult]) -> ReelValidationReport:
    total = len(results)
    passed = [r for r in results if r.exact_grid_match and r.wild_count_match and r.scatter_count_match and r.bonus_count_match and r.ingestion_success and not r.errors]
    failed = [r for r in results if r not in passed]
    total_cells = sum(r.total_cells for r in results)
    total_unknown = sum(r.unknown_count for r in results)
    game_buckets: dict[str, list[ReelValidationResult]] = {}
    for result in results:
        game_buckets.setdefault(result.game_id, []).append(result)
    per_game = {gid: (sum(r.cell_accuracy for r in vals) / len(vals) if vals else 0.0) for gid, vals in game_buckets.items()}
    reasons: dict[str, list[str]] = {}
    for r in failed:
        items: list[str] = []
        if not r.exact_grid_match:
            items.append("grid_mismatch")
        if not r.wild_count_match:
            items.append("wild_count_mismatch")
        if not r.scatter_count_match:
            items.append("scatter_count_mismatch")
        if not r.bonus_count_match:
            items.append("bonus_count_mismatch")
        if not r.ingestion_success:
            items.append("ingestion_failed")
        items.extend(r.errors)
        reasons[f"{r.game_id}:{r.sample_id}"] = items

    return ReelValidationReport(
        total_samples=total,
        passed_samples=len(passed),
        failed_samples=len(failed),
        average_cell_accuracy=(sum(r.cell_accuracy for r in results) / total if total else 0.0),
        exact_grid_match_rate=(sum(1 for r in results if r.exact_grid_match) / total if total else 0.0),
        average_parser_confidence=(sum(r.avg_confidence for r in results) / total if total else 0.0),
        total_unknown_cells=total_unknown,
        unknown_cell_rate=((total_unknown / total_cells) if total_cells else 0.0),
        per_game_accuracy=per_game,
        failing_sample_ids=[f"{r.game_id}:{r.sample_id}" for r in failed],
        failure_reasons=reasons,
    )


def evaluate_thresholds(report: ReelValidationReport) -> tuple[bool, list[str]]:
    min_cell = float(os.getenv("REEL_VALIDATION_MIN_CELL_ACCURACY", "0.95"))
    min_exact = float(os.getenv("REEL_VALIDATION_MIN_EXACT_MATCH_RATE", "0.80"))
    max_unknown = float(os.getenv("REEL_VALIDATION_MAX_UNKNOWN_RATE", "0.05"))
    errors: list[str] = []
    if report.average_cell_accuracy < min_cell:
        errors.append(f"average_cell_accuracy {report.average_cell_accuracy:.4f} < {min_cell:.4f}")
    if report.exact_grid_match_rate < min_exact:
        errors.append(f"exact_grid_match_rate {report.exact_grid_match_rate:.4f} < {min_exact:.4f}")
    if report.unknown_cell_rate > max_unknown:
        errors.append(f"unknown_cell_rate {report.unknown_cell_rate:.4f} > {max_unknown:.4f}")
    return (len(errors) == 0), errors
