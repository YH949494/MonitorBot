from __future__ import annotations

import json
from pathlib import Path

from src.multi_slot import MultiSlotEngine
from src.reel_validation import aggregate_results, evaluate_thresholds, load_validation_samples, validate_sample
from src.validation_fixture_generator import generate_missing_synthetic_images


def main() -> int:
    root = Path("validation_samples")
    generate_missing_synthetic_images(root)
    engine = MultiSlotEngine("config/slot_profiles")
    samples = load_validation_samples(root)
    results = [validate_sample(sample, engine) for sample in samples]
    report = aggregate_results(results)

    print("=== Reel Validation Report ===")
    print(json.dumps({
        "total_samples": report.total_samples,
        "passed_samples": report.passed_samples,
        "failed_samples": report.failed_samples,
        "average_cell_accuracy": report.average_cell_accuracy,
        "exact_grid_match_rate": report.exact_grid_match_rate,
        "average_parser_confidence": report.average_parser_confidence,
        "total_unknown_cells": report.total_unknown_cells,
        "unknown_cell_rate": report.unknown_cell_rate,
        "per_game_accuracy": report.per_game_accuracy,
        "failing_sample_ids": report.failing_sample_ids,
        "failure_reasons": report.failure_reasons,
    }, indent=2))

    ok, errors = evaluate_thresholds(report)
    if not ok:
        print("Threshold check failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Threshold check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
