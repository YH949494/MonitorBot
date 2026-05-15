from __future__ import annotations

import json
import os
from pathlib import Path

from src.multi_slot import MultiSlotEngine
from src.template_calibration import (
    build_templates_for_game,
    load_calibration_labels,
    suggested_mapping_text,
    validate_calibration_labels,
)
from src.validation_fixture_generator import generate_missing_synthetic_calibration_screenshots


def main() -> int:
    try:
        labels = load_calibration_labels("calibration")
    except Exception as exc:
        print(f"Malformed label data: {exc}")
        return 2
    engine = MultiSlotEngine("config/slot_profiles")
    generated = generate_missing_synthetic_calibration_screenshots("calibration")
    by_game = {}
    for label in labels:
        by_game.setdefault(label.game_id, []).append(label)

    any_real = False
    any_malformed = False
    any_threshold_fail = False
    min_cell = float(os.getenv("CALIBRATION_MIN_CELL_ACCURACY", "0.95"))
    min_exact = float(os.getenv("CALIBRATION_MIN_EXACT_MATCH_RATE", "0.80"))
    max_unknown = float(os.getenv("CALIBRATION_MAX_UNKNOWN_RATE", "0.05"))

    print(f"Loaded labels: {len(labels)}; generated synthetic screenshots: {len(generated)}")
    for game_id, game_labels in sorted(by_game.items()):
        try:
            manifest, errors = build_templates_for_game(game_labels, engine)
        except Exception as exc:
            print(f"[{game_id}] malformed label or calibration error: {exc}")
            any_malformed = True
            continue
        print(json.dumps(manifest, indent=2))
        print(suggested_mapping_text(game_id, manifest))
        profile = engine.profiles.get(game_id)
        if profile is not None:
            profile.symbol_templates = {sym: f"calibration/games/{game_id}/templates/{spec['recommended_template']}" for sym, spec in manifest.get("symbols", {}).items() if spec.get("recommended_template")}
        stats = validate_calibration_labels(game_labels, engine)
        print(f"[{game_id}] stats={json.dumps(stats)}")

        if stats["sample_count"] > 0:
            any_real = True
            failed = stats["cell_accuracy"] < min_cell or stats["exact_match_rate"] < min_exact or stats["unknown_rate"] > max_unknown
            if failed:
                any_threshold_fail = True
        for err in errors:
            print(f"[{game_id}] warning: {err}")

    if any_malformed:
        return 2
    if any_real and any_threshold_fail:
        return 1
    if not any_real:
        print("Warning: no calibration screenshots available yet; pipeline self-test only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
