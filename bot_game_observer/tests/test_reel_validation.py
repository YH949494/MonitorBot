from __future__ import annotations

import hashlib
import json
import os

from src.multi_slot import MultiSlotEngine
from src.reel_validation import (
    aggregate_results,
    evaluate_thresholds,
    load_validation_samples,
    validate_sample,
)
from src.validation_fixture_generator import generate_missing_synthetic_images


def test_load_validation_samples():
    samples = load_validation_samples("validation_samples")
    ids = {(s.game_id, s.sample_id) for s in samples}
    assert ("candy_demo", "sample_001") in ids


def test_validate_sample_success_and_ingestion():
    generate_missing_synthetic_images("validation_samples")
    engine = MultiSlotEngine("config/slot_profiles")
    sample = [s for s in load_validation_samples("validation_samples") if s.game_id == "candy_demo"][0]
    result = validate_sample(sample, engine)
    assert result.ingestion_success is True
    assert result.total_cells == 15
    assert result.matched_cells == 15
    assert result.cell_accuracy == 1.0
    assert result.exact_grid_match is True
    assert result.unknown_count == 0


def test_validation_works_when_png_absent_for_synthetic(tmp_path):
    base = tmp_path / "validation_samples" / "candy_demo"
    base.mkdir(parents=True)
    data = {
        "game_id": "candy_demo",
        "sample_id": "sample_001",
        "image": "sample_001.png",
        "frame_index": 1,
        "reels_region": {"left": 10, "top": 10, "width": 150, "height": 90},
        "expected_grid": [["Wild", "Bonus", "A"], ["K", "Scatter", "Wild"], ["A", "K", "Bonus"], ["Scatter", "A", "K"], ["Wild", "A", "Scatter"]],
        "expected_symbols": {"wild_count": 3, "scatter_count": 2, "bonus_count": 3},
        "bet_amount": 1.0,
        "payout_amount": 0.5,
        "notes": "synthetic placeholder",
    }
    (base / "sample_001.json").write_text(json.dumps(data), encoding="utf-8")
    assert not (base / "sample_001.png").exists()
    generate_missing_synthetic_images(tmp_path / "validation_samples")
    assert (base / "sample_001.png").exists()


def test_generator_deterministic_and_no_overwrite(tmp_path):
    base = tmp_path / "validation_samples" / "candy_demo"
    base.mkdir(parents=True)
    data = {
        "game_id": "candy_demo",
        "sample_id": "sample_001",
        "image": "sample_001.png",
        "frame_index": 1,
        "reels_region": {"left": 10, "top": 10, "width": 150, "height": 90},
        "expected_grid": [["A", "A", "A"], ["A", "A", "A"], ["A", "A", "A"], ["A", "A", "A"], ["A", "A", "A"]],
        "expected_symbols": {"wild_count": 0, "scatter_count": 0, "bonus_count": 0},
        "bet_amount": 1.0,
        "payout_amount": 0.0,
        "notes": "synthetic",
    }
    (base / "sample_001.json").write_text(json.dumps(data), encoding="utf-8")
    generate_missing_synthetic_images(tmp_path / "validation_samples")
    p = base / "sample_001.png"
    h1 = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_bytes(b"manual")
    generate_missing_synthetic_images(tmp_path / "validation_samples")
    assert p.read_bytes() == b"manual"
    generate_missing_synthetic_images(tmp_path / "validation_samples", overwrite=True)
    h2 = hashlib.sha256(p.read_bytes()).hexdigest()
    assert h1 == h2


def test_aggregate_per_game_accuracy_and_unknown_accounting():
    generate_missing_synthetic_images("validation_samples")
    engine = MultiSlotEngine("config/slot_profiles")
    results = [validate_sample(s, engine) for s in load_validation_samples("validation_samples")]
    report = aggregate_results(results)
    assert "candy_demo" in report.per_game_accuracy
    assert report.total_unknown_cells >= 0
    assert report.average_cell_accuracy >= 0.0


def test_threshold_pass_fail_behavior():
    generate_missing_synthetic_images("validation_samples")
    engine = MultiSlotEngine("config/slot_profiles")
    report = aggregate_results([validate_sample(s, engine) for s in load_validation_samples("validation_samples")])
    os.environ["REEL_VALIDATION_MIN_CELL_ACCURACY"] = "0.0"
    os.environ["REEL_VALIDATION_MIN_EXACT_MATCH_RATE"] = "0.0"
    os.environ["REEL_VALIDATION_MAX_UNKNOWN_RATE"] = "1.0"
    ok, _ = evaluate_thresholds(report)
    assert ok is True
    os.environ["REEL_VALIDATION_MIN_CELL_ACCURACY"] = "1.1"
    ok, errs = evaluate_thresholds(report)
    assert ok is False
    assert errs


def test_missing_image_non_synthetic_fails(tmp_path):
    base = tmp_path / "validation_samples" / "egypt_demo"
    base.mkdir(parents=True)
    data = {
        "game_id": "egypt_demo",
        "sample_id": "sample_001",
        "image": "missing.png",
        "frame_index": 1,
        "reels_region": {"left": 10, "top": 10, "width": 150, "height": 90},
        "expected_grid": [["A", "K", "WILD"], ["SCATTER", "Q", "A"], ["J", "BONUS", "WILD"], ["A", "K", "Q"], ["SCATTER", "A", "J"]],
        "expected_symbols": {"wild_count": 2, "scatter_count": 2, "bonus_count": 1},
        "bet_amount": 1.0,
        "payout_amount": 0.5,
        "notes": "real screenshot required",
    }
    (base / "sample_001.json").write_text(json.dumps(data), encoding="utf-8")
    generate_missing_synthetic_images(tmp_path / "validation_samples")
    engine = MultiSlotEngine("config/slot_profiles")
    sample = load_validation_samples(tmp_path / "validation_samples")[0]
    result = validate_sample(sample, engine)
    assert result.parser_status == "failed"
    assert any("missing image" in e for e in result.errors)


def test_missing_image_or_malformed_label_returns_error_not_crash(tmp_path):
    malformed_dir = tmp_path / "validation_samples" / "candy_demo"
    malformed_dir.mkdir(parents=True)
    (malformed_dir / "bad.json").write_text("{bad json", encoding="utf-8")
    samples = load_validation_samples(tmp_path / "validation_samples")
    engine = MultiSlotEngine("config/slot_profiles")
    result = validate_sample(samples[0], engine)
    assert result.parser_status == "failed"
    assert result.errors
