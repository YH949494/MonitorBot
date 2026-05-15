from __future__ import annotations

import json
import os
from pathlib import Path

from calibrate_reel_templates import main as calibrate_main
from src.multi_slot import MultiSlotEngine
from src.template_calibration import build_templates_for_game, load_calibration_labels, suggested_mapping_text


def test_load_calibration_labels():
    labels = load_calibration_labels("calibration")
    assert any(l.game_id == "candy_demo" for l in labels)


def test_missing_screenshot_returns_warning(tmp_path):
    base = tmp_path / "calibration" / "games" / "egypt_demo"
    (base / "labels").mkdir(parents=True)
    (base / "templates").mkdir(parents=True)
    data = {
        "game_id": "egypt_demo", "sample_id": "sample_001", "screenshot": "../screenshots/missing.png", "frame_index": 1,
        "reels_region": {"left": 10, "top": 10, "width": 150, "height": 90},
        "expected_grid": [["A", "K", "WILD"], ["SCATTER", "Q", "A"], ["J", "BONUS", "WILD"], ["A", "K", "Q"], ["SCATTER", "A", "J"]],
        "bet_amount": 1.0, "payout_amount": 0.5, "label_quality": "human_verified", "notes": "real screenshot label"
    }
    (base / "labels" / "sample_001.json").write_text(json.dumps(data), encoding="utf-8")
    labels = load_calibration_labels(tmp_path / "calibration")
    manifest, errs = build_templates_for_game(labels, MultiSlotEngine("config/slot_profiles"))
    assert manifest["game_id"] == "egypt_demo"
    assert any("missing screenshot" in e for e in errs)


def test_mapping_output_and_no_overwrite_behavior(tmp_path):
    calibrate_main()
    labels = load_calibration_labels("calibration")
    engine = MultiSlotEngine("config/slot_profiles")
    os.environ["CALIBRATION_OVERWRITE"] = "1"
    manifest, _ = build_templates_for_game(labels, engine)
    out = suggested_mapping_text("candy_demo", manifest)
    assert "symbol_templates" in out
    os.environ["CALIBRATION_OVERWRITE"] = "0"
    _, errs = build_templates_for_game(labels, engine)
    assert any("template exists" in e for e in errs)


def test_calibration_cli_clean_checkout_success():
    code = calibrate_main()
    assert code == 0


def test_malformed_label_returns_error(tmp_path, monkeypatch):
    bad = tmp_path / "calibration" / "games" / "candy_demo" / "labels"
    bad.mkdir(parents=True)
    (bad / "x.json").write_text("{", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    Path("config/slot_profiles").mkdir(parents=True)
    (tmp_path / "config/slot_profiles/candy_demo.json").write_text((Path(__file__).resolve().parents[1] / "config/slot_profiles/candy_demo.json").read_text(), encoding="utf-8")
    assert calibrate_main() == 2


def test_manifest_created_at_runtime_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("calibration/games/candy_demo/labels").mkdir(parents=True)
    Path("calibration/games/candy_demo/screenshots").mkdir(parents=True)
    Path("calibration/games/candy_demo/templates").mkdir(parents=True)
    src_label = Path(__file__).resolve().parents[1] / "calibration/games/candy_demo/labels/sample_001.json"
    (tmp_path / "calibration/games/candy_demo/labels/sample_001.json").write_text(src_label.read_text(), encoding="utf-8")
    Path("config/slot_profiles").mkdir(parents=True)
    (tmp_path / "config/slot_profiles/candy_demo.json").write_text((Path(__file__).resolve().parents[1] / "config/slot_profiles/candy_demo.json").read_text(), encoding="utf-8")
    manifest = Path("calibration/games/candy_demo/templates/manifest.json")
    assert not manifest.exists()
    assert calibrate_main() == 0
    assert manifest.exists()


def test_repeated_cli_run_without_overwrite_stays_zero(monkeypatch):
    monkeypatch.setenv("CALIBRATION_OVERWRITE", "0")
    assert calibrate_main() == 0
    assert calibrate_main() == 0


def test_gitignore_covers_generated_calibration_artifacts():
    text = Path('.gitignore').read_text(encoding='utf-8')
    assert 'calibration/games/*/screenshots/*' in text
    assert 'calibration/games/*/templates/*.npy' in text
    assert 'calibration/games/*/templates/manifest.json' in text
