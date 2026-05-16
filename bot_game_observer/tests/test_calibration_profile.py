import numpy as np

from app.calibration.detector import run_auto_detection
from app.calibration.profile import load_profile, save_profile, validate_profile


def _valid_profile():
    return {
        "game_profile": "test",
        "calibrated": True,
        "capture": {"region": {"left": 0, "top": 0, "width": 800, "height": 600}},
        "regions": {
            "reels": {"left": 100, "top": 100, "width": 400, "height": 300},
            "spin_button": {"left": 600, "top": 450, "width": 100, "height": 100},
        },
    }


def test_validate_profile_accepts_valid_profile():
    ok, errors = validate_profile(_valid_profile())
    assert ok is True
    assert errors == []


def test_validate_profile_rejects_missing_reels():
    profile = _valid_profile()
    del profile["regions"]["reels"]
    ok, errors = validate_profile(profile)
    assert ok is False
    assert "missing_reels" in errors


def test_validate_profile_rejects_spin_button_outside_capture_bounds():
    profile = _valid_profile()
    profile["regions"]["spin_button"]["left"] = 790
    ok, errors = validate_profile(profile)
    assert ok is False
    assert "spin_button_outside_capture" in errors


def test_calibrated_true_rejected_when_required_regions_invalid():
    profile = _valid_profile()
    profile["regions"]["spin_button"]["width"] = 0
    ok, errors = validate_profile(profile)
    assert ok is False
    assert "calibrated_invalid_required_regions" in errors


def test_detector_returns_structured_result_on_blank_image_without_crashing():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    result = run_auto_detection(frame)
    assert set(result.keys()) == {"reels", "spin_button", "popup_close", "grid"}
    for key in ("reels", "spin_button", "popup_close"):
        assert "box" in result[key]
        assert "confidence" in result[key]
        assert "reason" in result[key]


def test_detector_returns_confidence_values_between_zero_and_one():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    result = run_auto_detection(frame)
    for key in ("reels", "spin_button", "popup_close"):
        assert 0.0 <= result[key]["confidence"] <= 1.0
    assert 0.0 <= result["grid"]["confidence"] <= 1.0


def test_save_load_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    profile = _valid_profile()
    save_profile("roundtrip", profile)
    loaded = load_profile("roundtrip")
    assert loaded["game_profile"] == "test"
    assert loaded["regions"]["reels"]["width"] == 400
