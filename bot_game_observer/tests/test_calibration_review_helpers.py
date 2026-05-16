import pytest
import numpy as np

from app.calibration import overlay
from app.calibration.overlay import draw_detection_overlay, draw_grid_lines
from app.calibration.review import apply_manual_override, clamp_box_to_image, finalize_profile_for_save, scale_box_to_display, scale_box_to_image


def _profile():
    return {
        "game_profile": "x",
        "confirmed": False,
        "calibrated": False,
        "capture": {"region": {"left": 0, "top": 0, "width": 1000, "height": 800}},
        "regions": {
            "reels": {"left": 100, "top": 100, "width": 300, "height": 300, "confidence": 0.5, "source": "auto"},
            "spin_button": {"left": 600, "top": 600, "width": 120, "height": 120, "confidence": 0.5, "source": "auto"},
            "popup_close": {"left": 900, "top": 10, "width": 40, "height": 40, "confidence": 0.5, "source": "auto"},
        },
        "grid": {"rows": 3, "cols": 5, "confidence": 0.8, "source": "auto"},
        "manual_required": True,
    }


@pytest.mark.skipif(overlay.cv2 is None, reason="opencv unavailable")
def test_draw_detection_overlay_does_not_mutate_input_image():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    original = img.copy()
    out = draw_detection_overlay(
        img,
        {"reels": {"box": {"left": 10, "top": 10, "width": 100, "height": 100}, "confidence": 0.9}},
        {"rows": 3, "cols": 5},
    )
    assert np.array_equal(img, original)
    assert out.shape == img.shape


@pytest.mark.skipif(overlay.cv2 is None, reason="opencv unavailable")
def test_draw_grid_lines_handles_valid_rows_cols():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_grid_lines(img, {"left": 10, "top": 10, "width": 150, "height": 90}, 3, 5)


@pytest.mark.skipif(overlay.cv2 is None, reason="opencv unavailable")
def test_draw_grid_lines_ignores_invalid_rows_cols_without_crashing():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_grid_lines(img, {"left": 10, "top": 10, "width": 150, "height": 90}, None, 5)
    draw_grid_lines(img, {"left": 10, "top": 10, "width": 150, "height": 90}, 0, 0)


def test_clamp_box_to_image_clamps_negative_and_oversized_boxes():
    box = clamp_box_to_image({"left": -20, "top": -30, "width": 500, "height": 500}, 200, 100)
    assert box["left"] == 0 and box["top"] == 0
    assert box["left"] + box["width"] <= 200
    assert box["top"] + box["height"] <= 100


def test_scale_box_roundtrip_approximately_preserves_box():
    box = {"left": 22, "top": 35, "width": 111, "height": 77}
    disp = scale_box_to_display(box, 1.5, 20, 30)
    back = scale_box_to_image(disp, 1.5, 20, 30)
    for k in box:
        assert abs(back[k] - box[k]) <= 1


def test_manual_override_sets_source_manual_and_confidence_1():
    profile = _profile()
    apply_manual_override(profile, "reels", {"left": 10, "top": 10, "width": 100, "height": 100})
    assert profile["regions"]["reels"]["source"] == "manual"
    assert profile["regions"]["reels"]["confidence"] == 1.0


def test_save_rules_keep_calibrated_false_when_not_confirmed():
    profile = _profile()
    ok, _ = finalize_profile_for_save(profile)
    assert ok is True
    assert profile["calibrated"] is False


def test_save_rules_allow_calibrated_true_when_confirmed_and_required_regions_valid():
    profile = _profile()
    profile["confirmed"] = True
    ok, _ = finalize_profile_for_save(profile)
    assert ok is True
    assert profile["calibrated"] is True
