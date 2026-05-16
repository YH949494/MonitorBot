from __future__ import annotations

from src.calibration_review.core import (
    apply_manual_override,
    confirm_calibration,
    editable_region_names,
    fit_image_to_view,
    region_from_display,
    region_to_display,
    resolve_grid_dimensions,
    validate_required_regions,
)
from src.models import BotSettings, CaptureConfig, CaptureMode, GameRegions, Region


def _settings() -> BotSettings:
    return BotSettings(
        game_profile="demo",
        calibrated=False,
        capture=CaptureConfig(
            mode=CaptureMode.REGION,
            region=Region(left=0, top=0, width=1000, height=800),
        ),
        regions=GameRegions(
            reels=Region(left=100, top=80, width=500, height=300),
            spin_button=Region(left=700, top=500, width=120, height=80),
            popup_close=None,
        ),
    )


def test_fit_image_to_view_preserves_aspect_ratio_and_centers() -> None:
    transform = fit_image_to_view(image_width=1920, image_height=1080, view_width=960, view_height=720)

    assert transform.scale == 0.5
    assert transform.offset_x == 0
    assert transform.offset_y == 90
    assert transform.display_width == 960
    assert transform.display_height == 540


def test_region_round_trip_between_image_and_display_coordinates() -> None:
    transform = fit_image_to_view(image_width=1000, image_height=500, view_width=500, view_height=500)
    original = Region(left=100, top=50, width=200, height=100)

    display = region_to_display(original, transform)
    restored = region_from_display(display, transform, image_width=1000, image_height=500)

    assert display == Region(left=50, top=150, width=100, height=50)
    assert restored == original


def test_manual_override_updates_only_named_editable_region() -> None:
    settings = _settings()
    updated = apply_manual_override(
        settings,
        "popup_close",
        Region(left=900, top=20, width=40, height=40),
    )

    assert editable_region_names() == ("reels", "spin_button", "popup_close")
    assert updated.regions.popup_close == Region(left=900, top=20, width=40, height=40)
    assert updated.regions.reels == settings.regions.reels
    assert updated.calibrated is False


def test_manual_override_rejects_non_editable_region() -> None:
    settings = _settings()

    try:
        apply_manual_override(settings, "win_banner", Region(left=1, top=1, width=10, height=10))
    except ValueError as exc:
        assert "not editable" in str(exc)
    else:
        raise AssertionError("manual override accepted a non-editable region")


def test_required_regions_validation_requires_reels_and_spin_button() -> None:
    settings = _settings()
    settings.regions.spin_button = None  # type: ignore[assignment]

    result = validate_required_regions(settings)

    assert result.ok is False
    assert result.missing == ("spin_button",)


def test_confirm_calibration_marks_calibrated_only_after_required_regions_exist() -> None:
    settings = _settings()

    confirmed = confirm_calibration(settings)

    assert settings.calibrated is False
    assert confirmed.calibrated is True


def test_confirm_calibration_refuses_missing_required_regions() -> None:
    settings = _settings()
    settings.regions.reels = None  # type: ignore[assignment]

    try:
        confirm_calibration(settings)
    except ValueError as exc:
        assert "reels" in str(exc)
    else:
        raise AssertionError("confirmation accepted a missing reels region")


def test_resolve_grid_dimensions_uses_matching_slot_profile(tmp_path) -> None:
    profiles = tmp_path / "slot_profiles"
    profiles.mkdir()
    (profiles / "wide_demo.json").write_text(
        """
        {
          "game_id": "wide_demo",
          "game_name": "Wide Demo",
          "provider": "DemoProvider",
          "reel_count": 6,
          "row_count": 5,
          "layout_type": "reels",
          "paylines_or_ways": "cluster",
          "symbol_mappings": {},
          "wild_symbols": [],
          "scatter_symbols": [],
          "bonus_symbols": [],
          "bonus_trigger_rule": {},
          "created_at": "2026-05-16T00:00:00Z",
          "updated_at": "2026-05-16T00:00:00Z"
        }
        """,
        encoding="utf-8",
    )
    settings = _settings()
    settings.game_profile = "wide_demo"

    assert resolve_grid_dimensions(settings, profiles) == (6, 5)


def test_resolve_grid_dimensions_falls_back_when_profile_missing(tmp_path) -> None:
    settings = _settings()
    settings.game_profile = "missing_demo"

    assert resolve_grid_dimensions(settings, tmp_path) == (5, 3)
