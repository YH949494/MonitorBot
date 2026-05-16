"""Pure helpers for reviewing and confirming calibration regions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.models import BotSettings, Region


EDITABLE_REGIONS = ("reels", "spin_button", "popup_close")
REQUIRED_REGIONS = ("reels", "spin_button")


@dataclass(frozen=True)
class DisplayTransform:
    scale: float
    offset_x: int
    offset_y: int
    display_width: int
    display_height: int


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    missing: tuple[str, ...]


def editable_region_names() -> tuple[str, str, str]:
    return EDITABLE_REGIONS


def fit_image_to_view(
    image_width: int,
    image_height: int,
    view_width: int,
    view_height: int,
) -> DisplayTransform:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if view_width <= 0 or view_height <= 0:
        raise ValueError("view dimensions must be positive")

    scale = min(view_width / image_width, view_height / image_height)
    display_width = max(1, int(round(image_width * scale)))
    display_height = max(1, int(round(image_height * scale)))
    return DisplayTransform(
        scale=scale,
        offset_x=max(0, int(round((view_width - display_width) / 2))),
        offset_y=max(0, int(round((view_height - display_height) / 2))),
        display_width=display_width,
        display_height=display_height,
    )


def region_to_display(region: Region, transform: DisplayTransform) -> Region:
    return Region(
        left=int(round(region.left * transform.scale)) + transform.offset_x,
        top=int(round(region.top * transform.scale)) + transform.offset_y,
        width=max(1, int(round(region.width * transform.scale))),
        height=max(1, int(round(region.height * transform.scale))),
    )


def region_from_display(
    region: Region,
    transform: DisplayTransform,
    image_width: int,
    image_height: int,
) -> Region:
    if transform.scale <= 0:
        raise ValueError("display transform scale must be positive")
    left = int(round((region.left - transform.offset_x) / transform.scale))
    top = int(round((region.top - transform.offset_y) / transform.scale))
    width = int(round(region.width / transform.scale))
    height = int(round(region.height / transform.scale))
    return Region(
        left=max(0, min(left, image_width - 1)),
        top=max(0, min(top, image_height - 1)),
        width=max(1, width),
        height=max(1, height),
    ).clip_to(image_width, image_height)


def apply_manual_override(settings: BotSettings, region_name: str, region: Region) -> BotSettings:
    if region_name not in EDITABLE_REGIONS:
        raise ValueError(f"region is not editable: {region_name}")
    updated = settings.model_copy(deep=True)
    setattr(updated.regions, region_name, region)
    updated.calibrated = False
    return updated


def validate_required_regions(settings: BotSettings) -> ValidationResult:
    missing = tuple(
        name for name in REQUIRED_REGIONS if getattr(settings.regions, name, None) is None
    )
    return ValidationResult(ok=not missing, missing=missing)


def confirm_calibration(settings: BotSettings) -> BotSettings:
    validation = validate_required_regions(settings)
    if not validation.ok:
        raise ValueError(
            "cannot confirm calibration; missing required regions: "
            + ", ".join(validation.missing)
        )
    updated = settings.model_copy(deep=True)
    updated.calibrated = True
    return updated


def resolve_grid_dimensions(
    settings: BotSettings,
    profiles_dir: str | Path,
    fallback: tuple[int, int] = (5, 3),
) -> tuple[int, int]:
    profile_path = Path(profiles_dir) / f"{settings.game_profile}.json"
    if not profile_path.is_file():
        return fallback
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        reels = int(data["reel_count"])
        rows = int(data["row_count"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback
    if reels <= 0 or rows <= 0:
        return fallback
    return reels, rows
