from __future__ import annotations

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

PROFILES_DIR = Path("profiles")


def _within(box: dict[str, int], capture: dict[str, int]) -> bool:
    if box["width"] <= 0 or box["height"] <= 0:
        return False
    return (
        box["left"] >= 0
        and box["top"] >= 0
        and box["left"] + box["width"] <= capture["width"]
        and box["top"] + box["height"] <= capture["height"]
    )


def validate_profile(data: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    capture_region = data.get("capture", {}).get("region", {})
    for field in ("left", "top", "width", "height"):
        if field not in capture_region:
            errors.append(f"missing_capture_region_{field}")
    if capture_region.get("width", 0) <= 0 or capture_region.get("height", 0) <= 0:
        errors.append("invalid_capture_region_size")

    regions = data.get("regions", {})
    reels = regions.get("reels")
    spin = regions.get("spin_button")
    if not reels:
        errors.append("missing_reels")
    if not spin:
        errors.append("missing_spin_button")

    if reels and not _within(reels, capture_region):
        errors.append("reels_outside_capture")
    if spin and not _within(spin, capture_region):
        errors.append("spin_button_outside_capture")

    calibrated = bool(data.get("calibrated"))
    if calibrated and any(err in errors for err in ["missing_reels", "missing_spin_button", "reels_outside_capture", "spin_button_outside_capture"]):
        errors.append("calibrated_invalid_required_regions")

    return len(errors) == 0, errors


def save_profile(profile_name: str, data: dict[str, Any]) -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    data.setdefault("created_at", now)
    data["updated_at"] = now
    path = PROFILES_DIR / f"{profile_name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_profile(profile_name: str) -> dict[str, Any]:
    path = PROFILES_DIR / f"{profile_name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
