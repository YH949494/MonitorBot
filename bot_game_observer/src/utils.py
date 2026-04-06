"""Misc helpers: paths, timing, safe math."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import TypeVar

from .app_paths import get_app_root, resource_path

T = TypeVar("T")


def project_root() -> Path:
    """
    Portable application root (same as :func:`get_app_root`).

    Kept for backward compatibility with earlier code that expected the repo root.
    """
    return get_app_root()


def resolve_asset(path_str: str) -> Path:
    """
    Resolve a path from config; absolute paths unchanged.

    Relative paths prefer the portable app tree, then bundled resources (PyInstaller).
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    cand = get_app_root() / p
    if cand.exists():
        return cand.resolve()
    alt = resource_path(path_str)
    if alt.exists():
        return alt.resolve()
    return cand.resolve()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def random_delay(min_sec: float, max_sec: float) -> float:
    """Sleep a random duration in [min_sec, max_sec]."""
    lo, hi = (min_sec, max_sec) if min_sec <= max_sec else (max_sec, min_sec)
    t = random.uniform(lo, hi)
    time.sleep(t)
    return t


def exponential_moving_average(prev: float, value: float, alpha: float) -> float:
    return alpha * value + (1.0 - alpha) * prev
