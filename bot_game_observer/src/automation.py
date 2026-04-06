"""Safe mouse automation: dry-run, jitter, rate limits."""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field

import pyautogui

from .logger import get_logger
from .models import Region

log = get_logger(__name__)

# Fail-safe: moving mouse to corner can abort - document in README
pyautogui.FAILSAFE = True


@dataclass
class ClickBudget:
    """Sliding window for max clicks per minute."""

    max_per_minute: int
    timestamps: deque[float] = field(default_factory=lambda: deque())

    def allow(self) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        return len(self.timestamps) < self.max_per_minute

    def record(self) -> None:
        self.timestamps.append(time.monotonic())


def jitter_point_in_region(region: Region, jitter_px: int) -> tuple[float, float]:
    """Random point inside region with optional jitter around center."""
    j = max(0, jitter_px)
    cx = region.left + region.width / 2.0
    cy = region.top + region.height / 2.0
    if j == 0:
        return cx, cy
    rx = random.uniform(-min(j, region.width / 2.1), min(j, region.width / 2.1))
    ry = random.uniform(-min(j, region.height / 2.1), min(j, region.height / 2.1))
    x = clamp(region.left + 2, region.left + region.width - 2, cx + rx)
    y = clamp(region.top + 2, region.top + region.height - 2, cy + ry)
    return float(x), float(y)


def clamp(lo: float, hi: float, v: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class SafeClickService:
    """Click only in configured region when live mode and budget allows."""

    spin_region_screen: Region
    dry_run: bool = True
    live: bool = False
    jitter_px: int = 6
    budget: ClickBudget | None = None

    def __post_init__(self) -> None:
        if self.budget is None:
            self.budget = ClickBudget(max_per_minute=40)

    def click_spin(self) -> tuple[float, float] | None:
        """
        Returns screen coordinates if a click was performed (or dry-run),
        or None if blocked.
        """
        if not self.live:
            x, y = jitter_point_in_region(self.spin_region_screen, self.jitter_px)
            log.info("[dry-run] would click at (%.1f, %.1f)", x, y)
            return x, y
        if not self.budget or not self.budget.allow():
            log.warning("Click rate limit reached; skipping click")
            return None
        x, y = jitter_point_in_region(self.spin_region_screen, self.jitter_px)
        pyautogui.moveTo(x, y, duration=random.uniform(0.05, 0.15))
        pyautogui.click()
        self.budget.record()
        log.info("Live click at (%.1f, %.1f)", x, y)
        return x, y
