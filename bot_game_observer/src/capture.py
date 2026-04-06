"""High-speed region capture using mss."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
import cv2
import mss
import numpy as np
from numpy.typing import NDArray

from .models import FramePacket, Region


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CaptureService:
    """Captures a fixed screen rectangle at a target interval."""

    region: Region
    monitor_index: int = 1

    def __post_init__(self) -> None:
        self._sct = mss.mss()

    def bbox_dict(self) -> dict[str, int]:
        r = self.region
        return {
            "left": r.left,
            "top": r.top,
            "width": r.width,
            "height": r.height,
        }

    def grab_bgr(self) -> np.ndarray:
        """Capture region as BGR uint8."""
        shot = self._sct.grab(self.bbox_dict())
        arr = np.asarray(shot, dtype=np.uint8)
        # BGRA -> BGR
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr

    def grab_frame_packet(self, frame_index: int) -> FramePacket:
        ts = utcnow()
        bgr = self.grab_bgr()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return FramePacket(ts=ts, frame_index=frame_index, image_bgr=bgr, gray=gray)

    def sleep_for_fps(self, fps: float, loop_start: float) -> None:
        """Sleep to maintain approximate FPS from loop_start (time.monotonic())."""
        if fps <= 0:
            return
        period = 1.0 / fps
        elapsed = time.monotonic() - loop_start
        rem = period - elapsed
        if rem > 0:
            time.sleep(rem)


def numpy_bgr_to_gray(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
