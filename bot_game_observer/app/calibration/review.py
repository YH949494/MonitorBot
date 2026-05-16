from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .detector import run_auto_detection
from .overlay import draw_detection_overlay
from .profile import save_profile, validate_profile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calibration.review")


def clamp_box_to_image(box: dict[str, int], image_width: int, image_height: int) -> dict[str, int]:
    left = max(0, min(int(box.get("left", 0)), image_width - 1))
    top = max(0, min(int(box.get("top", 0)), image_height - 1))
    right = max(left + 1, min(int(box.get("left", 0)) + int(box.get("width", 0)), image_width))
    bottom = max(top + 1, min(int(box.get("top", 0)) + int(box.get("height", 0)), image_height))
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def scale_box_to_display(box: dict[str, int], scale: float, offset_x: int, offset_y: int) -> dict[str, int]:
    return {
        "left": int(round(box["left"] * scale + offset_x)),
        "top": int(round(box["top"] * scale + offset_y)),
        "width": int(round(box["width"] * scale)),
        "height": int(round(box["height"] * scale)),
    }


def scale_box_to_image(display_rect: dict[str, int], scale: float, offset_x: int, offset_y: int) -> dict[str, int]:
    inv = 1.0 / scale if scale > 0 else 1.0
    return {
        "left": int(round((display_rect["left"] - offset_x) * inv)),
        "top": int(round((display_rect["top"] - offset_y) * inv)),
        "width": int(round(display_rect["width"] * inv)),
        "height": int(round(display_rect["height"] * inv)),
    }


def apply_manual_override(profile: dict[str, Any], region: str, box: dict[str, int]) -> None:
    profile.setdefault("regions", {})
    profile["regions"][region] = {**box, "confidence": 1.0, "source": "manual"}
    capture = profile.get("capture", {}).get("region", {})
    reels = profile["regions"].get("reels", {})
    spin = profile["regions"].get("spin_button", {})
    required_valid = (
        reels.get("width", 0) > 0 and reels.get("height", 0) > 0 and spin.get("width", 0) > 0 and spin.get("height", 0) > 0
        and reels.get("left", 0) + reels.get("width", 0) <= capture.get("width", 0)
        and reels.get("top", 0) + reels.get("height", 0) <= capture.get("height", 0)
        and spin.get("left", 0) + spin.get("width", 0) <= capture.get("width", 0)
        and spin.get("top", 0) + spin.get("height", 0) <= capture.get("height", 0)
    )
    if required_valid:
        profile["manual_required"] = False


def finalize_profile_for_save(profile: dict[str, Any]) -> tuple[bool, list[str]]:
    ok, errors = validate_profile(profile)
    confirmed = bool(profile.get("confirmed", False))
    profile["calibrated"] = bool(ok and confirmed)
    return ok, errors


def _build_profile(frame: np.ndarray, profile_name: str, detected: dict[str, Any]) -> dict[str, Any]:
    h, w = frame.shape[:2]
    return {
        "game_profile": profile_name,
        "calibrated": False,
        "confirmed": False,
        "confirmed_at": None,
        "capture": {
            "mode": "window_or_region",
            "window_title_contains": "",
            "region": {"left": 0, "top": 0, "width": w, "height": h},
            "fps": 8.0,
            "coordinate_mode": "relative_to_capture",
        },
        "regions": {
            "reels": {**detected["reels"]["box"], "confidence": detected["reels"]["confidence"], "source": "auto"},
            "spin_button": {**detected["spin_button"]["box"], "confidence": detected["spin_button"]["confidence"], "source": "auto"},
            "popup_close": {**detected["popup_close"]["box"], "confidence": detected["popup_close"]["confidence"], "source": "auto"},
        },
        "grid": {"rows": detected["grid"]["rows"], "cols": detected["grid"]["cols"], "confidence": detected["grid"]["confidence"], "source": "auto"},
        "manual_required": any(detected[k]["confidence"] < 0.45 for k in ("reels", "spin_button")),
    }


def main() -> int:
    logger.info("[CALIBRATION][REVIEW_START]")
    if cv2 is None:
        print("[CALIBRATION][REVIEW_UNAVAILABLE] PySide6 is required for interactive review UI")
        return 1
    try:
        from PySide6.QtCore import QPoint, QRect, Qt
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QRubberBand, QVBoxLayout, QWidget
    except Exception as exc:
        logger.error(f"[CALIBRATION][REVIEW_UNAVAILABLE] reason={exc}")
        print("[CALIBRATION][REVIEW_UNAVAILABLE] PySide6 is required for interactive review UI")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile-name", default="auto_detected")
    parser.add_argument("--profile", type=str)
    args = parser.parse_args()

    frame = cv2.imread(args.source)
    if frame is None:
        logger.error(f"[CALIBRATION][SOURCE_LOAD_FAILED] path={args.source}")
        return 1

    detected = run_auto_detection(frame)
    profile = _build_profile(frame, args.profile_name, detected)

    if args.profile:
        from .profile import load_profile

        p = Path(args.profile)
        profile = load_profile(p.stem)

    class Canvas(QLabel):
        def __init__(self, wnd: "Window") -> None:
            super().__init__()
            self.wnd = wnd
            self.rubber = QRubberBand(QRubberBand.Rectangle, self)
            self.origin = QPoint()

        def mousePressEvent(self, event):
            self.origin = event.position().toPoint()
            self.rubber.setGeometry(QRect(self.origin, self.origin))
            self.rubber.show()

        def mouseMoveEvent(self, event):
            self.rubber.setGeometry(QRect(self.origin, event.position().toPoint()).normalized())

        def mouseReleaseEvent(self, event):
            rect = self.rubber.geometry()
            self.rubber.hide()
            display_rect = {"left": rect.x(), "top": rect.y(), "width": rect.width(), "height": rect.height()}
            box = clamp_box_to_image(scale_box_to_image(display_rect, self.wnd.scale, self.wnd.offset_x, self.wnd.offset_y), self.wnd.image_width, self.wnd.image_height)
            self.wnd.set_manual_box(self.wnd.active_region, box)

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.profile = profile
            self.source = args.source
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self.image_height, self.image_width = frame.shape[:2]
            self.active_region = "reels"

            root = QWidget()
            self.setCentralWidget(root)
            v = QVBoxLayout(root)
            self.info = QLabel()
            v.addWidget(self.info)
            self.canvas = Canvas(self)
            v.addWidget(self.canvas)
            h = QHBoxLayout()
            self.region = QComboBox()
            self.region.addItems(["reels", "spin_button", "popup_close"])
            self.region.currentTextChanged.connect(self.on_region)
            h.addWidget(self.region)
            btn_rerun = QPushButton("Re-run Auto Detect")
            btn_rerun.clicked.connect(self.rerun)
            h.addWidget(btn_rerun)
            btn_confirm = QPushButton("Confirm Calibration")
            btn_confirm.clicked.connect(self.confirm)
            h.addWidget(btn_confirm)
            btn_save = QPushButton("Save Profile")
            btn_save.clicked.connect(self.save)
            h.addWidget(btn_save)
            btn_exit = QPushButton("Exit")
            btn_exit.clicked.connect(self.close)
            h.addWidget(btn_exit)
            v.addLayout(h)
            self.refresh()

        def on_region(self, text: str) -> None:
            self.active_region = text
            logger.info(f"[CALIBRATION][REGION_SELECTED] region={text}")

        def set_manual_box(self, region: str, box: dict[str, int]) -> None:
            apply_manual_override(self.profile, region, box)
            logger.info(f"[CALIBRATION][MANUAL_REGION_UPDATED] region={region} box={box}")
            self.refresh()

        def rerun(self) -> None:
            logger.info("[CALIBRATION][AUTO_RERUN]")
            d = run_auto_detection(frame)
            self.profile = _build_profile(frame, self.profile.get("game_profile", args.profile_name), d)
            self.refresh()

        def confirm(self) -> None:
            test = dict(self.profile)
            test["confirmed"] = True
            ok, errors = finalize_profile_for_save(test)
            if not ok:
                logger.info(f"[CALIBRATION][CONFIRM_FAILED] errors={errors}")
                return
            self.profile["confirmed"] = True
            self.profile["confirmed_at"] = datetime.now(UTC).isoformat()
            self.profile["manual_required"] = False
            logger.info("[CALIBRATION][CONFIRMED]")
            self.refresh()

        def save(self) -> None:
            self.profile["ui_reviewed_at"] = datetime.now(UTC).isoformat()
            ok, errors = finalize_profile_for_save(self.profile)
            if not ok:
                logger.info(f"[CALIBRATION][CONFIRM_FAILED] errors={errors}")
                self.profile["calibrated"] = False
            path = save_profile(self.profile.get("game_profile", args.profile_name), self.profile)
            logger.info(f"[CALIBRATION][PROFILE_SAVED] path={path}")

        def refresh(self) -> None:
            detections = {
                "reels": {"box": {k: self.profile["regions"]["reels"][k] for k in ("left", "top", "width", "height")}, "confidence": self.profile["regions"]["reels"].get("confidence", 0.0)},
                "spin_button": {"box": {k: self.profile["regions"]["spin_button"][k] for k in ("left", "top", "width", "height")}, "confidence": self.profile["regions"]["spin_button"].get("confidence", 0.0)},
                "popup_close": {"box": {k: self.profile["regions"]["popup_close"][k] for k in ("left", "top", "width", "height")}, "confidence": self.profile["regions"]["popup_close"].get("confidence", 0.0)},
            }
            img = draw_detection_overlay(frame, detections, self.profile.get("grid"))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, c * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            self.scale = 1.0
            self.offset_x = 0
            self.offset_y = 0
            self.canvas.setPixmap(pix)
            self.info.setText(f"source={self.source} profile={self.profile.get('game_profile')} confirmed={self.profile.get('confirmed', False)} calibrated={self.profile.get('calibrated', False)} manual_required={self.profile.get('manual_required', False)}")

    app = QApplication([])
    win = Window()
    win.setWindowTitle("Calibration Review")
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
