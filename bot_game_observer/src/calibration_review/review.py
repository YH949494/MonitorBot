"""PySide6 calibration review window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.app_paths import get_app_root, screenshots_path
from src.bootstrap import init_portable_app
from src.calibration_review.core import (
    apply_manual_override,
    confirm_calibration,
    editable_region_names,
    validate_required_regions,
)
from src.calibration_review.overlay import CalibrationOverlay
from src.capture import CaptureService
from src.config import load_settings_auto, resolve_config_path, save_settings
from src.models import BotSettings, Region

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by users without GUI deps.
    raise RuntimeError(
        "PySide6 is required for the calibration review UI. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


def capture_review_screenshot(settings: BotSettings) -> Path:
    import cv2

    path = screenshots_path("calibration_review.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = CaptureService(settings.capture.region).grab_bgr()
    cv2.imwrite(str(path), frame)
    return path


def region_dict(settings: BotSettings) -> dict[str, Region | None]:
    return {name: getattr(settings.regions, name, None) for name in editable_region_names()}


class CalibrationReviewWindow(QMainWindow):
    def __init__(
        self,
        settings: BotSettings,
        config_path: Path,
        screenshot_path: Path,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.config_path = config_path
        self.screenshot_path = screenshot_path
        self.setWindowTitle("MonitorBot Calibration Review")
        self.resize(1200, 780)

        self.overlay = CalibrationOverlay(
            str(screenshot_path),
            region_dict(settings),
            reel_count=5,
            row_count=3,
        )
        self.overlay.regionChanged.connect(self.on_region_changed)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.region_label = QLabel()
        self.region_label.setWordWrap(True)
        self.confidence_label = QLabel()
        self.confidence_label.setWordWrap(True)

        rerun_button = QPushButton("Re-run Auto Detect")
        rerun_button.clicked.connect(self.on_rerun_auto_detect)
        confirm_button = QPushButton("Confirm Calibration")
        confirm_button.clicked.connect(self.on_confirm)
        save_button = QPushButton("Save Profile")
        save_button.clicked.connect(self.on_save)
        exit_button = QPushButton("Exit")
        exit_button.clicked.connect(self.close)

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Calibration Status</b>"))
        side.addWidget(self.status_label)
        side.addWidget(QLabel("<b>Confidence</b>"))
        side.addWidget(self.confidence_label)
        side.addWidget(QLabel("<b>Editable Regions</b>"))
        side.addWidget(self.region_label)
        side.addStretch(1)
        side.addWidget(rerun_button)
        side.addWidget(confirm_button)
        side.addWidget(save_button)
        side.addWidget(exit_button)

        side_container = QWidget()
        side_container.setLayout(side)
        side_container.setFixedWidth(300)

        layout = QHBoxLayout()
        layout.addWidget(self.overlay, stretch=1)
        layout.addWidget(side_container)
        root = QWidget()
        root.setLayout(layout)
        self.setCentralWidget(root)
        self.update_summary()

    def on_region_changed(self, name: str, region: Region) -> None:
        self.settings = apply_manual_override(self.settings, name, region)
        self.update_summary()

    def on_rerun_auto_detect(self) -> None:
        QMessageBox.information(
            self,
            "Auto Detect",
            "Auto detect rerun is not wired into the review UI yet. "
            "Current manual edits were kept unchanged.",
        )

    def on_confirm(self) -> None:
        try:
            self.settings = confirm_calibration(self.settings)
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot Confirm", str(exc))
            return
        self.update_summary()
        QMessageBox.information(self, "Confirmed", "Calibration is confirmed. Save the profile to persist it.")

    def on_save(self) -> None:
        save_settings(self.settings, self.config_path)
        QMessageBox.information(self, "Saved", f"Profile saved to {self.config_path}")

    def update_summary(self) -> None:
        validation = validate_required_regions(self.settings)
        if validation.ok:
            required = "Required regions present."
        else:
            required = "Missing required regions: " + ", ".join(validation.missing)
        self.status_label.setText(
            f"Config: {self.config_path}\n"
            f"Screenshot: {self.screenshot_path}\n"
            f"calibrated={self.settings.calibrated}\n"
            f"{required}"
        )
        self.confidence_label.setText(
            "Manual review required. This UI does not invent confidence values; "
            "confirm only after visually checking the boxes and grid lines."
        )
        lines = []
        for name in editable_region_names():
            region = getattr(self.settings.regions, name, None)
            lines.append(f"{name}: {region.model_dump() if region else 'not set'}")
        self.region_label.setText("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review and confirm MonitorBot calibration regions.")
    parser.add_argument("--config", default=None, help="Config path, default config/settings.json")
    parser.add_argument("--screenshot", default=None, help="Screenshot to review; captures current region if omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    init_portable_app()
    args = build_parser().parse_args(argv)
    config_path = resolve_config_path(args.config)
    settings = load_settings_auto(args.config)
    screenshot_path = Path(args.screenshot).resolve() if args.screenshot else capture_review_screenshot(settings)
    app = QApplication(sys.argv if argv is None else ["review.py", *argv])
    app.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)
    window = CalibrationReviewWindow(settings, config_path, screenshot_path)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
