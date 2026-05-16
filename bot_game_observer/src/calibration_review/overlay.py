"""PySide6 overlay widget for interactive calibration review."""

from __future__ import annotations

from dataclasses import dataclass

from src.calibration_review.core import (
    DisplayTransform,
    editable_region_names,
    fit_image_to_view,
    region_from_display,
    region_to_display,
)
from src.models import Region

try:
    from PySide6.QtCore import QPoint, QRect, Qt, Signal
    from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import QWidget
except ImportError as exc:  # pragma: no cover - exercised by users without GUI deps.
    raise RuntimeError(
        "PySide6 is required for the calibration review UI. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


REGION_COLORS = {
    "reels": QColor(0, 180, 255),
    "spin_button": QColor(40, 210, 110),
    "popup_close": QColor(255, 190, 50),
}


@dataclass(frozen=True)
class DragState:
    region_name: str
    press_pos: QPoint
    start_region: Region


class CalibrationOverlay(QWidget):
    """Render a screenshot with draggable calibration regions."""

    regionChanged = Signal(str, object)

    def __init__(
        self,
        screenshot_path: str,
        regions: dict[str, Region | None],
        reel_count: int = 5,
        row_count: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)
        self.pixmap = QPixmap(screenshot_path)
        if self.pixmap.isNull():
            raise RuntimeError(f"could not load screenshot: {screenshot_path}")
        self.regions = dict(regions)
        self.reel_count = reel_count
        self.row_count = row_count
        self.selected_region: str | None = None
        self.drag: DragState | None = None

    def update_regions(self, regions: dict[str, Region | None]) -> None:
        self.regions = dict(regions)
        self.update()

    def transform(self) -> DisplayTransform:
        return fit_image_to_view(
            image_width=self.pixmap.width(),
            image_height=self.pixmap.height(),
            view_width=max(1, self.width()),
            view_height=max(1, self.height()),
        )

    def paintEvent(self, _event: object) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 27, 31))
        transform = self.transform()
        target = QRect(
            transform.offset_x,
            transform.offset_y,
            transform.display_width,
            transform.display_height,
        )
        painter.drawPixmap(target, self.pixmap)
        for name in editable_region_names():
            region = self.regions.get(name)
            if region is None:
                continue
            self._draw_region(painter, name, region, transform)
        painter.end()

    def mousePressEvent(self, event: object) -> None:
        pos = event.position().toPoint()
        transform = self.transform()
        for name in reversed(editable_region_names()):
            region = self.regions.get(name)
            if region is None:
                continue
            display = region_to_display(region, transform)
            rect = QRect(display.left, display.top, display.width, display.height)
            if rect.contains(pos):
                self.selected_region = name
                self.drag = DragState(name, pos, region)
                self.update()
                return

    def mouseMoveEvent(self, event: object) -> None:
        if self.drag is None:
            return
        pos = event.position().toPoint()
        delta = pos - self.drag.press_pos
        transform = self.transform()
        start_display = region_to_display(self.drag.start_region, transform)
        moved_display = Region(
            left=start_display.left + delta.x(),
            top=start_display.top + delta.y(),
            width=start_display.width,
            height=start_display.height,
        )
        moved = region_from_display(
            moved_display,
            transform,
            image_width=self.pixmap.width(),
            image_height=self.pixmap.height(),
        )
        self.regions[self.drag.region_name] = moved
        self.regionChanged.emit(self.drag.region_name, moved)
        self.update()

    def mouseReleaseEvent(self, _event: object) -> None:
        self.drag = None

    def _draw_region(
        self,
        painter: QPainter,
        name: str,
        region: Region,
        transform: DisplayTransform,
    ) -> None:
        display = region_to_display(region, transform)
        color = REGION_COLORS.get(name, QColor(255, 255, 255))
        width = 3 if name == self.selected_region else 2
        painter.setPen(QPen(color, width))
        painter.drawRect(display.left, display.top, display.width, display.height)
        painter.fillRect(display.left, display.top - 22, 130, 22, QColor(0, 0, 0, 170))
        painter.setPen(QPen(color, 1))
        painter.drawText(display.left + 6, display.top - 7, name)
        if name == "reels":
            self._draw_grid(painter, display, color)

    def _draw_grid(self, painter: QPainter, display: Region, color: QColor) -> None:
        if self.reel_count <= 1 and self.row_count <= 1:
            return
        grid_color = QColor(color)
        grid_color.setAlpha(150)
        painter.setPen(QPen(grid_color, 1, Qt.PenStyle.DashLine))
        for col in range(1, self.reel_count):
            x = display.left + round((display.width * col) / self.reel_count)
            painter.drawLine(x, display.top, x, display.top + display.height)
        for row in range(1, self.row_count):
            y = display.top + round((display.height * row) / self.row_count)
            painter.drawLine(display.left, y, display.left + display.width, y)
