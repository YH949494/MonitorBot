#!/usr/bin/env python3
"""
Interactive calibration: capture region, sub-regions, and template crops.

Uses portable paths: ``config/settings.json``, ``screenshots/``, ``assets/templates/``.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from src.app_paths import assets_path, get_app_root, screenshots_path
from src.bootstrap import init_portable_app, log_startup_paths
from src.capture import CaptureService
from src.config import load_settings_auto, resolve_config_path, save_settings
from src.logger import get_logger, setup_logging
from src.models import CaptureMode, CoordinateMode, Region, TemplateSpec
from src.window_finder import pick_best_window

console = Console()


class RegionSelector(tk.Toplevel):
    """Draw a rectangle on a screenshot to define a region."""

    def __init__(self, master: tk.Tk, image_path: Path) -> None:
        super().__init__(master)
        self.title("Drag to select region — release to confirm")
        self.image_path = image_path
        self.start_x = 0
        self.start_y = 0
        self.rect_id: int | None = None

        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError("Could not load preview image")
        self.h, self.w = img.shape[:2]
        try:
            from PIL import Image, ImageTk

            pil_img = Image.open(str(image_path))
            self.photo = ImageTk.PhotoImage(pil_img)
        except Exception:
            self.photo = tk.PhotoImage(file=str(image_path))
        self.canvas = tk.Canvas(self, width=self.w, height=self.h, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event: tk.Event) -> None:
        self.start_x = int(event.x)
        self.start_y = int(event.y)
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_drag(self, event: tk.Event) -> None:
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x, y = int(event.x), int(event.y)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, x, y, outline="red", width=2
        )

    def on_release(self, event: tk.Event) -> None:
        x2, y2 = int(event.x), int(event.y)
        x1, y1 = self.start_x, self.start_y
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        if width < 4 or height < 4:
            messagebox.showwarning("Too small", "Please drag a larger rectangle.")
            return
        self.result = (left, top, width, height)
        self.destroy()


def grab_full_screenshot(path: Path) -> Region:
    import mss
    import cv2

    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = sct.grab(mon)
        import numpy as np

        arr = np.asarray(shot, dtype=np.uint8)
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(path), arr)
        return Region(
            left=int(mon["left"]),
            top=int(mon["top"]),
            width=int(mon["width"]),
            height=int(mon["height"]),
        )


def select_region_interactive(image_path: Path) -> Region | None:
    root = tk.Tk()
    root.withdraw()
    sel = RegionSelector(root, image_path)
    sel.grab_set()
    root.wait_window(sel)
    root.destroy()
    if not sel.result:
        return None
    left, top, w, h = sel.result
    return Region(left=left, top=top, width=w, height=h)


def crop_template_from_capture(
    capture_region: Region,
    rel_region: Region,
    dest_png: Path,
) -> None:
    import cv2

    cap = CaptureService(capture_region)
    frame = cap.grab_bgr()
    crop = frame[rel_region.top : rel_region.top + rel_region.height, rel_region.left : rel_region.left + rel_region.width]
    dest_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest_png), crop)
    console.print(f"Saved template [green]{dest_png}[/green]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate capture regions and templates.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config to load/save (default: config/settings.json)",
    )
    args = parser.parse_args()

    init_portable_app()
    setup_logging()
    log_startup_paths(get_logger())

    cfg_path = resolve_config_path(args.config)
    try:
        settings = load_settings_auto(args.config)
    except FileNotFoundError:
        console.print(
            "[red]No configuration found. Ensure config/default.yaml exists or run from "
            "the portable folder so defaults can be seeded to config/settings.json.[/red]"
        )
        return 2

    root = get_app_root()
    console.print("[bold]Calibration wizard[/bold] (demo/QA use only)\n")

    if Confirm.ask("Enumerate windows and pick by title?", default=True):
        sub = Prompt.ask("Title substring to search", default=settings.capture.window_title_contains or "Chrome")
        w = pick_best_window(sub)
        if w:
            left, top, right, bottom = w.rect
            settings.capture.mode = CaptureMode.WINDOW
            settings.capture.window_title_contains = sub
            settings.capture.region = Region(
                left=left,
                top=top,
                width=max(1, right - left),
                height=max(1, bottom - top),
            )
            console.print(f"Selected window region: {settings.capture.region}")
        else:
            console.print("[yellow]No window matched; use screenshot selection.[/yellow]")

    if Confirm.ask("Select capture region using a screen preview?", default=True):
        preview = screenshots_path("calibration_preview.png")
        preview.parent.mkdir(parents=True, exist_ok=True)
        grab_full_screenshot(preview)
        reg = select_region_interactive(preview)
        if reg:
            settings.capture.region = reg
            settings.capture.mode = CaptureMode.REGION

    console.print("\nEnter sub-regions [b]relative to capture area[/b] (0,0 = top-left of capture).")
    if Confirm.ask("Set spin_button region now?", default=True):
        settings.regions.spin_button = Region(
            left=IntPrompt.ask("spin_button left"),
            top=IntPrompt.ask("spin_button top"),
            width=IntPrompt.ask("spin_button width"),
            height=IntPrompt.ask("spin_button height"),
        )
    if Confirm.ask("Set reels region now?", default=True):
        settings.regions.reels = Region(
            left=IntPrompt.ask("reels left"),
            top=IntPrompt.ask("reels top"),
            width=IntPrompt.ask("reels width"),
            height=IntPrompt.ask("reels height"),
        )

    if Confirm.ask("Capture spin_button template from live window?", default=False):
        dest = assets_path("templates", "spin_button_ready.png")
        crop_template_from_capture(settings.capture.region, settings.regions.spin_button, dest)
        settings.templates["spin_button_ready"] = TemplateSpec(
            path=str(dest.relative_to(root)),
            threshold=FloatPrompt.ask("Threshold", default=0.9),
        )

    settings.capture.coordinate_mode = CoordinateMode.RELATIVE_TO_CAPTURE
    settings.calibrated = Confirm.ask("Mark configuration as calibrated?", default=True)

    save_settings(settings, cfg_path)
    console.print(f"\nSaved configuration to [green]{cfg_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
