"""Windows: enumerate top-level windows by title substring."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable

user32 = ctypes.windll.user32
WCHAR = ctypes.c_wchar


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]  # left, top, right, bottom


def _get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buff = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buff, length + 1)
    return buff.value


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return (0, 0, 0, 0)
    return (rect.left, rect.top, rect.right, rect.bottom)


def enum_windows() -> list[WindowInfo]:
    """Return visible top-level windows with non-empty titles."""

    results: list[WindowInfo] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_title(hwnd)
        if not title.strip():
            return True
        rect = _get_window_rect(hwnd)
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            return True
        results.append(WindowInfo(hwnd=hwnd, title=title, rect=rect))
        return True

    user32.EnumWindows(callback, 0)
    return results


def find_windows_title_contains(substring: str) -> list[WindowInfo]:
    sub = substring.lower()
    return [w for w in enum_windows() if sub in w.title.lower()]


def pick_best_window(substring: str) -> WindowInfo | None:
    matches = find_windows_title_contains(substring)
    if not matches:
        return None
    # Prefer largest area
    def area(w: WindowInfo) -> int:
        r = w.rect
        return (r[2] - r[0]) * (r[3] - r[1])

    return max(matches, key=area)
