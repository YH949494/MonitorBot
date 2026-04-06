"""
Portable application paths: all persistence is under :func:`get_app_root`.

Works in development (source tree) and in PyInstaller builds (``sys.frozen``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- Core resolution ---------------------------------------------------------

def get_app_root() -> Path:
    """
    Directory that contains the portable app (writable), not a temp extract dir.

    * **Frozen (PyInstaller):** parent directory of the executable.
    * **Development:** repository root (parent of ``src``).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(relative: str | Path) -> Path:
    """
    Read-only bundled file (e.g. default config) shipped inside the build.

    In one-file PyInstaller builds, falls back to ``sys._MEIPASS`` when present.
    Otherwise resolves under :func:`get_app_root`.
    """
    rel = Path(relative)
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bundled = Path(meipass) / rel
            if bundled.is_file() or bundled.is_dir():
                return bundled
    return get_app_root() / rel


def data_path(*parts: str | Path) -> Path:
    """``<app_root>/data/`` + optional sub-path."""
    return get_app_root().joinpath("data", *map(Path, parts))


def logs_path(*parts: str | Path) -> Path:
    """``<app_root>/logs/`` + optional sub-path."""
    return get_app_root().joinpath("logs", *map(Path, parts))


def screenshots_path(*parts: str | Path) -> Path:
    """``<app_root>/screenshots/`` + optional sub-path."""
    return get_app_root().joinpath("screenshots", *map(Path, parts))


def config_path(*parts: str | Path) -> Path:
    """``<app_root>/config/`` + optional sub-path."""
    return get_app_root().joinpath("config", *map(Path, parts))


def exports_path(*parts: str | Path) -> Path:
    """``<app_root>/exports/`` + optional sub-path."""
    return get_app_root().joinpath("exports", *map(Path, parts))


def assets_path(*parts: str | Path) -> Path:
    """``<app_root>/assets/`` (user templates and bundled defaults)."""
    return get_app_root().joinpath("assets", *map(Path, parts))


# --- Well-known files --------------------------------------------------------

SETTINGS_JSON = config_path("settings.json")
DEFAULT_YAML = config_path("default.yaml")
LEGACY_OUTPUT_DIR = get_app_root() / "output"  # pre-portable layout


def portable_subdirs() -> list[Path]:
    """All writable subdirectories created at startup."""
    root = get_app_root()
    return [
        root / "data",
        root / "logs",
        root / "logs" / "sessions",
        root / "screenshots",
        root / "config",
        root / "exports",
        root / "exports" / "reports",
        root / "assets",
        root / "assets" / "templates",
    ]


def app_log_file() -> Path:
    return logs_path("app.log")


def resolve_path_relative_to_app(path_str: str) -> Path:
    """
    Resolve a config path that is relative to the app root (e.g. ``logs/STOP.txt``).
    Absolute paths are returned unchanged.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (get_app_root() / p).resolve()
