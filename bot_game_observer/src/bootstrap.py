"""
Portable startup: create directories, verify writability, migrate legacy paths.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import yaml

from .app_paths import (
    LEGACY_OUTPUT_DIR,
    SETTINGS_JSON,
    app_log_file,
    config_path,
    exports_path,
    get_app_root,
    logs_path,
    portable_subdirs,
    resource_path,
    screenshots_path,
)
from .models import BotSettings

log = logging.getLogger("bot_game_observer.bootstrap")


def _test_write(dir_path: Path) -> None:
    probe = dir_path / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write to portable folder {dir_path}: {e}. "
            "Copy the app to a writable location or adjust permissions."
        ) from e


def ensure_portable_directories() -> None:
    for d in portable_subdirs():
        d.mkdir(parents=True, exist_ok=True)
    _test_write(get_app_root())


def migrate_legacy_layout() -> None:
    """
    If a pre-portable ``output/`` tree exists, copy into the new layout once.
    Does not delete legacy folders (safe rollback).
    """
    legacy = LEGACY_OUTPUT_DIR
    if not legacy.is_dir():
        return

    # Session JSONL: output/logs -> logs/sessions
    old_logs = legacy / "logs"
    new_sessions = logs_path("sessions")
    if old_logs.is_dir():
        for f in old_logs.glob("*.jsonl"):
            dest = new_sessions / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
                log.info("Migrated session log: %s -> %s", f, dest)

    # Reports: output/reports -> exports/reports
    old_rep = legacy / "reports"
    new_rep = exports_path("reports")
    if old_rep.is_dir():
        for f in old_rep.glob("*"):
            if f.is_file():
                dest = new_rep / f.name
                if not dest.exists():
                    shutil.copy2(f, dest)
                    log.info("Migrated report: %s -> %s", f, dest)

    # Panic stop file
    stop_old = legacy / "STOP.txt"
    stop_new = logs_path("STOP.txt")
    if stop_old.is_file() and not stop_new.exists():
        shutil.copy2(stop_old, stop_new)
        log.info("Migrated panic stop file to %s", stop_new)

    # Calibration preview
    prev_old = legacy / "calibration_preview.png"
    prev_new = screenshots_path("calibration_preview.png")
    if prev_old.is_file() and not prev_new.exists():
        shutil.copy2(prev_old, prev_new)
        log.info("Migrated calibration preview to %s", prev_new)


def ensure_settings_json() -> None:
    """
    Create ``config/settings.json`` from ``config/default.yaml`` or bundled default
    if the JSON file is missing.
    """
    if SETTINGS_JSON.is_file():
        return

    candidates: list[Path] = [config_path("default.yaml"), resource_path("config/default.yaml")]
    raw: dict | None = None
    for p in candidates:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            log.info("Seeding settings from %s", p)
            break
    if raw is None:
        raise FileNotFoundError(
            "No config/default.yaml found under app root or bundled resources. "
            "Add config/default.yaml next to the executable."
        )
    settings = BotSettings.model_validate(raw)
    SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_JSON.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    log.info("Created portable config %s", SETTINGS_JSON)


def migrate_legacy_config() -> None:
    """
    Safely migrate old config locations into ``config/settings.json`` when possible.
    """
    if SETTINGS_JSON.exists():
        return

    legacy_candidates = [
        get_app_root() / "settings.json",
        get_app_root() / "config.yaml",
        get_app_root() / "config.yml",
    ]
    for legacy in legacy_candidates:
        if not legacy.is_file():
            continue
        try:
            if legacy.suffix.lower() == ".json":
                settings = BotSettings.model_validate_json(legacy.read_text(encoding="utf-8"))
            else:
                with legacy.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                settings = BotSettings.model_validate(raw)
        except Exception as e:
            log.warning("Skipping legacy config migration from %s: %s", legacy, e)
            continue

        SETTINGS_JSON.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_JSON.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        log.info("Migrated legacy config: %s -> %s", legacy, SETTINGS_JSON)
        return


def init_portable_app(*, create_config: bool = True, migrate: bool = True) -> Path:
    """
    Full startup sequence. Returns :func:`get_app_root`.
    """
    root = get_app_root()
    ensure_portable_directories()
    if migrate:
        migrate_legacy_layout()
        migrate_legacy_config()
    if create_config:
        ensure_settings_json()
    _test_write(config_path())
    return root


def log_startup_paths(logger: logging.Logger | None = None) -> None:
    """Log resolved paths once (call after logging is configured)."""
    lg = logger or log
    root = get_app_root()
    mode = "frozen" if getattr(sys, "frozen", False) else "dev"
    lg.info("Startup mode: %s", mode)
    lg.info("Portable app root: %s", root)
    lg.info("Config: %s", SETTINGS_JSON)
    lg.info("Data dir: %s", root / "data")
    lg.info("Logs dir: %s", root / "logs")
    lg.info("Sessions (JSONL): %s", logs_path("sessions"))
    lg.info("Exports (CSV/MD): %s", exports_path("reports"))
    lg.info("Screenshots: %s", root / "screenshots")
    lg.info("App log file: %s", app_log_file())
