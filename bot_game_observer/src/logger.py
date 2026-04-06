"""Structured logging setup (console + portable file under logs/)."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from .app_paths import app_log_file, logs_path


def setup_logging(
    level: int = logging.INFO,
    json_style: bool = False,
    *,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Configure the package logger with Rich (or stdout) plus a rotating file under
    ``logs/app.log`` unless ``json_style`` is True.
    """
    log = logging.getLogger("bot_game_observer")
    log.handlers.clear()
    log.setLevel(level)

    if json_style:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(level)
        log.addHandler(handler)
        return log

    rh = RichHandler(rich_tracebacks=True, show_time=True, markup=True)
    rh.setLevel(level)
    log.addHandler(rh)

    logs_path().mkdir(parents=True, exist_ok=True)
    path = log_file or app_log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        path,
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    log.addHandler(fh)
    return log


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "bot_game_observer")
