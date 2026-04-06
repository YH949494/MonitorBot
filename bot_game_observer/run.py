#!/usr/bin/env python3
"""Run the session observer (portable: uses ``config/settings.json`` by default)."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from src.bootstrap import init_portable_app, log_startup_paths
from src.config import load_settings_auto
from src.logger import get_logger, setup_logging
from src.session_runner import run_session

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description="Game window observer / QA bot (demo mode only).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file (.json or .yaml). Default: config/settings.json under app folder.",
    )
    parser.add_argument(
        "--live-click",
        action="store_true",
        help="Perform real mouse clicks (still requires automation.enable_clicking: true in YAML).",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Disable dry-run logging for clicks (use with --live-click).",
    )
    args = parser.parse_args()

    init_portable_app()
    setup_logging()
    log_startup_paths(get_logger())

    try:
        settings = load_settings_auto(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 2

    if not settings.calibrated:
        console.print(
            "[red]Config has calibrated: false. Run `python calibrate.py` first, "
            "or set calibrated: true after reviewing regions.[/red]"
        )
        return 2

    dry_run = not args.no_dry_run
    if args.live_click:
        console.print(
            "[yellow]Live clicking requested. Ensure demo/sandbox mode, "
            "automation.enable_clicking: true, and supervision.[/yellow]"
        )

    run_session(settings, live_click=args.live_click, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
