#!/usr/bin/env python3
"""Portable PyInstaller entrypoint."""

from __future__ import annotations

import sys
import traceback

from src.bootstrap import init_portable_app, log_startup_paths
from src.logger import get_logger, setup_logging


def _exit_code_from_system_exit(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    return 1


def _dispatch(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "run"

    if command == "run":
        from src.main import main as app_main

        sys.argv = [argv[0], *argv[2:]]
        return app_main()

    if command == "analyze":
        from analyze_session import main as analyze_main

        sys.argv = [argv[0], *argv[2:]]
        return analyze_main()

    if command == "calibrate":
        try:
            from calibrate import main as calibrate_main
        except ModuleNotFoundError:
            get_logger().error("'calibrate' command is not available in this build")
            return 2

        sys.argv = [argv[0], *argv[2:]]
        return calibrate_main()

    get_logger().error("Unknown command: %s", command)
    return 2


def main() -> int:
    try:
        init_portable_app(create_config=True, migrate=True)
        setup_logging()
        log_startup_paths(get_logger())

        return _dispatch(sys.argv)
    except SystemExit as exc:
        return _exit_code_from_system_exit(exc)
    except Exception:
        try:
            get_logger().exception("Fatal unhandled exception during launcher startup")
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
