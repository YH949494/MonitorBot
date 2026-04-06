#!/usr/bin/env python3
"""Portable PyInstaller entrypoint."""

from __future__ import annotations

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


def main() -> int:
    try:
        init_portable_app(create_config=True, migrate=True)
        setup_logging()
        log_startup_paths(get_logger())

        from src.main import main as app_main

        return app_main()
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
