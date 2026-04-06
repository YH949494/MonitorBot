#!/usr/bin/env python3
"""
Unified entry point for portable builds: ``bot_game_observer.exe``,
``bot_game_observer.exe calibrate``, ``bot_game_observer.exe analyze <jsonl>``.
"""

from __future__ import annotations

import sys


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "calibrate":
        sys.argv = [sys.argv[0]] + args[1:]
        from calibrate import main as calibrate_main

        return calibrate_main()
    if args and args[0] == "analyze":
        sys.argv = [sys.argv[0]] + args[1:]
        from analyze_session import main as analyze_main

        return analyze_main()
    from run import main as run_main

    return run_main()


if __name__ == "__main__":
    sys.exit(main())
