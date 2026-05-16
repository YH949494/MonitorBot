#!/usr/bin/env python3
"""Launch the PySide6 calibration review UI."""

from __future__ import annotations

import sys

from src.calibration_review.review import main


if __name__ == "__main__":
    sys.exit(main())
