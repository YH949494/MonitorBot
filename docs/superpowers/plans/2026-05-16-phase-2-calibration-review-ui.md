# Phase 2 Calibration Review UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-safe Phase 2 calibration review layer so users can inspect, adjust, confirm, and save detected regions without marking bad profiles as calibrated.

**Architecture:** Keep testable calibration review behavior independent from GUI imports. Put scaling, region conversion, manual override, and confirmation validation in a pure module; put PySide6 drawing and window code in thin UI modules that call the pure module.

**Tech Stack:** Python 3.11+, Pydantic models already in `src.models`, pytest, PySide6 for the desktop review UI.

---

### Task 1: Pure Review Core

**Files:**
- Create: `bot_game_observer/src/calibration_review/__init__.py`
- Create: `bot_game_observer/src/calibration_review/core.py`
- Test: `bot_game_observer/tests/test_calibration_review_core.py`

- [ ] **Step 1: Write failing tests**

Add tests for:
- fitting image dimensions into a widget while preserving aspect ratio
- converting image regions to display coordinates and back
- applying manual overrides to editable regions
- refusing confirmation when required regions are missing
- marking `calibrated=true` only after confirmation

- [ ] **Step 2: Run failing tests**

Run: `python -m pytest tests/test_calibration_review_core.py -q`
Expected: import failure for `src.calibration_review.core`.

- [ ] **Step 3: Implement pure helpers**

Implement small dataclasses and functions:
- `DisplayTransform`
- `fit_image_to_view`
- `region_to_display`
- `region_from_display`
- `editable_region_names`
- `apply_manual_override`
- `validate_required_regions`
- `confirm_calibration`

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_calibration_review_core.py -q`
Expected: all tests pass.

### Task 2: PySide6 Overlay

**Files:**
- Create: `bot_game_observer/src/calibration_review/overlay.py`
- Create: `bot_game_observer/src/calibration_review/review.py`
- Modify: `bot_game_observer/requirements.txt`

- [ ] **Step 1: Add `PySide6>=6.6.0` to requirements**

- [ ] **Step 2: Implement overlay widget**

Create a `CalibrationOverlay` widget that:
- renders screenshot pixmap scaled into the available view
- draws `reels`, `spin_button`, and `popup_close`
- draws grid lines for `reels`
- supports dragging selected region rectangles
- emits/manual stores updated image-coordinate regions

- [ ] **Step 3: Implement review window**

Create a CLI-capable PySide6 entrypoint with:
- config loading
- screenshot loading
- confidence summary panel
- Re-run Auto Detect placeholder button
- Confirm Calibration
- Save Profile
- Exit

The first implementation may show "auto detect rerun is not wired yet" instead of inventing fake detection.

### Task 3: Verification

**Files:**
- Existing test suite and git state

- [ ] **Step 1: Run focused tests**

Run: `python -m pytest tests/test_calibration_review_core.py -q`

- [ ] **Step 2: Run existing calibration/model compatibility tests**

Run: `python -m pytest tests/test_models_config_compat.py tests/test_template_calibration.py -q`

- [ ] **Step 3: Inspect diff**

Run: `git status --short`
Run: `git diff -- bot_game_observer/src/calibration_review bot_game_observer/tests/test_calibration_review_core.py bot_game_observer/requirements.txt`

