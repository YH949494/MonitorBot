# bot_game_observer

Local **QA / research** desktop helper for **demo, sandbox, or test modes** of browser or desktop **slot-style** games. It captures a configured screen region, estimates game state from motion and optional image templates, logs a session, and writes CSV + Markdown analytics.

**Portable layout:** all persistent data (config, logs, exports, screenshots) lives under the **application folder** (next to the executable when packaged). No AppData, registry, or profile storage for normal operation.

This tool is **not** for bypassing anti-cheat, tampering with game clients, reverse engineering binaries, or automating real-money play. It uses only normal screen capture and optional mouse clicks in a user-defined region.

## Features

- **Window or region capture** via `mss` (fast ROI capture).
- **Calibration wizard** (`calibrate.py` or `bot_game_observer.exe calibrate`) for capture area, sub-regions, and template crops.
- **Calibration review UI** (`review.py` or `bot_game_observer.exe review`) for visually confirming regions before live runs.
- **Vision pipeline**: template matching (OpenCV), reel motion via frame differencing, optional OCR hook (pytesseract) if you extend it.
- **State machine** with debounced transitions and confidence metadata.
- **Safe automation**: `enable_clicking: false` by default; `--live-click` required for real clicks; jittered click positions; clicks/minute cap; optional panic-stop file (`logs/STOP.txt`); ESC hotkey when the `keyboard` module works.
- **Outputs**: `logs/sessions/session_<id>.jsonl`, `exports/reports/session_<id>.csv`, `exports/reports/session_<id>.md`.

## Requirements

- **Python 3.11+** on **Windows** (window enumeration uses Win32 APIs), *or* use the **PyInstaller** build (no Python install on target).
- Dependencies: see `requirements.txt`.
- **Tesseract** is **optional** and only needed if you add OCR-based balance reads; core flows do not require it.

## Installation (development)

```powershell
cd bot_game_observer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` only if your environment expects it; settings are loaded from the app folder.

## Portable data layout

On first run the app creates (next to the executable or repo root):

| Path | Purpose |
|------|---------|
| `config/settings.json` | Primary settings (seeded from `config/default.yaml` if missing) |
| `config/default.yaml` | Shipped defaults / reference |
| `data/` | Reserved for local structured data |
| `logs/app.log` | Rotating application log |
| `logs/sessions/` | Session JSONL |
| `logs/STOP.txt` | Create this file to panic-stop |
| `exports/reports/` | CSV and Markdown reports |
| `screenshots/` | Calibration previews |
| `assets/templates/` | Template PNGs |

If an older **`output/`** tree exists (pre-portable), it is **copied** into the new layout once on startup (legacy folders are not deleted).

## Quick start (source)

1. Run once to seed `config/settings.json` from `config/default.yaml`, or run calibration:

   ```powershell
   python calibrate.py
   ```

2. Review the detected regions and explicitly confirm calibration:

   ```powershell
   python review.py
   ```

3. Add template PNGs under `assets/templates/` (see `assets/templates/README.md`) or capture them from the wizard.

4. **Observe only** (default: dry-run, no real clicks):

   ```powershell
   python run.py
   ```

   Optional: `python run.py --config config\default.yaml` (YAML or JSON).

5. **Live clicking** (requires `--live-click` **and** `automation.enable_clicking: true` in config):

   ```powershell
   python run.py --live-click --no-dry-run
   ```

6. Rebuild reports from a log:

   ```powershell
   python analyze_session.py logs\sessions\session_<id>.jsonl
   ```

## Quick start (portable EXE)

From the built folder (`dist/bot_game_observer/`):

```text
bot_game_observer.exe              # same as python run.py
bot_game_observer.exe calibrate
bot_game_observer.exe review
bot_game_observer.exe analyze logs\sessions\session_<id>.jsonl
```

## Portable build (PyInstaller)

From the project root (after `pip install pyinstaller`):

```powershell
pyinstaller --clean build_portable.spec
```

Copy the entire **`dist/bot_game_observer/`** directory to a USB drive or another PC. The first launch creates `config/settings.json` and the folders above. **Do not** split the exe from `_internal` when using the one-folder build.

## Calibration

- Prefer **regions relative to the capture** (`coordinate_mode: relative_to_capture`): when you move the game window, you only update the main capture rectangle.
- Templates are **game-specific**: thresholds must be tuned per title/skin.
- `calibrated: true` should only be saved from the review UI after the required `reels` and `spin_button` regions are visually checked.
- **Near-miss** and **bonus tease** are **not** universal concepts—define them with your own crops and conservative thresholds.

## Safety defaults

- Paused until the review UI confirms and saves `calibrated: true`.
- **Dry-run** for click logging unless `--no-dry-run` is passed with `--live-click`.
- No keyboard automation; only optional single clicks in the spin region.
- Create file `logs/STOP.txt` (or path in config) to stop the session on the next loop iteration.

## Limitations

- **Per-game calibration** is mandatory for meaningful metrics.
- **False positives/negatives** are possible (lighting, animations, overlays).
- **OCR** is fragile; prefer templates and motion where possible.
- **ESC hotkey** needs the `keyboard` package; if it fails to load, use Ctrl+C or the panic file.

## Troubleshooting

- **Empty detections**: verify capture region, reduce `spinning_motion_threshold`, or increase FPS slightly.
- **Templates never match**: re-capture PNGs from the same resolution/scale; lower `threshold` slightly.
- **Import errors**: run commands from the project root so `src` resolves on `PYTHONPATH` (see `pyproject.toml` for pytest).

## Sample outputs

See `output/samples/` in the repo for example JSONL, CSV, and Markdown lines (illustrative paths).

## License

Use only in compliance with the game’s terms of service and applicable law. This repository is intended for authorized QA and research in demo/sandbox environments.
