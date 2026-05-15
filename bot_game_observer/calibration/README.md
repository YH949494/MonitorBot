# Calibration workspace

Use this workspace to calibrate real screenshot symbol templates per game profile.

1. Put local screenshots under `calibration/games/<game_id>/screenshots/` (gitignored).
2. Add labels in `labels/*.json` with `reels_region` and `expected_grid`.
3. Run `python -m calibrate_reel_templates` to generate local templates (`.npy`, gitignored), update text `manifest.json`, print profile mapping, and run accuracy checks.
4. Copy suggested `symbol_templates` mapping into `config/slot_profiles/<game_id>.json` when metrics pass.

Readiness for live monitoring requires real screenshots validated at thresholds; synthetic placeholders only verify pipeline wiring.

Limitations: template matching is sensitive to blur, scale changes, moving symbols, overlays, and provider UI variations.
