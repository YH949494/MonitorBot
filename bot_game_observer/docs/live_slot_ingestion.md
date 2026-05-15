# Live slot ingestion (Phase 6)

## Enable
Set environment variables:
- `MULTI_SLOT_LIVE_INGEST_ENABLED=1`
- `MULTI_SLOT_GAME_ID=<game_id>`
- `MULTI_SLOT_PROFILE_DIR=config/slot_profiles`
- `MULTI_SLOT_MIN_PARSE_CONFIDENCE=0.80`
- `MULTI_SLOT_REQUIRE_EXACT_READY_STATE=1`
- `MULTI_SLOT_OUTPUT_DIR=logs/multi_slot`

Default behavior is disabled (`MULTI_SLOT_LIVE_INGEST_ENABLED=0`).

## Prerequisites
- A valid game profile JSON must exist in `MULTI_SLOT_PROFILE_DIR`.
- `symbol_templates` in the profile must point to calibrated template files.
- Run calibration before enabling (`python -m calibrate_reel_templates`).

## Runtime behavior
At spin finalization, the runner performs best-effort ingestion:
1. parse frame into a reel grid,
2. verify parser status and confidence,
3. ingest into `MultiSlotEngine` only when parser status is `ok` and confidence threshold passes.

Low-confidence or failed parsing is skipped to avoid false confirmed spins.
Failures are non-fatal and only logged as warnings.

Ready-state gate behavior:
- `MULTI_SLOT_REQUIRE_EXACT_READY_STATE=1` (recommended production default): ingestion only proceeds when runner confirms `READY_TO_SPIN`.
- `MULTI_SLOT_REQUIRE_EXACT_READY_STATE=0`: ingestion may parse/ingest even when exact ready-state confirmation is not present.

## Outputs
On successful ingestion:
- `logs/multi_slot/spins.jsonl`
- `logs/multi_slot/latest_report.txt`
- `logs/multi_slot/latest_metrics.json`

## Limits
This pipeline does **not** prove RTP/fairness by itself.
Without validated real screenshot datasets, no real-world accuracy claims should be made.

## Staged rollout
1. Start disabled in production.
2. Enable for one game profile in a controlled session.
3. Review skipped/ingested ratios and confidence distribution.
4. Validate against manually labeled real screenshots.
5. Expand gradually only after empirical validation.
