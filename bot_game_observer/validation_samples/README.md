# Reel validation samples

This folder stores human-labeled reel screenshots used to measure parser accuracy on gameplay frames.
Committed binary images are intentionally avoided in this repository patch flow.

## Dataset layout

- `validation_samples/<game_id>/sample_###.json`: required label metadata.
- `validation_samples/<game_id>/sample_###.png`: runtime-generated synthetic placeholder image (if sample notes mark it as synthetic/placeholder) or manually provided real screenshot.

## Label JSON fields

Each label supports:

- `game_id`, `sample_id`, `image`, `frame_index`
- `reels_region`: `left`, `top`, `width`, `height` pixel coordinates for the reel area.
- `expected_grid`: column-major symbols (`reel_count` columns, each with `row_count` rows).
- `expected_symbols`: counts used to validate `MultiSlotEngine.ingest_spin()` output:
  - `wild_count`
  - `scatter_count`
  - `bonus_count`
- `bet_amount`, `payout_amount`, `notes`

## Synthetic placeholders vs real screenshots

- If `notes` contains `synthetic` or `placeholder`, validation tooling generates the referenced image locally if missing.
- Generation is deterministic from `expected_grid` and `reels_region` and does not use external APIs.
- Existing image files are not overwritten by default.
- Real screenshots should be captured and added manually by the operator.

## How to add a real screenshot

1. Capture a full game frame while reels are visible.
2. Save it as `validation_samples/<game_id>/sample_###.png`.
3. Measure reel boundaries in pixels and set `reels_region`.
4. Label every reel cell in `expected_grid` exactly as intended parser symbol IDs.
5. Set expected wild/scatter/bonus counts consistent with game profile symbol mapping.
6. Avoid using `synthetic`/`placeholder` in `notes` for real frames.

## Run validation

```bash
python -m validate_reel_samples
```

Optional threshold overrides via environment variables:

- `REEL_VALIDATION_MIN_CELL_ACCURACY` (default `0.95`)
- `REEL_VALIDATION_MIN_EXACT_MATCH_RATE` (default `0.80`)
- `REEL_VALIDATION_MAX_UNKNOWN_RATE` (default `0.05`)

## Interpreting failures

- Low confidence means one or more cells were weak template matches.
- `UNKNOWN` cells indicate symbol could not be matched above confidence threshold.
- Grid mismatch means parser symbol(s) differ from labels.
- Count mismatches mean ingestion normalization/counting does not match expected wild/scatter/bonus totals.

Generated placeholder images only validate pipeline wiring; they are not proof of real-world parser accuracy.
This process does **not** validate RTP, payout fairness, or long-term statistical behavior.
