from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .multi_slot import MultiSlotEngine, calculate_game_metrics, render_game_report
from .reel_parser import parse_frame_to_spin_grid


@dataclass
class LiveSlotIngestionResult:
    enabled: bool
    game_id: str | None
    session_id: str
    frame_index: int
    parser_status: str
    ingested: bool
    reason: str
    grid: list[list[str]]
    avg_confidence: float
    min_confidence: float
    unknown_count: int
    spin_id: str | None
    output_path: str | None


@dataclass
class LiveSlotIngestionConfig:
    enabled: bool = False
    game_id: str = ""
    profile_dir: str = "config/slot_profiles"
    min_parse_confidence: float = 0.80
    require_exact_ready_state: bool = True
    output_dir: str = "logs/multi_slot"

    @classmethod
    def from_env(cls) -> "LiveSlotIngestionConfig":
        return cls(
            enabled=os.getenv("MULTI_SLOT_LIVE_INGEST_ENABLED", "0") == "1",
            game_id=os.getenv("MULTI_SLOT_GAME_ID", ""),
            profile_dir=os.getenv("MULTI_SLOT_PROFILE_DIR", "config/slot_profiles"),
            min_parse_confidence=float(os.getenv("MULTI_SLOT_MIN_PARSE_CONFIDENCE", "0.80")),
            require_exact_ready_state=os.getenv("MULTI_SLOT_REQUIRE_EXACT_READY_STATE", "1") == "1",
            output_dir=os.getenv("MULTI_SLOT_OUTPUT_DIR", "logs/multi_slot"),
        )


def ingest_live_spin_event(
    *,
    frame: np.ndarray,
    frame_index: int,
    session_id: str,
    regions_or_settings: Any,
    spin_id: str | None = None,
    bet_amount: float | None = None,
    payout_amount: float | None = None,
    free_spin_mode: bool = False,
    config: LiveSlotIngestionConfig | None = None,
    ready_state_confirmed: bool = True,
) -> LiveSlotIngestionResult:
    cfg = config or LiveSlotIngestionConfig.from_env()
    if not cfg.enabled:
        return LiveSlotIngestionResult(False, None, session_id, frame_index, "disabled", False, "live_ingestion_disabled", [], 0.0, 0.0, 0, spin_id, None)
    if not cfg.game_id:
        return LiveSlotIngestionResult(True, "", session_id, frame_index, "failed", False, "missing_game_id", [], 0.0, 0.0, 0, spin_id, None)

    if cfg.require_exact_ready_state and not ready_state_confirmed:
        return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, "skipped", False, "ready_state_not_confirmed", [], 0.0, 0.0, 0, spin_id, None)
    try:
        engine = MultiSlotEngine(cfg.profile_dir)
        if cfg.game_id not in engine.profiles:
            return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, "failed", False, "game_profile_not_found", [], 0.0, 0.0, 0, spin_id, None)
        profile = engine.profiles[cfg.game_id]
        parsed = parse_frame_to_spin_grid(frame, profile, regions_or_settings, frame_index=frame_index)
        if parsed.parser_status != "ok":
            return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, parsed.parser_status, False, parsed.reason, parsed.grid, parsed.avg_confidence, parsed.min_confidence, parsed.unknown_count, spin_id, None)
        if parsed.avg_confidence < cfg.min_parse_confidence:
            return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, parsed.parser_status, False, "avg_confidence_below_threshold", parsed.grid, parsed.avg_confidence, parsed.min_confidence, parsed.unknown_count, spin_id, None)

        payload = {
            "game_id": cfg.game_id,
            "session_id": session_id,
            "spin_id": spin_id,
            "grid": parsed.grid,
            "bet_amount": 0.0 if bet_amount is None else bet_amount,
            "payout_amount": 0.0 if payout_amount is None else payout_amount,
            "free_spin_mode": free_spin_mode,
            "confidence": parsed.avg_confidence,
        }
        event = engine.ingest_spin(payload)
        metrics = calculate_game_metrics(cfg.game_id, engine.store.game_spins(cfg.game_id), profile)
        report = render_game_report(profile, metrics)

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spins_path = out_dir / "spins.jsonl"
        with spins_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        (out_dir / "latest_report.txt").write_text(report, encoding="utf-8")
        (out_dir / "latest_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, parsed.parser_status, True, "ingested", parsed.grid, parsed.avg_confidence, parsed.min_confidence, parsed.unknown_count, event.get("spin_id"), str(spins_path))
    except Exception as exc:
        return LiveSlotIngestionResult(True, cfg.game_id, session_id, frame_index, "failed", False, f"ingestion_error:{exc}", [], 0.0, 0.0, 0, spin_id, None)


def write_result_jsonl(path: Path, result: LiveSlotIngestionResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
