from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.live_slot_ingestion import LiveSlotIngestionConfig, ingest_live_spin_event
from src.models import Region


def _mk_templates(tmp_path: Path) -> dict[str, str]:
    out = {}
    for name, v in {"A": 40, "K": 120, "WILD": 220}.items():
        p = tmp_path / f"{name}.npy"
        np.save(p, np.full((20, 20), v, dtype=np.uint8))
        out[name] = str(p)
    return out


def _mk_frame(reels: Region, grid: list[list[str]], value_map: dict[str, int]) -> np.ndarray:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cw = reels.width // len(grid)
    ch = reels.height // len(grid[0])
    for c, col in enumerate(grid):
        for r, sym in enumerate(col):
            y0 = reels.top + r * ch
            x0 = reels.left + c * cw
            v = value_map[sym]
            frame[y0:y0 + ch, x0:x0 + cw] = (v, v, v)
    return frame


def _profile_dir(tmp_path: Path, templates: dict[str, str]) -> Path:
    src = Path("config/slot_profiles/candy_demo.json")
    data = json.loads(src.read_text(encoding="utf-8"))
    data["symbol_templates"] = templates
    out_dir = tmp_path / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candy_demo.json").write_text(json.dumps(data), encoding="utf-8")
    return out_dir


def test_live_ingestion_disabled(tmp_path):
    res = ingest_live_spin_event(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_index=1, session_id="s1", regions_or_settings=SimpleNamespace(reels=Region(left=0, top=0, width=1, height=1)), config=LiveSlotIngestionConfig(enabled=False))
    assert res.enabled is False
    assert res.ingested is False


def test_missing_game_id(tmp_path):
    cfg = LiveSlotIngestionConfig(enabled=True, game_id="", profile_dir="config/slot_profiles")
    res = ingest_live_spin_event(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_index=2, session_id="s2", regions_or_settings=SimpleNamespace(reels=Region(left=0, top=0, width=1, height=1)), config=cfg)
    assert res.reason == "missing_game_id"


def test_low_confidence_not_ingested(tmp_path):
    templates = _mk_templates(tmp_path)
    pdir = _profile_dir(tmp_path, templates)
    reels = Region(left=10, top=10, width=150, height=90)
    grid = [["A", "K", "WILD"] for _ in range(5)]
    frame = _mk_frame(reels, grid, {"A": 40, "K": 120, "WILD": 220})
    frame[10:40, 10:40] = (0, 0, 0)
    cfg = LiveSlotIngestionConfig(enabled=True, game_id="candy_demo", profile_dir=str(pdir), min_parse_confidence=0.99, output_dir=str(tmp_path / "out"))
    res = ingest_live_spin_event(frame=frame, frame_index=3, session_id="s3", regions_or_settings=SimpleNamespace(reels=reels), config=cfg)
    assert res.ingested is False


def test_ok_parse_ingested_and_outputs_written(tmp_path):
    templates = _mk_templates(tmp_path)
    pdir = _profile_dir(tmp_path, templates)
    reels = Region(left=10, top=10, width=150, height=90)
    frame = _mk_frame(reels, [["A", "K", "WILD"] for _ in range(5)], {"A": 40, "K": 120, "WILD": 220})
    outdir = tmp_path / "out"
    cfg = LiveSlotIngestionConfig(enabled=True, game_id="candy_demo", profile_dir=str(pdir), min_parse_confidence=0.8, output_dir=str(outdir))
    res = ingest_live_spin_event(frame=frame, frame_index=4, session_id="s4", regions_or_settings=SimpleNamespace(reels=reels), bet_amount=1.0, payout_amount=0.0, config=cfg)
    assert res.ingested is True
    assert (outdir / "spins.jsonl").exists()
    assert (outdir / "latest_report.txt").exists()


def test_ingestion_failure_nonfatal(tmp_path):
    cfg = LiveSlotIngestionConfig(enabled=True, game_id="candy_demo", profile_dir="does_not_exist")
    res = ingest_live_spin_event(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_index=5, session_id="s5", regions_or_settings=SimpleNamespace(reels=Region(left=0, top=0, width=1, height=1)), config=cfg)
    assert res.ingested is False
    assert "ingestion_error" in res.reason or res.reason == "game_profile_not_found"


def test_default_config_requires_exact_ready_state():
    cfg = LiveSlotIngestionConfig()
    assert cfg.require_exact_ready_state is True


def test_require_exact_ready_state_skips_before_parse(tmp_path):
    cfg = LiveSlotIngestionConfig(enabled=True, game_id="candy_demo", profile_dir="config/slot_profiles", require_exact_ready_state=True)
    res = ingest_live_spin_event(
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        frame_index=6,
        session_id="s6",
        regions_or_settings=SimpleNamespace(reels=Region(left=0, top=0, width=1, height=1)),
        config=cfg,
        ready_state_confirmed=False,
    )
    assert res.parser_status == "skipped"
    assert res.reason == "ready_state_not_confirmed"
    assert res.ingested is False


def test_require_exact_ready_state_false_allows_parse_and_ingest(tmp_path):
    templates = _mk_templates(tmp_path)
    pdir = _profile_dir(tmp_path, templates)
    reels = Region(left=10, top=10, width=150, height=90)
    frame = _mk_frame(reels, [["A", "K", "WILD"] for _ in range(5)], {"A": 40, "K": 120, "WILD": 220})
    outdir = tmp_path / "out2"
    cfg = LiveSlotIngestionConfig(
        enabled=True,
        game_id="candy_demo",
        profile_dir=str(pdir),
        min_parse_confidence=0.8,
        output_dir=str(outdir),
        require_exact_ready_state=False,
    )
    res = ingest_live_spin_event(
        frame=frame,
        frame_index=7,
        session_id="s7",
        regions_or_settings=SimpleNamespace(reels=reels),
        bet_amount=1.0,
        payout_amount=0.0,
        config=cfg,
        ready_state_confirmed=False,
    )
    assert res.ingested is True

