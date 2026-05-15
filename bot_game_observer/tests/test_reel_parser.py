from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.models import Region
from src.multi_slot import MultiSlotEngine
from src.reel_parser import _split_reel_cells, parse_frame_to_spin_grid


def _mk_templates(tmp_path, symbols: dict[str, int]) -> dict[str, str]:
    out = {}
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name, v in symbols.items():
        arr = np.full((20, 20), v, dtype=np.uint8)
        p = tmp_path / f"{name}.npy"
        np.save(p, arr)
        out[name] = str(p)
    return out


def _mk_frame(grid: list[list[str]], value_map: dict[str, int], reels_region: Region) -> np.ndarray:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cw = reels_region.width // len(grid)
    ch = reels_region.height // len(grid[0])
    for c, col in enumerate(grid):
        for r, sym in enumerate(col):
            v = value_map[sym]
            y0 = reels_region.top + r * ch
            x0 = reels_region.left + c * cw
            frame[y0:y0 + ch, x0:x0 + cw] = (v, v, v)
    return frame


def test_split_reel_cells_shape():
    img = np.zeros((90, 150, 3), dtype=np.uint8)
    cells = _split_reel_cells(img, 5, 3)
    assert len(cells) == 5
    assert all(len(col) == 3 for col in cells)


def test_parse_template_matching_and_ingest(tmp_path):
    profiles_dir = "config/slot_profiles"
    engine = MultiSlotEngine(profiles_dir)
    gp = engine.profiles["candy_demo"]
    templates = _mk_templates(tmp_path, {"A": 30, "K": 120, "WILD": 220})
    gp.symbol_templates = templates

    reels = Region(left=10, top=10, width=150, height=90)
    expected = [["A", "K", "WILD"] for _ in range(gp.reel_count)]
    frame = _mk_frame(expected, {"A": 30, "K": 120, "WILD": 220}, reels)

    parsed = parse_frame_to_spin_grid(frame, gp, SimpleNamespace(reels=reels))
    assert parsed.grid == expected
    assert parsed.unknown_count == 0

    event = engine.ingest_spin({"game_id": gp.game_id, "session_id": "s1", "bet_amount": 1, "payout_amount": 0, "grid": parsed.grid})
    assert event["game_id"] == gp.game_id


def test_unknown_low_confidence_and_missing_templates(tmp_path):
    engine = MultiSlotEngine("config/slot_profiles")
    gp = engine.profiles["candy_demo"]
    gp.symbol_templates = _mk_templates(tmp_path, {"A": 30})
    reels = Region(left=0, top=0, width=150, height=90)
    grid = [["A", "A", "A"] for _ in range(gp.reel_count)]
    frame = _mk_frame(grid, {"A": 30}, reels)
    frame[0:30, 0:30] = (250, 250, 250)
    parsed = parse_frame_to_spin_grid(frame, gp, {"reels": reels})
    assert parsed.parser_status == "low_confidence"
    assert parsed.unknown_count >= 1

    gp.symbol_templates = {"A": "missing.png"}
    parsed2 = parse_frame_to_spin_grid(frame, gp, {"reels": reels})
    assert parsed2.parser_status == "failed"


def test_different_games_use_different_templates(tmp_path):
    engine = MultiSlotEngine("config/slot_profiles")
    candy = engine.profiles["candy_demo"]
    egypt = engine.profiles["egypt_demo"]
    candy.symbol_templates = _mk_templates(tmp_path / "c", {"A": 10})
    egypt.symbol_templates = _mk_templates(tmp_path / "e", {"A": 200})
    reels = Region(left=0, top=0, width=150, height=90)
    frame = _mk_frame([["A", "A", "A"] for _ in range(candy.reel_count)], {"A": 10}, reels)
    p1 = parse_frame_to_spin_grid(frame, candy, {"reels": reels})
    p2 = parse_frame_to_spin_grid(frame, egypt, {"reels": reels})
    assert p1.avg_confidence > p2.avg_confidence
