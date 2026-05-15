from __future__ import annotations

from types import SimpleNamespace
import os

import numpy as np

from src.models import Region
from src.multi_slot import MultiSlotEngine, calculate_game_metrics, render_game_report
from src.reel_parser import parse_frame_to_spin_grid


def main() -> None:
    engine = MultiSlotEngine("config/slot_profiles")
    profile = engine.profiles["candy_demo"]
    reels = Region(left=10, top=10, width=150, height=90)
    vals = {"A": 30, "K": 120, "WILD": 220}
    os.makedirs("logs", exist_ok=True)
    template_paths = {}
    for sym, v in vals.items():
        arr = np.full((20, 20), v, dtype=np.uint8)
        path = f"logs/{sym}_tmpl.npy"
        np.save(path, arr)
        template_paths[sym] = path
    profile.symbol_templates = template_paths

    expected = [["A", "K", "WILD"] for _ in range(profile.reel_count)]
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    cw = reels.width // profile.reel_count
    ch = reels.height // profile.row_count
    for c, col in enumerate(expected):
        for r, sym in enumerate(col):
            v = vals[sym]
            frame[reels.top + r * ch: reels.top + (r + 1) * ch, reels.left + c * cw: reels.left + (c + 1) * cw] = (v, v, v)

    parsed = parse_frame_to_spin_grid(frame, profile, SimpleNamespace(reels=reels), frame_index=1)
    print("parsed grid:", parsed.grid)
    print("confidence summary:", {"min": parsed.min_confidence, "avg": parsed.avg_confidence, "unknown": parsed.unknown_count, "status": parsed.parser_status})

    event = engine.ingest_spin({"game_id": profile.game_id, "session_id": "demo", "bet_amount": 1.0, "payout_amount": 0.0, "grid": parsed.grid, "confidence": parsed.avg_confidence})
    print("ingestion result:", {"game_id": event["game_id"], "wild_count": event["wild_count"], "scatter_count": event["scatter_count"]})

    spins = engine.store.game_spins(profile.game_id)
    metrics = calculate_game_metrics(profile.game_id, spins, profile)
    print("generated game report:\n", render_game_report(profile, metrics))


if __name__ == "__main__":
    main()
