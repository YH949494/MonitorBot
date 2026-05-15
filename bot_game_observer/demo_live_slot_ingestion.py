from __future__ import annotations

from pathlib import Path
import json
from types import SimpleNamespace
import tempfile

import numpy as np

from src.live_slot_ingestion import LiveSlotIngestionConfig, ingest_live_spin_event
from src.models import Region


def _mk_templates(tmp: Path) -> dict[str, str]:
    out = {}
    for name, v in {"A": 40, "K": 120, "WILD": 220}.items():
        p = tmp / f"{name}.npy"
        np.save(p, np.full((20, 20), v, dtype=np.uint8))
        out[name] = str(p)
    return out


def _mk_frame(reels: Region) -> np.ndarray:
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    grid = [["A", "K", "WILD"] for _ in range(5)]
    cw = reels.width // 5
    ch = reels.height // 3
    val = {"A": 40, "K": 120, "WILD": 220}
    for c, col in enumerate(grid):
        for r, sym in enumerate(col):
            y0 = reels.top + r * ch
            x0 = reels.left + c * cw
            v = val[sym]
            frame[y0:y0 + ch, x0:x0 + cw] = (v, v, v)
    return frame


if __name__ == "__main__":
    reels = Region(left=10, top=10, width=150, height=90)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        templates = _mk_templates(tmp)
        profile = json.loads(Path("config/slot_profiles/candy_demo.json").read_text(encoding="utf-8"))
        profile["symbol_templates"] = templates
        profile_dir = tmp / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "candy_demo.json").write_text(json.dumps(profile), encoding="utf-8")
        cfg = LiveSlotIngestionConfig(
            enabled=True,
            game_id="candy_demo",
            profile_dir=str(profile_dir),
            min_parse_confidence=0.8,
            output_dir=str(tmp / "logs"),
        )
        frame = _mk_frame(reels)
        res = ingest_live_spin_event(
            frame=frame,
            frame_index=1,
            session_id="demo-session",
            regions_or_settings=SimpleNamespace(reels=reels),
            spin_id="demo-spin-1",
            bet_amount=1.0,
            payout_amount=0.0,
            config=cfg,
        )
        print(res)
        print(f"report={Path(cfg.output_dir) / 'latest_report.txt'}")
