from __future__ import annotations

import json
from pathlib import Path

from replay_helper import run_spin_replay


def _fixture(name: str) -> list[dict[str, str]]:
    p = Path(__file__).parent / "fixtures" / name
    return json.loads(p.read_text(encoding="utf-8"))


def test_replay_real_win_fixture() -> None:
    rec = run_spin_replay(_fixture("replay_real_win.json"), visual_win=True)
    assert rec["any_payout"] is True
    assert rec["real_win"] is True
    assert rec["result_class"] == "real_win"


def test_replay_visual_win_unresolved_fixture() -> None:
    rec = run_spin_replay(_fixture("replay_visual_win_unresolved.json"), visual_win=True)
    assert rec["visual_win"] is True
    assert rec["result_class"] == "result_unknown"
    assert rec["fallback_used"] is True or rec["payout_resolution_status"] == "unresolved"
