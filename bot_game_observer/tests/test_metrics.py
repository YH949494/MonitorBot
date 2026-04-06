"""Tests for metrics aggregation."""

from __future__ import annotations

from datetime import datetime, timezone

from src.metrics import build_summary


def test_build_summary_basic() -> None:
    events = [
        {
            "session_id": "s1",
            "ts": "2026-04-06T10:00:00+00:00",
            "event_type": "spin_started",
            "payload": {},
        },
        {
            "session_id": "s1",
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "win_detected",
            "payload": {},
        },
        {
            "session_id": "s1",
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_started",
            "payload": {},
        },
        {
            "session_id": "s1",
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "no_win_detected",
            "payload": {},
        },
        {
            "session_id": "s1",
            "ts": "2026-04-06T10:00:04+00:00",
            "event_type": "session_stopped",
            "payload": {"reason": "max_spins"},
        },
    ]
    s = build_summary("s1", events)
    assert s.total_spins == 2
    assert s.total_wins == 1
    assert s.total_no_win == 1
    assert s.first_win_spin_index == 1
    assert s.spins_before_first_win == 0
    assert s.gaps_between_wins == []
    assert s.end_reason == "max_spins"


def test_gaps_between_wins() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:03+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:04+00:00", "event_type": "win_detected", "payload": {}},
    ]
    s = build_summary("g", events)
    assert s.gaps_between_wins == [2]
