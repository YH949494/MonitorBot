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


def test_spin_result_summary_counts_and_timeouts() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "visual_win": True,
                "any_payout": None,
                "result_class": "result_unknown",
                "timeouts": {"click_to_spinning": True, "spinning_to_result": False, "result_to_ready": False},
            },
        },
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "visual_win": False,
                "any_payout": True,
                "result_class": "real_win",
                "timeouts": {"click_to_spinning": False, "spinning_to_result": True, "result_to_ready": True},
            },
        },
    ]
    s = build_summary("r", events)
    assert s.visual_win_count == 1
    assert s.any_payout_count == 1
    assert s.real_win_count == 1
    assert s.result_unknown_count == 1
    assert s.click_to_spinning_timeout_count == 1
    assert s.spinning_to_result_timeout_count == 1
    assert s.result_to_ready_timeout_count == 1


def test_session_summary_tracks_empty_visual_big_and_missing_payout() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "no_win",
                "payout": 0.0,
                "visual_win_by_bet": False,
                "big_win": False,
                "locked_session_bet": 2.0,
            },
        },
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 2,
                "result_kind": "win",
                "payout": 1.0,
                "visual_win_by_bet": True,
                "big_win": False,
            },
        },
        {"ts": "2026-04-06T10:00:04+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:05+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 3,
                "result_kind": "win",
                "payout": None,
                "visual_win_by_bet": None,
                "big_win": None,
            },
        },
    ]
    s = build_summary("m", events)
    assert s.empty_spin_count == 1
    assert s.visual_win_count == 1
    assert s.big_win_count == 0
    assert s.missing_payout_count == 1
    assert s.locked_session_bet == 2.0
