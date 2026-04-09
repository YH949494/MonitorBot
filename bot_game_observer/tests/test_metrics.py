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
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout_effective_value": 1.0},
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
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "no_win", "payout_effective_value": 0.0},
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
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 1, "result_kind": "win"}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 2, "result_kind": "no_win"}},
        {"ts": "2026-04-06T10:00:03+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 3, "result_kind": "win"}},
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
                "payout_truth_source": "ocr_balance_conflict",
                "payout_truth_conflict": True,
                "payout_effective_value": 1.0,
                "payout_effective_source": "ocr_balance_conflict",
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
                "payout_truth_source": "balance_delta_confirmed",
                "payout_effective_value": 2.0,
                "payout_effective_source": "balance_delta_confirmed",
                "timeouts": {"click_to_spinning": False, "spinning_to_result": True, "result_to_ready": True},
            },
        },
    ]
    s = build_summary("r", events)
    assert s.visual_win_count == 1
    assert s.any_payout_count == 2
    assert s.real_win_count == 1
    assert s.result_unknown_count == 1
    assert s.click_to_spinning_timeout_count == 1
    assert s.spinning_to_result_timeout_count == 1
    assert s.result_to_ready_timeout_count == 1
    assert s.payout_truth_conflict_count == 1
    assert s.balance_delta_confirmed_count == 1
    assert s.payout_effective_resolved_count == 2
    assert s.payout_effective_unresolved_count == 0
    assert s.balance_backed_payout_count == 2


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


def test_spin_result_summary_is_authoritative_for_counts_and_first_win() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "no_win_detected", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "no_win", "payout": 0.0},
        },
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "win_detected", "payload": {}},
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "win_unreadable", "payout": None},
        },
        {"ts": "2026-04-06T10:00:04+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:04+00:00", "event_type": "win_detected", "payload": {}},
        {
            "ts": "2026-04-06T10:00:05+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 3, "result_kind": "no_win", "payout": 0.0},
        },
    ]
    s = build_summary("auth", events)
    assert s.total_spins == 3
    assert s.total_wins == 0
    assert s.total_no_win == 2
    assert s.unreadable_win_count == 1
    assert s.first_win_spin_index is None
    assert s.first_readable_win_spin_index is None
    assert s.first_finalized_non_no_win_spin_index == 2
    assert s.spins_before_first_win is None
    assert s.gaps_between_wins == []


def test_spin_result_summary_win_gap_includes_win_unreadable() -> None:
    events = [
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout": 2.0},
        },
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "no_win", "payout": 0.0},
        },
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 3, "result_kind": "win_unreadable", "payout": None},
        },
    ]
    s = build_summary("gap", events)
    assert s.total_wins == 1
    assert s.unreadable_win_count == 1
    assert s.first_win_spin_index == 1
    assert s.gaps_between_wins == []


def test_metrics_ignore_transient_events_without_spin_result_summary() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:03+00:00", "event_type": "no_win_detected", "payload": {}},
    ]
    s = build_summary("fallback", events)
    assert s.total_spins == 0
    assert s.total_wins == 0
    assert s.total_no_win == 0
    assert s.first_win_spin_index is None
    assert s.session_valid_for_analysis is False
    assert s.coverage_ratio == 0.0
    assert s.session_quality == "invalid"
    assert s.session_exclusion_reason == "low_coverage_ratio"


def test_finalized_outcomes_are_sorted_and_deduped_for_metrics() -> None:
    events = [
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 3, "result_kind": "no_win", "payout": 0.0},
        },
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout": 2.0},
        },
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "no_win", "payout": 0.0},
        },
        {
            "ts": "2026-04-06T10:00:04+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "win", "payout": 1.0},
        },
    ]
    s = build_summary("sorted", events)
    assert s.total_wins == 1
    assert s.total_no_win == 2
    assert s.first_win_spin_index == 1
    assert any("Duplicate finalized spin_result_summary indices detected" in w for w in s.warnings)


def test_metrics_use_only_spin_result_summary_with_coverage_warning() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {"ts": "2026-04-06T10:00:03+00:00", "event_type": "no_win_detected", "payload": {}},
        {
            "ts": "2026-04-06T10:00:04+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "no_win", "payout": 0.0},
        },
    ]
    s = build_summary("partial", events)
    assert s.total_spins == 1
    assert s.total_wins == 0
    assert s.total_no_win == 1
    assert s.coverage_ratio == 0.5
    assert s.session_quality == "invalid"
    assert s.session_valid_for_analysis is False
    assert s.session_exclusion_reason == "low_coverage_ratio"


def test_first_win_from_finalized_only() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 3, "result_kind": "win"}},
    ]
    s = build_summary("first", events)
    assert s.first_win_spin_index == 3


def test_missing_payout_uses_effective_value() -> None:
    events = [
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout": 2.0, "payout_effective_value": None},
        }
    ]
    s = build_summary("missing_effective", events)
    assert s.missing_payout_count == 1


def test_no_double_counting_from_event_overlap() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:01+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 1, "result_kind": "win"}},
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "no_win_detected", "payload": {}},
        {"ts": "2026-04-06T10:00:03+00:00", "event_type": "spin_result_summary", "payload": {"spin_index": 2, "result_kind": "no_win"}},
    ]
    s = build_summary("nodouble", events)
    assert s.total_spins == 2
    assert s.total_wins == 1
    assert s.total_no_win == 1


def test_session_metrics_and_trust_use_effective_payout_truth() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "win",
                "payout": 2.0,
                "payout_effective_value": 1.0,
                "payout_effective_source": "ocr_balance_conflict",
                "payout_truth_conflict": True,
                "visual_win_by_bet": True,
                "big_win": False,
            },
        },
        {"ts": "2026-04-06T10:00:01.500000+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 2,
                "result_kind": "win",
                "payout": None,
                "payout_effective_value": None,
                "payout_effective_source": "unresolved",
                "visual_win_by_bet": None,
                "big_win": None,
            },
        },
        {"ts": "2026-04-06T10:00:02.500000+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 3,
                "result_kind": "win",
                "payout": 3.0,
                "payout_effective_value": 3.0,
                "payout_effective_source": "ocr_confirmed",
                "visual_win_by_bet": False,
                "big_win": True,
            },
        },
    ]
    s = build_summary("trust", events)
    assert s.visual_win_count == 1
    assert s.big_win_count == 1
    assert s.payout_effective_resolved_count == 2
    assert s.payout_effective_unresolved_count == 1
    assert s.balance_backed_payout_count == 1
    assert s.ocr_only_payout_count == 1
    assert s.payout_conflict_count == 1
    assert s.session_trust_label == "medium"
    assert s.session_trust_score is not None
    assert 0.0 <= s.session_trust_score <= 1.0
    assert s.session_valid_for_analysis is True
    assert s.session_exclusion_reason is None
    assert s.coverage_ratio == 1.0
    assert s.session_quality == "valid"
    assert s.usable_spin_count == 2
    assert s.usable_spin_ratio == (2 / 3)
    assert s.conflict_spin_indices == [1]
    assert s.unresolved_spin_indices == [2]
    assert s.consecutive_conflict_spins_max == 1
    assert s.consecutive_unresolved_spins_max == 1


def test_session_trust_label_low_when_unresolved_and_conflicts_high() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "win",
                "payout_effective_value": None,
                "payout_effective_source": "unresolved",
                "payout_truth_conflict": True,
            },
        },
        {"ts": "2026-04-06T10:00:01.500000+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 2,
                "result_kind": "no_win",
                "payout_effective_value": None,
                "payout_effective_source": "unresolved",
                "payout_truth_conflict": True,
            },
        },
    ]
    s = build_summary("trust_low", events)
    assert s.session_trust_label == "low"
    assert s.session_trust_score is not None and s.session_trust_score < 0.45
    assert s.session_valid_for_analysis is False
    assert s.session_exclusion_reason == "low_trust_score"
    assert any("Low trust session — exclude from primary analysis" in w for w in s.warnings)


def test_session_trust_label_high_with_agreement_and_resolution() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "win",
                "payout_effective_value": 1.0,
                "payout_effective_source": "ocr_balance_agree",
                "payout_truth_source": "ocr_balance_agree",
                "payout_truth_conflict": False,
            },
        },
        {"ts": "2026-04-06T10:00:01.500000+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 2,
                "result_kind": "win",
                "payout_effective_value": 2.0,
                "payout_effective_source": "balance_delta_confirmed",
                "payout_truth_source": "balance_delta_confirmed",
                "payout_truth_conflict": False,
            },
        },
    ]
    s = build_summary("trust_high", events)
    assert s.session_trust_label == "high"
    assert s.session_trust_score is not None and s.session_trust_score >= 0.75
    assert s.session_valid_for_analysis is True
    assert s.session_exclusion_reason is None
    assert s.coverage_ratio == 1.0
    assert s.session_quality == "valid"
    assert s.usable_spin_count == 2
    assert s.usable_spin_ratio == 1.0


def test_any_payout_count_uses_effective_payout_not_legacy_flag() -> None:
    events = [
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_started",
            "payload": {},
        },
        {
            "ts": "2026-04-06T10:00:02+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "win",
                "any_payout": True,
                "payout_effective_value": 0.0,
                "payout_effective_source": "ocr_confirmed",
            },
        },
    ]
    s = build_summary("any_payout", events)
    assert s.any_payout_count == 0


def test_conflict_and_unresolved_streak_counters() -> None:
    events = [
        {"ts": "2026-04-06T10:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout_truth_conflict": True, "payout_effective_value": None},
        },
        {"ts": "2026-04-06T10:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "win", "payout_truth_conflict": True, "payout_effective_value": None},
        },
        {"ts": "2026-04-06T10:00:04+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T10:00:05+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 3, "result_kind": "no_win", "payout_truth_conflict": False, "payout_effective_value": None},
        },
    ]
    s = build_summary("streaks", events)
    assert s.conflict_spin_indices == [1, 2]
    assert s.unresolved_spin_indices == [1, 2, 3]
    assert s.consecutive_conflict_spins_max == 2
    assert s.consecutive_unresolved_spins_max == 3
    assert s.session_quality == "invalid"


def test_coverage_ratio_quality_degraded_at_ninety_percent() -> None:
    events: list[dict[str, object]] = []
    for spin_index in range(1, 11):
        events.append({"ts": f"2026-04-06T10:00:{spin_index:02d}+00:00", "event_type": "spin_started", "payload": {}})
        if spin_index <= 9:
            events.append(
                {
                    "ts": f"2026-04-06T10:01:{spin_index:02d}+00:00",
                    "event_type": "spin_result_summary",
                    "payload": {"spin_index": spin_index, "result_kind": "no_win", "payout_effective_value": 0.0},
                }
            )
    s = build_summary("cov90", events)
    assert s.coverage_ratio == 0.9
    assert s.session_quality == "degraded"
    assert s.session_valid_for_analysis is True
    assert s.session_exclusion_reason == "degraded_coverage"


def test_coverage_ratio_quality_invalid_at_seventy_percent() -> None:
    events: list[dict[str, object]] = []
    for spin_index in range(1, 11):
        events.append({"ts": f"2026-04-06T11:00:{spin_index:02d}+00:00", "event_type": "spin_started", "payload": {}})
        if spin_index <= 7:
            events.append(
                {
                    "ts": f"2026-04-06T11:01:{spin_index:02d}+00:00",
                    "event_type": "spin_result_summary",
                    "payload": {"spin_index": spin_index, "result_kind": "no_win", "payout_effective_value": 0.0},
                }
            )
    s = build_summary("cov70", events)
    assert s.coverage_ratio == 0.7
    assert s.session_quality == "invalid"
    assert s.session_valid_for_analysis is False
    assert s.session_exclusion_reason == "low_coverage_ratio"


def test_missing_expected_spin_count_marks_session_invalid() -> None:
    events = [
        {
            "ts": "2026-04-06T13:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "no_win", "payout_effective_value": 0.0},
        }
    ]
    s = build_summary("missing_expected", events)
    assert s.coverage_ratio is None
    assert s.session_quality == "invalid"
    assert s.session_valid_for_analysis is False
    assert s.session_exclusion_reason == "missing_expected_spin_count"
    assert any("Expected spin count missing — session excluded from primary analysis" in w for w in s.warnings)


def test_anomaly_threshold_exceeded_invalidates_session() -> None:
    events: list[dict[str, object]] = []
    for spin_index in range(1, 6):
        events.append({"ts": f"2026-04-06T12:00:{spin_index:02d}+00:00", "event_type": "spin_started", "payload": {}})
        events.append(
            {
                "ts": f"2026-04-06T12:01:{spin_index:02d}+00:00",
                "event_type": "spin_result_summary",
                "payload": {"spin_index": spin_index, "result_kind": "win", "payout_effective_value": None},
            }
        )
    s = build_summary("anom", events)
    assert s.consecutive_unresolved_spins_max >= 5
    assert s.session_quality == "invalid"
    assert s.session_valid_for_analysis is False
    assert s.session_exclusion_reason == "anomaly_threshold_exceeded"
    assert any("Anomaly threshold exceeded — exclude from primary analysis" in w for w in s.warnings)


def test_anomaly_threshold_invalidates_even_with_high_coverage() -> None:
    events: list[dict[str, object]] = []
    for spin_index in range(1, 6):
        events.append({"ts": f"2026-04-06T14:00:{spin_index:02d}+00:00", "event_type": "spin_started", "payload": {}})
        events.append(
            {
                "ts": f"2026-04-06T14:01:{spin_index:02d}+00:00",
                "event_type": "spin_result_summary",
                "payload": {
                    "spin_index": spin_index,
                    "result_kind": "win",
                    "payout_effective_value": None,
                    "payout_truth_conflict": True,
                },
            }
        )
    s = build_summary("anom_high_cov", events)
    assert s.coverage_ratio == 1.0
    assert s.session_quality == "invalid"
    assert s.session_exclusion_reason == "anomaly_threshold_exceeded"


def test_low_trust_invalidates_even_with_high_coverage() -> None:
    events = [
        {"ts": "2026-04-06T15:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T15:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "win", "payout_effective_value": None, "payout_truth_conflict": True},
        },
        {"ts": "2026-04-06T15:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T15:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "win", "payout_effective_value": None, "payout_truth_conflict": False},
        },
    ]
    s = build_summary("low_trust_high_cov", events)
    assert s.coverage_ratio == 1.0
    assert s.session_trust_score is not None and s.session_trust_score < 0.40
    assert s.session_quality == "invalid"
    assert s.session_exclusion_reason == "low_trust_score"


def test_valid_coverage_with_medium_trust_degrades_for_trust_reason() -> None:
    events = [
        {"ts": "2026-04-06T16:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T16:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 1,
                "result_kind": "win",
                "payout_effective_value": 1.0,
                "payout_truth_conflict": True,
            },
        },
        {"ts": "2026-04-06T16:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T16:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 2,
                "result_kind": "no_win",
                "payout_effective_value": None,
                "payout_truth_conflict": True,
            },
        },
        {"ts": "2026-04-06T16:00:04+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T16:00:05+00:00",
            "event_type": "spin_result_summary",
            "payload": {
                "spin_index": 3,
                "result_kind": "no_win",
                "payout_effective_value": 0.0,
                "payout_truth_conflict": False,
            },
        },
    ]
    s = build_summary("degraded_trust", events)
    assert s.coverage_ratio == 1.0
    assert s.session_trust_score is not None and 0.40 <= s.session_trust_score < 0.55
    assert s.session_quality == "degraded"
    assert s.session_exclusion_reason == "degraded_trust"
    assert s.session_exclusion_reason != "degraded_coverage"
    assert any("Session trust degraded — use with caution" in w for w in s.warnings)


def test_medium_coverage_with_acceptable_trust_degrades_for_coverage_reason() -> None:
    events: list[dict[str, object]] = []
    for spin_index in range(1, 11):
        events.append({"ts": f"2026-04-06T17:00:{spin_index:02d}+00:00", "event_type": "spin_started", "payload": {}})
        if spin_index <= 9:
            payload: dict[str, object] = {"spin_index": spin_index, "result_kind": "no_win", "payout_effective_value": 0.0}
            if spin_index == 1:
                payload["payout_truth_source"] = "ocr_balance_agree"
            events.append(
                {
                    "ts": f"2026-04-06T17:01:{spin_index:02d}+00:00",
                    "event_type": "spin_result_summary",
                    "payload": payload,
                }
            )
    s = build_summary("degraded_cov", events)
    assert s.coverage_ratio == 0.9
    assert s.session_trust_score is not None and s.session_trust_score >= 0.55
    assert s.session_quality == "degraded"
    assert s.session_exclusion_reason == "degraded_coverage"
    assert any("Session coverage degraded — use with caution" in w for w in s.warnings)


def test_usable_spin_ratio_none_when_no_spins() -> None:
    s = build_summary("no_spins", [])
    assert s.total_spins == 0
    assert s.usable_spin_ratio is None


def test_usable_spin_semantics_require_finalized_no_win_or_effective_payout() -> None:
    events = [
        {"ts": "2026-04-06T18:00:00+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T18:00:01+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 1, "result_kind": "no_win", "payout_effective_value": None},
        },
        {"ts": "2026-04-06T18:00:02+00:00", "event_type": "spin_started", "payload": {}},
        {
            "ts": "2026-04-06T18:00:03+00:00",
            "event_type": "spin_result_summary",
            "payload": {"spin_index": 2, "result_kind": "win", "payout_effective_value": None},
        },
    ]
    s = build_summary("usable_semantics", events)
    assert s.usable_spin_count == 1
    assert s.usable_spin_ratio == 0.5
