"""Tests for spin result semantic classification."""

from __future__ import annotations

from src.spin_result import SpinResult, classify_spin_result


def test_confirmed_win_classification() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=20.0,
        visual_win=True,
        detector_status="confirmed",
        reason="payout_read_success",
    )
    assert got["any_payout"] is True
    assert got["real_win"] is True
    assert got["result_class"] == "confirmed_win"


def test_break_even_is_confirmed_win_classification() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=2.0,
        visual_win=False,
        detector_status="confirmed",
        reason="payout_read_success",
    )
    assert got["break_even"] is True
    assert got["result_class"] == "confirmed_win"


def test_net_loss_with_payout_is_confirmed_win_classification() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=1.0,
        visual_win=False,
        detector_status="confirmed",
        reason="payout_read_success",
    )
    assert got["any_payout"] is True
    assert got["real_win"] is False
    assert got["net_loss_with_payout"] is True
    assert got["result_class"] == "confirmed_win"


def test_no_payout_classification() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=0.0,
        visual_win=False,
        detector_status="confirmed",
        reason="payout_read_success",
    )
    assert got["no_payout"] is True
    assert got["result_class"] == "confirmed_no_win"


def test_visual_win_unknown_payout_is_probable_win() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=None,
        visual_win=True,
        detector_status="partial",
        reason="payout_not_readable",
    )
    assert got["visual_win"] is True
    assert got["result_class"] == "probable_win"


def test_no_visual_unknown_payout_and_recovery_is_confirmed_no_win() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=None,
        visual_win=False,
        detector_status="confirmed",
        reason="payout_not_readable",
        result_kind="no_win",
        ready_recovered=True,
    )
    assert got["visual_win"] is False
    assert got["result_class"] == "confirmed_no_win"


def test_visible_win_balance_delta_success_real_win() -> None:
    got = classify_spin_result(
        bet=2.0,
        payout=20.0,
        visual_win=True,
        detector_status="partial",
        reason="balance_delta_estimate",
    )
    assert got["result_class"] == "confirmed_win"


def test_result_phase_timeout_flag_supported() -> None:
    spin = SpinResult(spin_index=1)
    assert spin.ts_result_detected is None
    spin.timeouts.result_to_ready = True
    got = classify_spin_result(
        bet=2.0,
        payout=None,
        visual_win=True,
        detector_status="timeout",
        reason="ready_not_recovered",
    )
    assert got["result_class"] == "unreadable_result"
