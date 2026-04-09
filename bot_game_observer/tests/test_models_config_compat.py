from __future__ import annotations

import logging

from src.models import DetectionConfig


def test_legacy_payout_config_bridges_to_new_sampling_fields() -> None:
    cfg = DetectionConfig(
        payout_read_delay_sec=0.5,
        payout_read_retry_window_sec=2.5,
        payout_read_max_attempts=9,
    )
    assert cfg.payout_sampling_initial_delay_ms == 500
    assert cfg.payout_sampling_window_ms == 2500
    assert cfg.payout_stabilization_max_attempts == 9


def test_new_payout_config_precedence_over_legacy_values() -> None:
    cfg = DetectionConfig(
        payout_read_delay_sec=0.5,
        payout_read_retry_window_sec=2.5,
        payout_read_max_attempts=9,
        payout_sampling_initial_delay_ms=111,
        payout_sampling_window_ms=2222,
        payout_stabilization_max_attempts=7,
    )
    assert cfg.payout_sampling_initial_delay_ms == 111
    assert cfg.payout_sampling_window_ms == 2222
    assert cfg.payout_stabilization_max_attempts == 7


def test_legacy_payout_config_emits_operator_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        DetectionConfig(
            payout_read_delay_sec=0.5,
            payout_read_retry_window_sec=2.5,
            payout_read_max_attempts=9,
        )
    assert any("Legacy payout config keys applied" in rec.message for rec in caplog.records)


def test_legacy_and_new_payout_config_emits_precedence_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        DetectionConfig(
            payout_read_delay_sec=0.5,
            payout_sampling_initial_delay_ms=111,
        )
    assert any("new payout settings took precedence" in rec.message for rec in caplog.records)
