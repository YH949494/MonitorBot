"""Spin result semantics and classification helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DetectorStatus = Literal["confirmed", "partial", "fallback", "timeout", "ambiguous"]
ResultClass = Literal["confirmed_no_win", "confirmed_win", "probable_win", "unreadable_result"]
PostResultVisualClassification = Literal["none", "normal_result", "long_animation", "bonus_like"]


class SpinTimeouts(BaseModel):
    click_to_spinning: bool = False
    spinning_to_result: bool = False
    result_to_ready: bool = False


class SpinResult(BaseModel):
    spin_index: int
    ts_start: datetime | None = None
    ts_click: datetime | None = None
    ts_spinning_detected: datetime | None = None
    ts_result_detected: datetime | None = None
    ts_ready_detected: datetime | None = None
    result_kind: Literal["win", "no_win", "win_unreadable"] | None = None
    post_result_animation_started_at: datetime | None = None
    post_result_animation_duration_sec: float | None = None
    post_result_visual_classification: PostResultVisualClassification = "none"

    bet: float | None = None
    payout: float | None = None
    balance_before: float | None = None
    balance_after: float | None = None
    payout_source: Literal["ocr", "balance_delta", "template", "unknown"] = "unknown"
    payout_resolution_status: Literal["confirmed", "estimated", "unresolved"] = "unresolved"
    raw_bet_ocr_samples: list[str] = Field(default_factory=list)
    raw_payout_ocr_samples: list[str] = Field(default_factory=list)
    raw_balance_samples: list[str] = Field(default_factory=list)
    chosen_payout_source: Literal["ocr", "balance_delta", "template", "unknown"] = "unknown"
    chosen_bet_source: Literal["ocr", "unknown"] = "unknown"
    payout_resolution_attempts: int = 0
    payout_read_attempts: int = 0
    payout_read_success: bool = False
    payout_raw_attempts: list[dict[str, object]] = Field(default_factory=list)
    payout_stabilized_value_source: str | None = None
    bet_read_attempts: int = 0
    bet_read_success: bool = False
    bet_raw_attempts: list[dict[str, object]] = Field(default_factory=list)
    stabilization_fail_reason: str | None = None
    locked_session_bet: float | None = None
    bet_lock_acquired_at_spin: int | None = None
    bet_lock_source: str | None = None
    current_spin_raw_bet: float | None = None
    bet_mismatch_vs_lock: bool = False
    canonical_bet: float | None = None
    empty_spin: bool | None = None
    visual_win_by_bet: bool | None = None
    big_win: bool | None = None
    win_signal_detected: bool = False

    # visual_win: observed visual signal that resembles a win presentation on screen
    visual_win: bool | None = None
    # any_payout: payout > 0
    any_payout: bool | None = None
    # real_win: payout > bet
    real_win: bool | None = None
    # break_even: payout == bet
    break_even: bool | None = None
    net_loss_with_payout: bool | None = None
    # no_payout: payout == 0
    no_payout: bool | None = None
    # unreadable_result: unable to reliably determine payout/result from available signals
    result_class: ResultClass = "unreadable_result"
    classification_version: str = "v2"
    result_evidence: list[str] = Field(default_factory=list)
    payout_sampling_diagnostics: list[dict[str, object]] = Field(default_factory=list)
    balance_sampling_diagnostics: list[dict[str, object]] = Field(default_factory=list)
    bet_sampling_diagnostics: list[dict[str, object]] = Field(default_factory=list)

    detector_status: DetectorStatus = "partial"
    reason: str = "result_unknown_fallback"

    confidence_overall: float | None = None
    confidence_visual: float | None = None
    confidence_payout: float | None = None
    confidence_motion: float | None = None
    confidence_ready: float | None = None

    scatter_count: int | None = None
    bonus_count: int | None = None
    scatter_detect_ok: bool = False
    bonus_detect_ok: bool = False
    scatter_near_miss: bool = False
    bonus_tease: bool = False
    scatter_trigger_count: int | None = None
    bonus_trigger_count: int | None = None
    symbol_detection_frame_path: str | None = None
    symbol_detection_frame_ts: str | None = None
    scatter_boxes: list[dict[str, int]] | None = None
    bonus_boxes: list[dict[str, int]] | None = None
    scatter_match_scores: list[float] | None = None
    bonus_match_scores: list[float] | None = None
    symbol_detection_reason_flags: list[str] | None = None
    scatter_debug_template_present: bool | None = None
    scatter_debug_template_shape: list[int] | None = None
    scatter_debug_reels_shape: list[int] | None = None
    scatter_debug_best_score: float | None = None
    scatter_debug_best_loc: list[int] | None = None
    scatter_debug_threshold: float | None = None
    scatter_debug_frame_index: int | None = None
    scatter_debug_ran: bool = False
    scatter_debug_reason: str | None = None
    scatter_debug_reels_path: str | None = None

    fallback_used: bool = False
    timeouts: SpinTimeouts = Field(default_factory=SpinTimeouts)


def classify_spin_result(
    *,
    bet: float | None,
    payout: float | None,
    visual_win: bool | None,
    detector_status: DetectorStatus,
    reason: str,
    result_kind: Literal["win", "no_win", "win_unreadable"] | None = None,
    ready_recovered: bool = False,
    win_signal_detected: bool = False,
    payout_source: Literal["ocr", "balance_delta", "template", "unknown"] = "unknown",
    balance_delta: float | None = None,
) -> dict[str, object]:
    """Classify result with explicit final categories required for production summaries."""
    any_payout: bool | None = None
    real_win: bool | None = None
    break_even: bool | None = None
    net_loss_with_payout: bool | None = None
    no_payout: bool | None = None
    result_class: ResultClass = "unreadable_result"
    result_evidence: list[str] = []

    if payout is None:
        if detector_status == "timeout":
            result_class = "unreadable_result"
            result_evidence.append("timeout_without_payout")
        elif visual_win is True or win_signal_detected or result_kind == "win_unreadable":
            result_class = "probable_win"
            result_evidence.append("visual_win_without_numeric_payout")
        elif ready_recovered and result_kind == "no_win" and visual_win is not True:
            result_class = "confirmed_no_win"
            result_evidence.append("ready_recovered_no_win_no_positive_signal")
        else:
            result_class = "unreadable_result"
            result_evidence.append("insufficient_numeric_and_visual_evidence")
    elif payout == 0:
        any_payout = False
        no_payout = True
        result_class = "confirmed_no_win"
        result_evidence.append("payout_zero")
    else:
        any_payout = True
        no_payout = False
        result_class = "confirmed_win"
        result_evidence.append("payout_positive")
        if bet is not None:
            if payout > bet:
                real_win = True
                break_even = False
                net_loss_with_payout = False
            elif payout == bet:
                real_win = False
                break_even = True
                net_loss_with_payout = False
            else:
                real_win = False
                break_even = False
                net_loss_with_payout = True

    if payout_source == "balance_delta" and payout is not None and payout > 0:
        result_evidence.append("balance_delta_positive")
    if balance_delta is not None and balance_delta < 0 and payout is not None and payout > 0:
        result_class = "unreadable_result"
        result_evidence.append("contradictory_balance_delta")

    return {
        "visual_win": visual_win,
        "any_payout": any_payout,
        "real_win": real_win,
        "break_even": break_even,
        "net_loss_with_payout": net_loss_with_payout,
        "no_payout": no_payout,
        "result_class": result_class,
        "classification_version": "v2",
        "result_evidence": result_evidence,
        "detector_status": detector_status,
        "reason": reason,
    }
