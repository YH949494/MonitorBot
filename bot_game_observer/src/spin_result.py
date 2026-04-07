"""Spin result semantics and classification helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DetectorStatus = Literal["confirmed", "partial", "fallback", "timeout", "ambiguous"]
ResultClass = Literal["real_win", "break_even", "net_loss_with_payout", "no_payout", "result_unknown"]
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
    # result_unknown: unable to reliably determine payout/result from available signals
    result_class: ResultClass = "result_unknown"

    detector_status: DetectorStatus = "partial"
    reason: str = "result_unknown_fallback"

    confidence_overall: float | None = None
    confidence_visual: float | None = None
    confidence_payout: float | None = None
    confidence_motion: float | None = None
    confidence_ready: float | None = None

    fallback_used: bool = False
    timeouts: SpinTimeouts = Field(default_factory=SpinTimeouts)


def classify_spin_result(
    *,
    bet: float | None,
    payout: float | None,
    visual_win: bool | None,
    detector_status: DetectorStatus,
    reason: str,
) -> dict[str, object]:
    """Classify result without inferring no_payout from missing/failed detectors."""
    any_payout: bool | None = None
    real_win: bool | None = None
    break_even: bool | None = None
    net_loss_with_payout: bool | None = None
    no_payout: bool | None = None
    result_class: ResultClass = "result_unknown"

    if payout is None:
        result_class = "result_unknown"
    elif payout == 0:
        any_payout = False
        no_payout = True
        result_class = "no_payout"
    else:
        any_payout = True
        no_payout = False
        if bet is None:
            result_class = "result_unknown"
        elif payout > bet:
            real_win = True
            break_even = False
            net_loss_with_payout = False
            result_class = "real_win"
        elif payout == bet:
            real_win = False
            break_even = True
            net_loss_with_payout = False
            result_class = "break_even"
        else:
            real_win = False
            break_even = False
            net_loss_with_payout = True
            result_class = "net_loss_with_payout"

    return {
        "visual_win": visual_win,
        "any_payout": any_payout,
        "real_win": real_win,
        "break_even": break_even,
        "net_loss_with_payout": net_loss_with_payout,
        "no_payout": no_payout,
        "result_class": result_class,
        "detector_status": detector_status,
        "reason": reason,
    }
