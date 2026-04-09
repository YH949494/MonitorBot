"""Pure analytics from ordered session events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import SessionEventType, SessionSummary


def parse_event_type(raw: str) -> str:
    return raw


def build_summary(session_id: str, events: list[dict[str, Any]]) -> SessionSummary:
    """
    Compute ``SessionSummary`` from JSONL-like dicts (must include event_type, ts).
    Uses semantic events emitted by the runner.
    """
    started_at = _first_ts(events) or datetime.now(timezone.utc)
    ended_at = _last_ts(events)

    expected_spin_count = 0
    visual_win_count = 0
    any_payout_count = 0
    real_win_count = 0
    break_even_count = 0
    net_loss_with_payout_count = 0
    no_payout_count = 0
    result_unknown_count = 0
    click_to_spinning_timeout_count = 0
    spinning_to_result_timeout_count = 0
    result_to_ready_timeout_count = 0
    near_miss = 0
    bonus_tease = 0
    bonus_trig = 0
    empty_spin_count = 0
    visual_win_by_bet_count = 0
    visual_win_spin_indices: list[int] = []
    big_win_count = 0
    big_win_spin_indices: list[int] = []
    missing_payout_count = 0
    payout_truth_conflict_count = 0
    ocr_balance_agree_count = 0
    balance_delta_confirmed_count = 0
    payout_effective_resolved_count = 0
    payout_effective_unresolved_count = 0
    balance_backed_payout_count = 0
    ocr_only_payout_count = 0
    conflict_spin_indices: list[int] = []
    unresolved_spin_indices: list[int] = []
    consecutive_conflict_spins_max = 0
    consecutive_unresolved_spins_max = 0
    _conflict_streak = 0
    _unresolved_streak = 0
    usable_spin_count = 0
    locked_session_bet: float | None = None
    end_reason = "unknown"
    conf_notes: list[str] = []
    warnings: list[str] = []

    gap_tracking: list[int] = []
    finalized_outcomes_raw: list[tuple[int, str, int]] = []
    spin_payloads_raw: list[tuple[int, dict[str, Any], int]] = []

    spin_events = [
        e for e in events if str(e.get("event_type", "")) in (SessionEventType.SPIN_RESULT_SUMMARY.value, "spin_result_summary")
    ]

    for e in events:
        et = str(e.get("event_type", ""))
        payload = e.get("payload") or {}

        if et == SessionEventType.SPIN_STARTED.value or et == "spin_started":
            expected_spin_count += 1
        if et == SessionEventType.NEAR_MISS_DETECTED.value or et == "near_miss_detected":
            near_miss += 1
        if et == SessionEventType.BONUS_TEASE_DETECTED.value or et == "bonus_tease_detected":
            bonus_tease += 1
        if et == SessionEventType.BONUS_TRIGGERED.value or et == "bonus_triggered":
            bonus_trig += 1
        if et == SessionEventType.SESSION_STOPPED.value or et == "session_stopped":
            end_reason = payload.get("reason", end_reason)
        if et == SessionEventType.STATE_TRANSITION.value or et == "state_transition":
            c = payload.get("confidence")
            if isinstance(c, (int, float)) and c < 0.5:
                conf_notes.append(
                    f"low_confidence_transition:{payload.get('to')}:{c:.2f}"
                )

    for spin_order, e in enumerate(spin_events, start=1):
        payload = e.get("payload") or {}
        spin_index = payload.get("spin_index")
        normalized_spin_index = spin_index if isinstance(spin_index, int) else spin_order
        spin_payloads_raw.append((normalized_spin_index, payload, spin_order))
        result_kind = payload.get("result_kind")
        if result_kind in ("win", "win_unreadable", "no_win"):
            finalized_outcomes_raw.append((normalized_spin_index, str(result_kind), spin_order))
        if payload.get("visual_win") is True:
            visual_win_count += 1
        if payload.get("visual_win_by_bet") is True:
            visual_win_by_bet_count += 1
            spin_index = payload.get("spin_index")
            if isinstance(spin_index, int):
                visual_win_spin_indices.append(spin_index)
        elif "visual_win_by_bet" not in payload and payload.get("visual_win") is True:
            visual_win_by_bet_count += 1
            spin_index = payload.get("spin_index")
            if isinstance(spin_index, int):
                visual_win_spin_indices.append(spin_index)
        if payload.get("big_win") is True:
            big_win_count += 1
            spin_index = payload.get("spin_index")
            if isinstance(spin_index, int):
                big_win_spin_indices.append(spin_index)
        if payload.get("result_kind") == "no_win":
            empty_spin_count += 1
        result_kind = payload.get("result_kind")
        effective_payout_value = payload.get("payout_effective_value", payload.get("payout"))
        if effective_payout_value is not None or result_kind == "no_win":
            # Conservative usable-spin rule: resolved effective payout OR explicit finalized no-win.
            usable_spin_count += 1
        if payload.get("payout_truth_conflict") is True:
            conflict_spin_indices.append(normalized_spin_index)
            _conflict_streak += 1
            consecutive_conflict_spins_max = max(consecutive_conflict_spins_max, _conflict_streak)
        else:
            _conflict_streak = 0
        if effective_payout_value is None:
            missing_payout_count += 1
            unresolved_spin_indices.append(normalized_spin_index)
            _unresolved_streak += 1
            consecutive_unresolved_spins_max = max(consecutive_unresolved_spins_max, _unresolved_streak)
        else:
            _unresolved_streak = 0
        if payload.get("payout_truth_conflict") is True:
            payout_truth_conflict_count += 1
        if payload.get("payout_truth_source") == "ocr_balance_agree":
            ocr_balance_agree_count += 1
        if payload.get("payout_truth_source") == "balance_delta_confirmed":
            balance_delta_confirmed_count += 1
        effective_source = payload.get("payout_effective_source")
        if effective_payout_value is None:
            payout_effective_unresolved_count += 1
        else:
            payout_effective_resolved_count += 1
        if effective_source in ("balance_delta_confirmed", "ocr_balance_conflict"):
            balance_backed_payout_count += 1
        if effective_source == "ocr_confirmed":
            ocr_only_payout_count += 1
        if locked_session_bet is None and isinstance(payload.get("locked_session_bet"), (int, float)):
            locked_session_bet = float(payload["locked_session_bet"])
        if effective_payout_value is not None and effective_payout_value > 0:
            any_payout_count += 1
        rc = payload.get("result_class")
        if rc == "real_win":
            real_win_count += 1
        elif rc == "break_even":
            break_even_count += 1
        elif rc == "net_loss_with_payout":
            net_loss_with_payout_count += 1
        elif rc == "no_payout":
            no_payout_count += 1
        elif rc == "result_unknown":
            result_unknown_count += 1
        tmo = payload.get("timeouts") or {}
        if tmo.get("click_to_spinning"):
            click_to_spinning_timeout_count += 1
        if tmo.get("spinning_to_result"):
            spinning_to_result_timeout_count += 1
        if tmo.get("result_to_ready"):
            result_to_ready_timeout_count += 1

    finalized_outcomes = _normalize_finalized_outcomes(finalized_outcomes_raw, warnings)
    spin_payloads = _normalize_spin_payloads(spin_payloads_raw)
    total_spins = len(spin_payloads)
    coverage_ratio: float | None = None
    if expected_spin_count > 0:
        coverage_ratio = max(0.0, min(1.0, total_spins / expected_spin_count))
    if coverage_ratio is not None and 0.80 <= coverage_ratio < 0.95:
        warnings.append("Session coverage degraded — use with caution")
    readable_win_spins = [spin_idx for spin_idx, kind in finalized_outcomes if kind == "win"]
    first_finalized_non_no_win = next(
        (spin_idx for spin_idx, kind in finalized_outcomes if kind != "no_win"),
        None,
    )
    total_wins = len(readable_win_spins)
    total_no_wins = sum(1 for _spin_idx, kind in finalized_outcomes if kind == "no_win")
    unreadable_win_count = sum(1 for _spin_idx, kind in finalized_outcomes if kind == "win_unreadable")
    finalized_non_no_win_count = sum(1 for _spin_idx, kind in finalized_outcomes if kind != "no_win")
    first_readable_win = readable_win_spins[0] if readable_win_spins else None
    first_win = first_readable_win
    first_win_spin_for_warning = first_finalized_non_no_win
    gap_tracking = []
    for idx in range(1, len(readable_win_spins)):
        gap_tracking.append(readable_win_spins[idx] - readable_win_spins[idx - 1])
    sbfw = (first_win - 1) if first_win is not None else None

    avg_gap = None
    max_gap = None
    if gap_tracking:
        avg_gap = sum(gap_tracking) / len(gap_tracking)
        max_gap = max(gap_tracking)

    duration_sec = 0.0
    if started_at and ended_at:
        duration_sec = max(0.0, (ended_at - started_at).total_seconds())

    total_decisions = total_wins + total_no_wins  # approximate
    nm_rate = (near_miss / total_decisions) if total_decisions else 0.0

    streak_max = _max_no_win_streak_from_outcomes(finalized_outcomes)

    if total_spins == 0:
        warnings.append("No spins recorded; check calibration and motion thresholds.")
    if first_win_spin_for_warning is None and total_spins > 5:
        warnings.append("No wins detected; verify win_banner template and region.")

    primary_spins = total_spins
    any_payout_rate = (any_payout_count / primary_spins) if primary_spins else 0.0
    real_win_rate = (real_win_count / primary_spins) if primary_spins else 0.0
    empty_spin_rate = (empty_spin_count / primary_spins) if primary_spins else 0.0
    visual_win_by_bet_rate = (visual_win_by_bet_count / primary_spins) if primary_spins else 0.0
    big_win_rate = (big_win_count / primary_spins) if primary_spins else 0.0
    missing_payout_rate = (missing_payout_count / primary_spins) if primary_spins else 0.0
    payout_conflict_count = payout_truth_conflict_count
    session_trust_score: float | None = None
    session_trust_label: str | None = None
    session_quality: str | None = None
    session_valid_for_analysis = True
    session_exclusion_reason: str | None = None
    if primary_spins > 0:
        conflict_rate = payout_conflict_count / primary_spins
        unresolved_rate = payout_effective_unresolved_count / primary_spins
        agreement_rate = ocr_balance_agree_count / primary_spins
        balance_backed_rate = balance_backed_payout_count / primary_spins
        trust = 1.0
        trust -= (0.60 * conflict_rate)
        trust -= (0.60 * unresolved_rate)
        trust += (0.10 * agreement_rate)
        trust += (0.05 * balance_backed_rate)
        session_trust_score = max(0.0, min(1.0, trust))
        if session_trust_score >= 0.75:
            session_trust_label = "high"
        elif session_trust_score >= 0.45:
            session_trust_label = "medium"
        else:
            session_trust_label = "low"
    anomaly_threshold_exceeded = (
        consecutive_unresolved_spins_max >= 5
        or consecutive_conflict_spins_max >= 4
    )
    if coverage_ratio is None:
        session_quality = "invalid"
        session_valid_for_analysis = False
        session_exclusion_reason = "missing_expected_spin_count"
    elif coverage_ratio < 0.80:
        session_quality = "invalid"
        session_valid_for_analysis = False
        session_exclusion_reason = "low_coverage_ratio"
    elif coverage_ratio < 0.95:
        session_quality = "degraded"
        session_valid_for_analysis = True
        session_exclusion_reason = "degraded_coverage"
    else:
        session_quality = "valid"
        session_valid_for_analysis = True
        session_exclusion_reason = None
    if anomaly_threshold_exceeded:
        session_quality = "invalid"
        session_valid_for_analysis = False
        session_exclusion_reason = "anomaly_threshold_exceeded"
    elif session_trust_score is not None and session_trust_score < 0.40:
        session_quality = "invalid"
        session_valid_for_analysis = False
        session_exclusion_reason = "low_trust_score"
        warnings.append("Low trust session — exclude from primary analysis")
    elif (
        session_quality == "valid"
        and session_trust_score is not None
        and session_trust_score < 0.55
    ):
        session_quality = "degraded"
        session_exclusion_reason = "degraded_coverage"

    return SessionSummary(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration_sec,
        total_spins=primary_spins,
        total_wins=total_wins,
        total_no_win=total_no_wins,
        unreadable_win_count=unreadable_win_count,
        finalized_non_no_win_count=finalized_non_no_win_count,
        any_payout_count=any_payout_count,
        real_win_count=real_win_count,
        break_even_count=break_even_count,
        net_loss_with_payout_count=net_loss_with_payout_count,
        no_payout_count=no_payout_count,
        result_unknown_count=result_unknown_count,
        click_to_spinning_timeout_count=click_to_spinning_timeout_count,
        spinning_to_result_timeout_count=spinning_to_result_timeout_count,
        result_to_ready_timeout_count=result_to_ready_timeout_count,
        any_payout_rate=any_payout_rate,
        real_win_rate=real_win_rate,
        first_win_spin_index=first_win,
        first_readable_win_spin_index=first_readable_win,
        first_finalized_non_no_win_spin_index=first_finalized_non_no_win,
        spins_before_first_win=sbfw,
        gaps_between_wins=gap_tracking,
        avg_spins_between_wins=avg_gap,
        max_spins_between_wins=max_gap,
        near_miss_count=near_miss,
        near_miss_rate=nm_rate,
        empty_spin_count=empty_spin_count,
        empty_spin_rate=empty_spin_rate,
        visual_win_count=visual_win_by_bet_count,
        visual_win_rate=visual_win_by_bet_rate,
        visual_win_spin_indices=visual_win_spin_indices,
        big_win_count=big_win_count,
        big_win_rate=big_win_rate,
        big_win_spin_indices=big_win_spin_indices,
        missing_payout_count=missing_payout_count,
        missing_payout_rate=missing_payout_rate,
        payout_truth_conflict_count=payout_truth_conflict_count,
        ocr_balance_agree_count=ocr_balance_agree_count,
        balance_delta_confirmed_count=balance_delta_confirmed_count,
        payout_effective_resolved_count=payout_effective_resolved_count,
        payout_effective_unresolved_count=payout_effective_unresolved_count,
        balance_backed_payout_count=balance_backed_payout_count,
        ocr_only_payout_count=ocr_only_payout_count,
        payout_conflict_count=payout_conflict_count,
        session_trust_score=session_trust_score,
        session_trust_label=session_trust_label,
        coverage_ratio=coverage_ratio,
        session_quality=session_quality,
        usable_spin_count=usable_spin_count,
        session_valid_for_analysis=session_valid_for_analysis,
        session_exclusion_reason=session_exclusion_reason,
        conflict_spin_indices=conflict_spin_indices,
        unresolved_spin_indices=unresolved_spin_indices,
        consecutive_conflict_spins_max=consecutive_conflict_spins_max,
        consecutive_unresolved_spins_max=consecutive_unresolved_spins_max,
        locked_session_bet=locked_session_bet,
        bonus_tease_count=bonus_tease,
        bonus_trigger_count=bonus_trig,
        end_reason=end_reason,
        consecutive_no_win_streak_max=streak_max,
        confidence_notes=conf_notes[:50],
        warnings=warnings,
    )


def _first_ts(events: list[dict[str, Any]]) -> datetime | None:
    for e in events:
        ts = e.get("ts")
        if ts:
            return _parse_dt(ts)
    return None


def _last_ts(events: list[dict[str, Any]]) -> datetime | None:
    for e in reversed(events):
        ts = e.get("ts")
        if ts:
            return _parse_dt(ts)
    return None


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        # ISO8601
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    raise ValueError(f"Bad timestamp: {val!r}")


def _max_no_win_streak(events: list[dict[str, Any]]) -> int:
    """Count max consecutive no-win results between wins (from ordered semantic events)."""
    streak = 0
    best = 0
    for e in events:
        et = str(e.get("event_type", ""))
        if et in ("win_detected", SessionEventType.WIN_DETECTED.value):
            streak = 0
        elif et in ("no_win_detected", SessionEventType.NO_WIN_DETECTED.value):
            streak += 1
            best = max(best, streak)
    return best


def _max_no_win_streak_from_outcomes(outcomes: list[tuple[int, str]]) -> int:
    streak = 0
    best = 0
    for _spin_index, result_kind in outcomes:
        if result_kind == "no_win":
            streak += 1
            best = max(best, streak)
        elif result_kind == "win":
            streak = 0
    return best


def _normalize_finalized_outcomes(
    outcomes_raw: list[tuple[int, str, int]],
    warnings: list[str],
) -> list[tuple[int, str]]:
    if not outcomes_raw:
        return []
    ordered = sorted(outcomes_raw, key=lambda item: (item[0], item[2]))
    deduped: list[tuple[int, str]] = []
    seen_spin_indices: set[int] = set()
    duplicate_count = 0
    for spin_index, result_kind, _order in ordered:
        if spin_index in seen_spin_indices:
            duplicate_count += 1
            continue
        seen_spin_indices.add(spin_index)
        deduped.append((spin_index, result_kind))
    if duplicate_count > 0:
        warnings.append(
            f"Duplicate finalized spin_result_summary indices detected: {duplicate_count}; first record kept."
        )
    return deduped


def _normalize_spin_payloads(
    spin_payloads_raw: list[tuple[int, dict[str, Any], int]],
) -> list[dict[str, Any]]:
    if not spin_payloads_raw:
        return []
    ordered = sorted(spin_payloads_raw, key=lambda item: (item[0], item[2]))
    deduped: list[dict[str, Any]] = []
    seen_spin_indices: set[int] = set()
    for spin_index, payload, _order in ordered:
        if spin_index in seen_spin_indices:
            continue
        seen_spin_indices.add(spin_index)
        deduped.append(payload)
    return deduped
