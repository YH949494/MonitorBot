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

    spins = 0
    wins: list[int] = []
    no_wins = 0
    near_miss = 0
    bonus_tease = 0
    bonus_trig = 0
    end_reason = "unknown"
    current_spin = 0
    conf_notes: list[str] = []
    warnings: list[str] = []

    gap_tracking: list[int] = []
    last_win_spin: int | None = None

    for e in events:
        et = str(e.get("event_type", ""))
        payload = e.get("payload") or {}

        if et == SessionEventType.SPIN_STARTED.value or et == "spin_started":
            spins += 1
            current_spin = spins

        if et == SessionEventType.WIN_DETECTED.value or et == "win_detected":
            win_spin = current_spin or spins
            wins.append(win_spin)
            if last_win_spin is not None:
                gap_tracking.append(win_spin - last_win_spin)
            last_win_spin = win_spin

        if et == SessionEventType.NO_WIN_DETECTED.value or et == "no_win_detected":
            no_wins += 1

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

    first_win = wins[0] if wins else None
    sbfw = (first_win - 1) if first_win is not None else None

    avg_gap = None
    max_gap = None
    if gap_tracking:
        avg_gap = sum(gap_tracking) / len(gap_tracking)
        max_gap = max(gap_tracking)

    duration_sec = 0.0
    if started_at and ended_at:
        duration_sec = max(0.0, (ended_at - started_at).total_seconds())

    total_decisions = len(wins) + no_wins  # approximate
    nm_rate = (near_miss / total_decisions) if total_decisions else 0.0

    # streak: approximate from alternating win/no_win events order — simplified
    streak_max = _max_no_win_streak(events)

    if spins == 0:
        warnings.append("No spins recorded; check calibration and motion thresholds.")
    if not wins and spins > 5:
        warnings.append("No wins detected; verify win_banner template and region.")

    return SessionSummary(
        session_id=session_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_sec=duration_sec,
        total_spins=spins,
        total_wins=len(wins),
        total_no_win=no_wins,
        first_win_spin_index=first_win,
        spins_before_first_win=sbfw,
        gaps_between_wins=gap_tracking,
        avg_spins_between_wins=avg_gap,
        max_spins_between_wins=max_gap,
        near_miss_count=near_miss,
        near_miss_rate=nm_rate,
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
