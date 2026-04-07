"""Deterministic state machine with debounced transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .models import BotState


@dataclass
class FrameSignals:
    """Per-frame semantic inputs derived from vision (already thresholded)."""

    ts: datetime
    frame_index: int
    motion_score: float
    reels_spinning: bool
    reels_stopped: bool
    popup: bool
    win: bool
    no_win_hint: bool
    bonus_tease: bool
    bonus_trigger: bool
    near_miss: bool
    session_end: bool
    spin_button_ready: bool
    post_result_ready_fallback: bool = False
    confidences: dict[str, float] = field(default_factory=dict)


@dataclass
class TransitionRecord:
    from_state: BotState
    to_state: BotState
    reason: str
    confidence: float
    ts: datetime
    frame_index: int
    detail: dict[str, Any] = field(default_factory=dict)


class GameStateMachine:
    """
    Priority-ordered guards:
    session end > popup > bonus trigger > bonus tease > spinning flow > results.
    """

    def __init__(
        self,
        debounce_frames: int = 2,
        min_confidence: float = 0.35,
    ) -> None:
        self.debounce_frames = max(1, debounce_frames)
        self.min_confidence = min_confidence
        self.state: BotState = BotState.IDLE
        self._pending: BotState | None = None
        self._pending_count: int = 0
        self._last_transition_confidence: float = 1.0

    def reset(self) -> None:
        self.state = BotState.IDLE
        self._pending = None
        self._pending_count = 0

    def _propose_next(self, sig: FrameSignals) -> tuple[BotState, float, str]:
        """Return desired state, confidence, reason (single step heuristic)."""
        c = sig.confidences

        def conf(name: str, default: float = 0.5) -> float:
            return float(c.get(name, default))

        if sig.session_end:
            return BotState.SESSION_ENDED, conf("session_end", 0.9), "session_end_template"

        if self.state == BotState.POPUP_BLOCKING:
            if not sig.popup:
                return BotState.READY_TO_SPIN, conf("spin_ready", 0.55), "popup_cleared"
            return BotState.POPUP_BLOCKING, conf("popup", 0.85), "popup_still_open"

        if sig.popup:
            return BotState.POPUP_BLOCKING, conf("popup", 0.85), "popup_template"

        if sig.bonus_trigger:
            return BotState.BONUS_TRIGGERED, conf("bonus_trigger", 0.85), "bonus_trigger_template"
        if sig.bonus_tease and not sig.reels_spinning:
            return BotState.BONUS_TEASE, conf("bonus_tease", 0.75), "bonus_tease_template"

        if sig.reels_spinning and self.state not in (
            BotState.RESULT_WIN,
            BotState.RESULT_NO_WIN,
        ):
            return BotState.SPINNING, max(conf("motion", 0.7), 0.6), "reel_motion"

        # Stopped reels: resolve outcome after a spin
        if sig.reels_stopped:
            if self.state == BotState.SPINNING:
                if sig.win:
                    return BotState.RESULT_WIN, conf("win", 0.85), "win_template"
                if sig.near_miss:
                    return BotState.RESULT_NO_WIN, conf("near_miss", 0.7), "near_miss_template"
                return BotState.RESULT_NO_WIN, conf("no_win", 0.55), "result_unknown_fallback"

        if sig.spin_button_ready and self.state in (
            BotState.IDLE,
            BotState.RESULT_WIN,
            BotState.RESULT_NO_WIN,
            BotState.BONUS_TEASE,
            BotState.BONUS_TRIGGERED,
        ):
            return BotState.READY_TO_SPIN, conf("spin_ready", 0.65), "spin_button_ready"

        if self.state in (BotState.RESULT_WIN, BotState.RESULT_NO_WIN):
            if sig.spin_button_ready:
                return BotState.READY_TO_SPIN, conf("spin_ready", 0.6), "post_result_ready"
            if sig.post_result_ready_fallback:
                return BotState.READY_TO_SPIN, conf("post_result_recovery", 0.45), "post_result_recovery_fallback"

        if self.state == BotState.IDLE:
            return BotState.READY_TO_SPIN, 0.5, "bootstrap_ready"

        return self.state, 0.4, "hold"

    def _debounced(self, proposed: BotState) -> bool:
        if proposed == self._pending:
            self._pending_count += 1
        else:
            self._pending = proposed
            self._pending_count = 1
        return self._pending_count >= self.debounce_frames

    def update(self, sig: FrameSignals) -> list[TransitionRecord]:
        """Process one frame; emit zero or one transition."""
        proposed, confidence, reason = self._propose_next(sig)

        if confidence < self.min_confidence and proposed != self.state:
            # Uncertain: do not transition to new semantic state; keep internal hold
            return []

        if proposed == self.state:
            self._pending = None
            self._pending_count = 0
            self._last_transition_confidence = confidence
            return []

        if not self._debounced(proposed):
            return []

        if proposed == self.state:
            return []

        rec = TransitionRecord(
            from_state=self.state,
            to_state=proposed,
            reason=reason,
            confidence=confidence,
            ts=sig.ts,
            frame_index=sig.frame_index,
            detail={"motion": sig.motion_score},
        )
        self.state = proposed
        self._pending = None
        self._pending_count = 0
        self._last_transition_confidence = confidence
        return [rec]
