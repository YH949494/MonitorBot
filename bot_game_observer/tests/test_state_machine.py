"""State machine debounce and transition tests."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models import BotState
from src.state_machine import FrameSignals, GameStateMachine


def _sig(**kwargs) -> FrameSignals:
    base = dict(
        ts=datetime.now(timezone.utc),
        frame_index=0,
        motion_score=0.0,
        reels_spinning=False,
        reels_stopped=True,
        popup=False,
        win=False,
        no_win_hint=True,
        bonus_tease=False,
        bonus_trigger=False,
        near_miss=False,
        session_end=False,
        spin_button_ready=True,
        confidences={},
    )
    base.update(kwargs)
    return FrameSignals(**base)


def test_debounce_requires_multiple_frames() -> None:
    sm = GameStateMachine(debounce_frames=2, min_confidence=0.1)
    # IDLE -> READY needs two identical proposals
    r1 = sm.update(_sig(spin_button_ready=True, reels_stopped=True))
    assert r1 == []
    r2 = sm.update(_sig(spin_button_ready=True, reels_stopped=True))
    assert len(r2) == 1
    assert r2[0].to_state == BotState.READY_TO_SPIN


def test_spinning_transition() -> None:
    sm = GameStateMachine(debounce_frames=1, min_confidence=0.1)
    sm.state = BotState.READY_TO_SPIN
    rec = sm.update(
        _sig(
            reels_spinning=True,
            reels_stopped=False,
            motion_score=20.0,
            confidences={"motion": 0.9},
        )
    )
    assert rec and rec[0].to_state == BotState.SPINNING


def test_popup_priority() -> None:
    sm = GameStateMachine(debounce_frames=1, min_confidence=0.1)
    sm.state = BotState.SPINNING
    rec = sm.update(_sig(popup=True, reels_spinning=True, confidences={"popup": 0.9}))
    assert rec[0].to_state == BotState.POPUP_BLOCKING


def test_post_result_fallback_recovers_ready_without_template_match() -> None:
    sm = GameStateMachine(debounce_frames=1, min_confidence=0.1)
    sm.state = BotState.READY_TO_SPIN
    rec = sm.update(
        _sig(
            reels_spinning=True,
            reels_stopped=False,
            motion_score=20.0,
            spin_button_ready=False,
            confidences={"motion": 0.9},
        )
    )
    assert rec and rec[0].to_state == BotState.SPINNING

    rec = sm.update(
        _sig(
            reels_spinning=False,
            reels_stopped=True,
            spin_button_ready=False,
            win=False,
            near_miss=False,
            confidences={"no_win": 0.8},
        )
    )
    assert rec and rec[0].to_state == BotState.RESULT_NO_WIN

    rec = sm.update(
        _sig(
            reels_spinning=False,
            reels_stopped=True,
            popup=False,
            spin_button_ready=False,
            post_result_ready_fallback=True,
            confidences={"post_result_recovery": 0.45},
        )
    )
    assert rec and rec[0].to_state == BotState.READY_TO_SPIN
    assert rec[0].reason == "post_result_recovery_fallback"


def test_result_enters_post_result_animation_when_motion_persists() -> None:
    sm = GameStateMachine(debounce_frames=1, min_confidence=0.1)
    sm.state = BotState.RESULT_WIN
    rec = sm.update(
        _sig(
            reels_spinning=True,
            reels_stopped=False,
            motion_score=20.0,
            popup=False,
            spin_button_ready=False,
            confidences={"motion": 0.9},
        )
    )
    assert rec and rec[0].to_state == BotState.POST_RESULT_ANIMATION
    assert rec[0].reason == "post_result_animation_motion"


def test_post_result_animation_recovers_to_ready() -> None:
    sm = GameStateMachine(debounce_frames=1, min_confidence=0.1)
    sm.state = BotState.POST_RESULT_ANIMATION
    rec = sm.update(
        _sig(
            reels_spinning=False,
            reels_stopped=True,
            spin_button_ready=True,
            confidences={"spin_ready": 0.8},
        )
    )
    assert rec and rec[0].to_state == BotState.READY_TO_SPIN
