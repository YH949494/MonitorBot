from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import sys
from types import SimpleNamespace
import types

import numpy as np

if "pyautogui" not in sys.modules:
    sys.modules["pyautogui"] = types.SimpleNamespace(
        moveTo=lambda *_args, **_kwargs: None,
        click=lambda *_args, **_kwargs: None,
    )
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(imwrite=lambda *_args, **_kwargs: True, __version__="4.0.0")

from src.models import BotState, FramePacket, SessionEventType
from src.session_runner import SessionRunner
from src.spin_result import SpinResult
from src.state_machine import FrameSignals


def _make_sig(*, motion: float, spin_button_ready: bool, bonus_trigger: bool = False) -> FrameSignals:
    return FrameSignals(
        ts=datetime.now(timezone.utc),
        frame_index=1,
        motion_score=motion,
        reels_spinning=motion > 0,
        reels_stopped=motion <= 0,
        popup=False,
        win=False,
        no_win_hint=True,
        bonus_tease=False,
        bonus_trigger=bonus_trigger,
        near_miss=False,
        session_end=False,
        spin_button_ready=spin_button_ready,
        confidences={"motion": 0.9},
    )


def _make_runner() -> tuple[SessionRunner, list[tuple[SessionEventType, dict]]]:
    runner = SessionRunner.__new__(SessionRunner)
    runner.settings = SimpleNamespace(
        detection=SimpleNamespace(result_to_ready_timeout_sec=4.0, result_animation_timeout_sec=12.0),
        regions=SimpleNamespace(spin_button=None),
    )
    runner.sm = SimpleNamespace(state=BotState.POST_RESULT_ANIMATION)
    runner._active_spin = SpinResult(spin_index=1, visual_win=True, reason="awaiting_payout_resolution")
    runner._awaiting_ready_since = 100.0
    runner._awaiting_spinning_since = None
    runner._spinning_since = None
    runner._post_result_animation_since = 102.0
    runner._result_ready_debug_frames = deque([{"motion_score": 18.1}, {"motion_score": 20.2}], maxlen=10)
    runner._armed_for_click = False
    runner._result_recovery_stable_frames = 0
    runner._result_spin_button_ready_evidence_saved = False
    runner._post_result_animation_reason = None
    runner._spin_counter = 1
    runner._save_evidence_crop = lambda *_args, **_kwargs: None
    events: list[tuple[SessionEventType, dict]] = []
    runner._emit = lambda event_type, payload: events.append((event_type, payload))
    runner._finalize_spin_result = lambda **kwargs: events.append((SessionEventType.SPIN_RESULT_SUMMARY, kwargs))
    return runner, events


def test_post_result_animation_timeout_uses_new_timeout(monkeypatch) -> None:
    runner, events = _make_runner()
    frame = FramePacket(
        ts=datetime.now(timezone.utc),
        frame_index=2,
        image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    sig = _make_sig(motion=21.0, spin_button_ready=False)

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 109.0)
    runner._check_spin_timeouts(sig, frame)
    assert not any(payload.get("reason") == "post_result_animation_timeout" for _et, payload in events if isinstance(payload, dict))

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 115.5)
    runner._check_spin_timeouts(sig, frame)
    assert any(payload.get("reason") == "post_result_animation_timeout" for _et, payload in events if isinstance(payload, dict))


def test_bonus_trigger_sets_post_result_animation_reason() -> None:
    runner, events = _make_runner()
    rec = [
        SimpleNamespace(
            from_state=BotState.POST_RESULT_ANIMATION,
            to_state=BotState.BONUS_TRIGGERED,
            reason="bonus_trigger_template",
            confidence=0.95,
            frame_index=3,
            ts=datetime.now(timezone.utc),
            detail={"motion": 19.0},
        )
    ]
    runner._handle_transitions(rec)
    assert runner._active_spin is not None
    assert runner._active_spin.reason == "bonus_feature_animation"
    assert any(payload.get("type") == "post_result_animation_bonus_feature" for _et, payload in events)
