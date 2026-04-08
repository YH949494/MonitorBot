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
        detection=SimpleNamespace(
            spinning_motion_threshold=12.0,
            result_to_ready_timeout_sec=4.0,
            result_animation_timeout_sec=12.0,
            post_result_normal_threshold_sec=1.0,
            post_result_long_animation_threshold_sec=3.0,
            post_result_bonus_like_threshold_sec=6.0,
            payout_read_delay_sec=0.25,
            payout_read_retry_window_sec=1.0,
            payout_read_max_attempts=5,
            ocr_lang="eng",
            use_ocr_balance=False,
        ),
        regions=SimpleNamespace(spin_button=None, bet_text=None, payout_text=None, balance_text=None),
    )
    runner.sm = SimpleNamespace(state=BotState.POST_RESULT_ANIMATION)
    runner._active_spin = SpinResult(spin_index=1, visual_win=True, reason="awaiting_payout_resolution")
    runner._awaiting_ready_since = 100.0
    runner._awaiting_spinning_since = None
    runner._spinning_since = None
    runner._post_result_animation_since = 102.0
    runner._payout_samples = deque(maxlen=3)
    runner._balance_samples = deque(maxlen=3)
    runner._result_evidence_saved = False
    runner._finalize_on_ready = False
    runner._result_ready_debug_frames = deque([{"motion_score": 18.1}, {"motion_score": 20.2}], maxlen=10)
    runner._armed_for_click = False
    runner._result_recovery_stable_frames = 0
    runner._result_spin_button_ready_evidence_saved = False
    runner._post_result_animation_reason = None
    runner._result_detected_mono = 100.0
    runner._spin_counter = 1
    runner._save_evidence_crop = lambda *_args, **_kwargs: None
    events: list[tuple[SessionEventType, dict]] = []
    runner._emit = lambda event_type, payload: events.append((event_type, payload))
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


def test_post_result_visual_short_win_is_normal_result(monkeypatch) -> None:
    runner, _events = _make_runner()
    rec = [
        SimpleNamespace(
            from_state=BotState.POST_RESULT_ANIMATION,
            to_state=BotState.READY_TO_SPIN,
            reason="spin_button_ready",
            confidence=0.95,
            frame_index=4,
            ts=datetime.now(timezone.utc),
            detail={},
        )
    ]
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 101.2)
    runner._handle_transitions(rec)
    assert runner._active_spin is not None
    assert runner._active_spin.post_result_animation_duration_sec == 1.2
    assert runner._active_spin.post_result_visual_classification == "normal_result"


def test_post_result_visual_long_win_animation(monkeypatch) -> None:
    runner, _events = _make_runner()
    rec = [
        SimpleNamespace(
            from_state=BotState.POST_RESULT_ANIMATION,
            to_state=BotState.READY_TO_SPIN,
            reason="spin_button_ready",
            confidence=0.95,
            frame_index=4,
            ts=datetime.now(timezone.utc),
            detail={},
        )
    ]
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 104.2)
    runner._handle_transitions(rec)
    assert runner._active_spin is not None
    assert runner._active_spin.post_result_visual_classification == "long_animation"


def test_post_result_visual_extra_long_or_bonus_like(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._post_result_animation_reason = "bonus_feature_animation"
    rec = [
        SimpleNamespace(
            from_state=BotState.POST_RESULT_ANIMATION,
            to_state=BotState.READY_TO_SPIN,
            reason="spin_button_ready",
            confidence=0.95,
            frame_index=4,
            ts=datetime.now(timezone.utc),
            detail={},
        )
    ]
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 106.5)
    runner._handle_transitions(rec)
    assert runner._active_spin is not None
    assert runner._active_spin.post_result_visual_classification == "bonus_like"


def test_post_result_visual_no_win_quick_recovery_is_none(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=2, visual_win=False, result_kind="no_win")
    runner._awaiting_ready_since = 100.0
    rec = [
        SimpleNamespace(
            from_state=BotState.RESULT_NO_WIN,
            to_state=BotState.READY_TO_SPIN,
            reason="spin_button_ready",
            confidence=0.95,
            frame_index=4,
            ts=datetime.now(timezone.utc),
            detail={},
        )
    ]
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.4)
    runner._handle_transitions(rec)
    assert runner._active_spin is not None
    assert runner._active_spin.post_result_visual_classification == "none"


def test_post_result_animation_does_not_rearm_click_until_ready() -> None:
    runner, _events = _make_runner()
    rec = [
        SimpleNamespace(
            from_state=BotState.RESULT_WIN,
            to_state=BotState.POST_RESULT_ANIMATION,
            reason="post_result_animation_motion",
            confidence=0.9,
            frame_index=3,
            ts=datetime.now(timezone.utc),
            detail={"motion": 19.0},
        )
    ]
    runner._armed_for_click = False
    runner._handle_transitions(rec)
    assert runner._armed_for_click is False


def test_timestamp_normalization_clamps_inverted_ready_timestamp() -> None:
    runner, _events = _make_runner()
    spin = SpinResult(spin_index=3)
    spin.ts_result_detected = datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    spin.ts_ready_detected = datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc)
    spin.post_result_animation_duration_sec = -0.2
    runner._normalize_spin_result_timestamps(spin)
    assert spin.ts_ready_detected == spin.ts_result_detected
    assert spin.post_result_animation_duration_sec == 0.0


def test_win_unreadable_fallback_when_signal_detected_and_payout_missing() -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=4, visual_win=True, result_kind="no_win")
    runner._finalize_spin_result(
        detector_status="partial",
        reason="win_unreadable",
        payout=None,
        visual_win=True,
        fallback_used=True,
        probable_win_signal=True,
    )
    assert runner._active_spin is None
    _et, payload = _events[-1]
    assert payload["result_kind"] == "win_unreadable"
    assert payload["result_kind"] != "no_win"


def test_repeated_payout_sampling_succeeds_within_window(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=5, visual_win=True, result_kind="win", bet=1.0)
    runner._awaiting_ready_since = 100.0
    runner.settings.regions = SimpleNamespace(
        spin_button=None,
        bet_text=None,
        payout_text=SimpleNamespace(left=0, top=0, width=1, height=1),
        balance_text=None,
    )
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=0.0, spin_button_ready=True)
    sig.win = True
    seq = iter([(None, 0.0), (None, 0.0), (2.5, 0.9), (2.5, 0.9)])
    monkeypatch.setattr("src.session_runner.vision.ocr_region_text", lambda *_args, **_kwargs: "2.5")
    monkeypatch.setattr("src.session_runner.vision.crop_region", lambda *_args, **_kwargs: np.ones((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr("src.session_runner.vision.parse_numeric_amount", lambda *_args, **_kwargs: next(seq))
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.5)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin is not None
    assert runner._active_spin.payout_read_attempts > 1
    assert runner._active_spin.payout_read_success is True
    assert runner._active_spin.payout == 2.5


def test_no_win_stability_does_not_become_win_unreadable() -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=6, visual_win=False, result_kind="no_win")
    runner._active_spin.ts_ready_detected = datetime.now(timezone.utc)
    runner._finalize_spin_result(
        detector_status="fallback",
        reason="payout_not_readable",
        payout=None,
        visual_win=False,
        fallback_used=True,
        probable_win_signal=False,
    )
    _et, payload = _events[-1]
    assert payload["result_kind"] != "win_unreadable"
    assert payload["result_class"] == "confirmed_no_win"


def test_sampling_attempt_diagnostics_present_on_blank_and_parse_fail(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=7, visual_win=False, result_kind="no_win")
    runner._awaiting_ready_since = 100.0
    runner._result_detected_mono = 100.0
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=0.0, spin_button_ready=False)
    runner.settings.regions = SimpleNamespace(
        spin_button=None,
        bet_text=SimpleNamespace(left=0, top=0, width=1, height=1),
        payout_text=SimpleNamespace(left=0, top=0, width=1, height=1),
        balance_text=None,
    )
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.5)
    monkeypatch.setattr("src.session_runner.vision.crop_region", lambda *_args, **_kwargs: np.ones((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr("src.session_runner.vision.ocr_region_text", lambda *_args, **_kwargs: "abc")
    monkeypatch.setattr("src.session_runner.vision.parse_numeric_amount", lambda *_args, **_kwargs: (None, 0.0))
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin is not None
    assert runner._active_spin.raw_payout_ocr_samples
    assert any(d.get("code") == "parse_failed" for d in runner._active_spin.payout_sampling_diagnostics)


def test_click_to_spinning_timeout_guard_on_motion_signal(monkeypatch) -> None:
    runner, events = _make_runner()
    runner._active_spin = SpinResult(spin_index=8)
    runner._awaiting_spinning_since = 100.0
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=20.0, spin_button_ready=False)
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 110.0)
    runner._check_spin_timeouts(sig, frame)
    assert not any(payload.get("reason") == "click_to_spinning_timeout" for _et, payload in events if isinstance(payload, dict))


def test_click_timeout_suppressed_by_motion_only_spin_start_evidence(monkeypatch) -> None:
    runner, events = _make_runner()
    runner._active_spin = SpinResult(spin_index=10)
    runner._awaiting_spinning_since = 100.0
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = FrameSignals(
        ts=datetime.now(timezone.utc),
        frame_index=2,
        motion_score=15.0,
        reels_spinning=False,
        reels_stopped=False,
        popup=False,
        win=False,
        no_win_hint=False,
        bonus_tease=False,
        bonus_trigger=False,
        near_miss=False,
        session_end=False,
        spin_button_ready=False,
        confidences={"motion": 0.9},
    )
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 110.0)
    runner._check_spin_timeouts(sig, frame)
    assert runner._awaiting_spinning_since is None
    assert runner._spinning_since == 110.0
    assert not any(payload.get("reason") == "click_to_spinning_timeout" for _et, payload in events if isinstance(payload, dict))


def test_sampling_diag_dedupes_repeated_state_and_window_codes(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=11, visual_win=False, result_kind="no_win")
    runner._result_detected_mono = None
    runner._awaiting_ready_since = None
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=0.0, spin_button_ready=False)
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.5)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin is not None
    skipped = [d for d in runner._active_spin.payout_sampling_diagnostics if d.get("code") == "sampling_skipped_state"]
    assert len(skipped) == 1

    runner._result_detected_mono = 100.0
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 101.5)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    expired = [d for d in runner._active_spin.payout_sampling_diagnostics if d.get("code") == "sampling_window_expired"]
    assert len(expired) == 1


def test_spinning_to_ready_starts_sampling_window_without_result_state(monkeypatch) -> None:
    runner, events = _make_runner()
    runner.sm.state = BotState.READY_TO_SPIN
    runner._active_spin = SpinResult(spin_index=13, visual_win=False)
    rec = [
        SimpleNamespace(
            from_state=BotState.SPINNING,
            to_state=BotState.READY_TO_SPIN,
            reason="spin_button_ready",
            confidence=0.9,
            frame_index=4,
            ts=datetime.now(timezone.utc),
            detail={},
        )
    ]
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 120.0)
    runner._handle_transitions(rec)
    assert runner._awaiting_ready_since == 120.0
    assert runner._result_detected_mono == 120.0
    assert runner._finalize_on_ready is True
    assert any(et == SessionEventType.SPIN_STOPPED for et, _payload in events)


def test_balance_delta_fallback_uses_staged_before_after_samples(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner.settings.detection.use_ocr_balance = True
    runner.settings.regions = SimpleNamespace(
        spin_button=None,
        bet_text=None,
        payout_text=SimpleNamespace(left=0, top=0, width=1, height=1),
        balance_text=SimpleNamespace(left=0, top=0, width=1, height=1),
    )
    runner._active_spin = SpinResult(spin_index=12, visual_win=False, result_kind="no_win", balance_before=100.0)
    runner._awaiting_ready_since = 100.0
    runner._result_detected_mono = 100.0
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=2, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=0.0, spin_button_ready=False)
    seq = iter([(None, 0.0), (101.0, 0.9), (None, 0.0), (101.0, 0.9)])
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.5)
    monkeypatch.setattr("src.session_runner.vision.crop_region", lambda *_args, **_kwargs: np.ones((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr("src.session_runner.vision.ocr_region_text", lambda *_args, **_kwargs: "101.0")
    monkeypatch.setattr("src.session_runner.vision.parse_numeric_amount", lambda *_args, **_kwargs: next(seq))
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin is not None
    assert runner._active_spin.balance_before == 100.0
    assert runner._active_spin.balance_after == 101.0
    assert runner._active_spin.payout_source == "balance_delta"
    assert runner._active_spin.payout == 1.0


def test_long_animation_without_win_cues_is_not_probable_win(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=7, visual_win=False, result_kind="no_win")
    runner._post_result_animation_since = 100.0
    runner._post_result_animation_reason = "post_result_animation"
    sig = _make_sig(motion=15.0, spin_button_ready=True)
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 110.0)
    assert runner._detect_probable_win_signal(sig) is False


def test_deferred_finalize_stays_armed_and_finalizes_after_late_payout(monkeypatch) -> None:
    runner, events = _make_runner()
    runner._active_spin = SpinResult(spin_index=8, visual_win=True, result_kind="win", bet=1.0)
    runner._awaiting_ready_since = 100.0
    runner._result_detected_mono = 100.0
    runner._finalize_on_ready = True
    runner.settings.regions = SimpleNamespace(
        spin_button=None,
        bet_text=None,
        payout_text=SimpleNamespace(left=0, top=0, width=1, height=1),
        balance_text=None,
    )
    sig = _make_sig(motion=0.0, spin_button_ready=True)
    sig.win = True

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.2)
    assert runner._should_defer_ready_finalize(True) is True
    assert runner._finalize_on_ready is True

    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=3, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    seq = iter([(None, 0.0), (2.0, 0.9), (2.0, 0.9)])
    monkeypatch.setattr("src.session_runner.vision.ocr_region_text", lambda *_args, **_kwargs: "2.0")
    monkeypatch.setattr("src.session_runner.vision.parse_numeric_amount", lambda *_args, **_kwargs: next(seq))
    monkeypatch.setattr("src.session_runner.vision.crop_region", lambda *_args, **_kwargs: np.ones((1, 1, 3), dtype=np.uint8))
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.5)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin is not None
    assert runner._active_spin.payout == 2.0
    assert runner._should_defer_ready_finalize(True) is False
    assert runner._finalize_on_ready is True

    runner._finalize_spin_result(
        detector_status=runner._active_spin.detector_status,
        reason=runner._active_spin.reason,
        payout=runner._active_spin.payout,
        visual_win=runner._active_spin.visual_win,
        fallback_used=False,
        probable_win_signal=True,
    )
    _et, payload = events[-1]
    assert payload["result_kind"] == "win"


def test_retry_window_bounds_payout_sampling_attempts(monkeypatch) -> None:
    runner, _events = _make_runner()
    runner._active_spin = SpinResult(spin_index=9, visual_win=True, result_kind="win", bet=1.0)
    runner._awaiting_ready_since = 100.0
    runner._result_detected_mono = 100.0
    frame = FramePacket(ts=datetime.now(timezone.utc), frame_index=4, image_bgr=np.zeros((4, 4, 3), dtype=np.uint8))
    sig = _make_sig(motion=0.0, spin_button_ready=True)
    sig.win = True
    monkeypatch.setattr("src.session_runner.vision.ocr_region_text", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("src.session_runner.vision.parse_numeric_amount", lambda *_args, **_kwargs: (None, 0.0))

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 100.6)
    runner._update_payout_resolution(frame, sig)
    attempts_in_window = runner._active_spin.payout_read_attempts

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 101.2)
    runner._update_payout_resolution(frame, sig)
    assert runner._active_spin.payout_read_attempts == attempts_in_window
