"""Main observe/automation loop: capture → vision → state machine → logs."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .automation import ClickBudget, SafeClickService, jitter_point_in_region
from .capture import CaptureService
from .logger import get_logger
from .models import (
    BotState,
    BotSettings,
    CoordinateMode,
    FramePacket,
    Region,
    SessionEvent,
    SessionEventType,
    new_session_id,
)
from .reporting import append_jsonl, ensure_output_dirs, write_csv_summary, write_markdown_report
from .state_machine import FrameSignals, GameStateMachine, TransitionRecord
from .templates import load_template_grayscale
from . import vision
from .metrics import build_summary
from .app_paths import resolve_path_relative_to_app
from .utils import random_delay

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def region_to_screen(capture: Region, sub: Region, mode: CoordinateMode) -> Region:
    if mode == CoordinateMode.ABSOLUTE_SCREEN:
        return sub
    return Region(
        left=capture.left + sub.left,
        top=capture.top + sub.top,
        width=sub.width,
        height=sub.height,
    )


class SessionRunner:
    def __init__(
        self,
        settings: BotSettings,
        live_click: bool = False,
        dry_run: bool = True,
    ) -> None:
        self.settings = settings
        self.live_click = live_click
        self.dry_run = dry_run
        self.session_id = new_session_id()
        self._stop = threading.Event()
        self._esc_pressed = threading.Event()
        self._session_stop_emitted = False

        cap = settings.capture.region
        self.capture = CaptureService(region=cap)
        self.fps = settings.capture.fps

        self._tmpl_cache: dict[str, np.ndarray | None] = {}
        for name, spec in settings.templates.items():
            self._tmpl_cache[name] = load_template_grayscale(spec.path)

        mode = settings.capture.coordinate_mode
        self._coord_mode = mode
        self._capture_bbox = cap

        spin_screen = region_to_screen(cap, settings.regions.spin_button, mode)
        enable = settings.automation.enable_clicking and live_click and not dry_run
        self.clicker = SafeClickService(
            spin_region_screen=spin_screen,
            dry_run=not enable,
            live=enable,
            jitter_px=settings.automation.click_jitter_px,
            budget=ClickBudget(max_per_minute=settings.automation.max_clicks_per_minute),
        )

        self.sm = GameStateMachine(
            debounce_frames=settings.detection.state_debounce_frames,
            min_confidence=settings.detection.min_confidence_for_transition,
        )

        self._motion_hist: deque[np.ndarray] = deque(maxlen=max(3, settings.detection.motion_history_frames))
        self._stopped_count = 0
        self._spinning_count = 0
        self._spin_counter = 0
        self._last_near_miss = 0.0
        self._last_tease = 0.0
        self._popup_frames = 0
        self._error_count = 0
        self._armed_for_click = True

        self._logs: list[dict[str, Any]] = []
        self._panic_path = self._resolve_panic_path(settings.automation.panic_stop_file)

    def _resolve_panic_path(self, p: str) -> Path:
        return resolve_path_relative_to_app(p)

    def request_stop(self) -> None:
        self._stop.set()

    def _on_esc(self) -> None:
        log.warning("Stop requested (ESC)")
        self._stop.set()

    def _setup_hotkeys(self) -> None:
        try:
            import keyboard

            keyboard.add_hotkey("esc", self._on_esc, suppress=False)
        except Exception as e:
            log.info("keyboard module unavailable for ESC hotkey: %s", e)

    def _panic_file_triggered(self) -> bool:
        try:
            return self._panic_path.is_file()
        except OSError:
            return False

    def _emit(self, event_type: SessionEventType, payload: dict[str, Any]) -> None:
        if event_type == SessionEventType.SESSION_STOPPED:
            self._session_stop_emitted = True
        ev = SessionEvent(session_id=self.session_id, ts=_utcnow(), event_type=event_type, payload=payload)
        rec = ev.model_dump(mode="json")
        self._logs.append(rec)
        logs_dir, _ = ensure_output_dirs()
        append_jsonl(logs_dir / f"session_{self.session_id}.jsonl", rec)
        log.debug("event %s %s", event_type, payload)

    def _crop_rel(self, frame: FramePacket, reg: Region) -> np.ndarray:
        return vision.crop_region(frame.image_bgr, reg)

    def _match(self, frame_gray: np.ndarray, name: str) -> tuple[bool, float]:
        spec = self.settings.templates.get(name)
        if not spec:
            return False, 0.0
        tmpl = self._tmpl_cache.get(name)
        if tmpl is None:
            return False, 0.0
        # match within full frame or relevant region — use full frame for simplicity
        score, _loc = vision.template_match_best(frame_gray, tmpl)
        return score >= spec.threshold, float(score)

    def _build_signals(self, frame: FramePacket, reel_prev: np.ndarray | None) -> FrameSignals:
        cfg = self.settings
        det = cfg.detection
        g = vision.to_gray(frame.image_bgr)
        reels = self._crop_rel(frame, cfg.regions.reels)
        reel_gray = vision.to_gray(reels)

        mscore = vision.motion_score(reel_prev, reel_gray)
        hist = list(self._motion_hist)
        self._motion_hist.append(reel_gray.copy())
        roll = vision.rolling_motion_score(hist, reel_gray, det.motion_history_frames)

        motion_val = max(mscore, roll)
        is_spinning = motion_val >= det.spinning_motion_threshold
        is_stopped = motion_val < det.spinning_motion_threshold

        if is_spinning:
            self._spinning_count += 1
            self._stopped_count = 0
        else:
            self._stopped_count += 1
            self._spinning_count = 0

        reels_stopped = self._stopped_count >= det.stopped_consecutive_frames
        reels_spinning = self._spinning_count >= 1  # immediate feedback for spin start

        popup = False
        pconf = 0.0
        if "popup_close_x" in cfg.templates:
            popup, pconf = self._match(g, "popup_close_x")

        win = False
        wconf = 0.0
        if "win_banner" in cfg.templates:
            win, wconf = self._match(g, "win_banner")

        near_m = False
        nmconf = 0.0
        if "near_miss" in cfg.templates:
            near_m, nmconf = self._match(g, "near_miss")

        btease = False
        btconf = 0.0
        if "bonus_tease" in cfg.templates:
            btease, btconf = self._match(g, "bonus_tease")

        btrig = False
        btrconf = 0.0
        if "bonus_trigger" in cfg.templates:
            btrig, btrconf = self._match(g, "bonus_trigger")

        s_end = False
        if "session_end" in cfg.templates:
            s_end, _ = self._match(g, "session_end")

        s_ready = False
        srconf = 0.0
        if "spin_button_ready" in cfg.templates:
            s_ready, srconf = self._match(g, "spin_button_ready")
        else:
            # Without template, assume ready when not spinning and not popup
            s_ready = not is_spinning and not popup
            srconf = 0.45

        # Cooldowns for near-miss / tease
        now = time.monotonic()
        if near_m and (now - self._last_near_miss) < det.near_miss_cooldown_sec:
            near_m = False
        elif near_m:
            self._last_near_miss = now

        if btease and (now - self._last_tease) < det.bonus_tease_cooldown_sec:
            btease = False
        elif btease:
            self._last_tease = now

        if popup:
            self._popup_frames += 1
        else:
            self._popup_frames = 0

        confidences = {
            "motion": min(1.0, motion_val / max(1.0, det.spinning_motion_threshold * 2)),
            "popup": pconf,
            "win": wconf,
            "near_miss": nmconf,
            "bonus_tease": btconf,
            "bonus_trigger": btrconf,
            "spin_ready": srconf,
            "session_end": 0.8 if s_end else 0.0,
        }

        return FrameSignals(
            ts=frame.ts,
            frame_index=frame.frame_index,
            motion_score=motion_val,
            reels_spinning=is_spinning and not reels_stopped,
            reels_stopped=reels_stopped,
            popup=popup,
            win=win and reels_stopped,
            no_win_hint=reels_stopped and not win,
            bonus_tease=btease,
            bonus_trigger=btrig,
            near_miss=near_m and reels_stopped,
            session_end=s_end,
            spin_button_ready=s_ready,
            confidences=confidences,
        )

    def _handle_transitions(self, recs: list[TransitionRecord]) -> None:
        for r in recs:
            self._emit(
                SessionEventType.STATE_TRANSITION,
                {
                    "from": r.from_state.value,
                    "to": r.to_state.value,
                    "reason": r.reason,
                    "confidence": r.confidence,
                    "frame_index": r.frame_index,
                },
            )
            # Semantic hooks
            if r.to_state == BotState.SPINNING and r.from_state in (
                BotState.READY_TO_SPIN,
                BotState.IDLE,
            ):
                self._spin_counter += 1
                self._emit(
                    SessionEventType.SPIN_STARTED,
                    {"spin_index": self._spin_counter, "confidence": r.confidence},
                )
            if r.from_state == BotState.SPINNING and r.to_state in (
                BotState.RESULT_WIN,
                BotState.RESULT_NO_WIN,
            ):
                self._emit(SessionEventType.SPIN_STOPPED, {"spin_index": self._spin_counter})
            if r.to_state == BotState.RESULT_WIN:
                self._emit(
                    SessionEventType.WIN_DETECTED,
                    {"spin_index": self._spin_counter, "confidence": r.confidence},
                )
            if r.to_state == BotState.RESULT_NO_WIN:
                self._emit(
                    SessionEventType.NO_WIN_DETECTED,
                    {"spin_index": self._spin_counter, "confidence": r.confidence},
                )
                if r.reason == "near_miss_template":
                    self._emit(
                        SessionEventType.NEAR_MISS_DETECTED,
                        {"spin_index": self._spin_counter, "confidence": r.confidence},
                    )
            if r.to_state == BotState.BONUS_TEASE:
                self._emit(SessionEventType.BONUS_TEASE_DETECTED, {"confidence": r.confidence})
            if r.to_state == BotState.BONUS_TRIGGERED:
                self._emit(SessionEventType.BONUS_TRIGGERED, {"confidence": r.confidence})
            if r.to_state == BotState.POPUP_BLOCKING:
                self._emit(SessionEventType.POPUP_DETECTED, {"confidence": r.confidence})
            if r.to_state == BotState.READY_TO_SPIN:
                self._armed_for_click = True
            if r.to_state == BotState.SESSION_ENDED:
                pass

    def run(self) -> None:
        if not self.settings.calibrated:
            raise RuntimeError(
                "Configuration not marked calibrated. Run `python calibrate.py` first."
            )

        self._setup_hotkeys()
        started = _utcnow()
        self._emit(
            SessionEventType.CALIBRATION,
            {"message": "session_start", "live_click": self.live_click, "dry_run": self.dry_run},
        )

        limits = self.settings.session
        t_end = time.monotonic() + limits.max_duration_min * 60.0
        frame_index = 0
        reel_prev: np.ndarray | None = None

        try:
            while not self._stop.is_set():
                if time.monotonic() > t_end:
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "max_duration"})
                    break
                if self._spin_counter >= limits.max_spins:
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "max_spins"})
                    break
                if self._error_count >= limits.max_consecutive_errors:
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "too_many_errors"})
                    break
                if self._panic_file_triggered():
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "panic_file"})
                    break
                if self._popup_frames >= self.settings.detection.popup_persist_frames:
                    # Auto-pause clicking via stopping session or just log — user asked auto-pause
                    log.error("Persistent popup — stopping session for safety")
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "popup_persist"})
                    break

                loop_start = time.monotonic()
                try:
                    frame = self.capture.grab_frame_packet(frame_index)
                except Exception as e:
                    log.exception("capture error: %s", e)
                    self._error_count += 1
                    continue

                sig = self._build_signals(frame, reel_prev)
                reel_prev = vision.to_gray(self._crop_rel(frame, self.settings.regions.reels))

                recs = self.sm.update(sig)
                self._handle_transitions(recs)

                if self.sm.state == BotState.SESSION_ENDED:
                    self._emit(SessionEventType.SESSION_STOPPED, {"reason": "session_end_detected"})
                    break

                # Automation: one action per READY cycle
                if (
                    self.sm.state == BotState.READY_TO_SPIN
                    and self.settings.automation.enable_clicking
                    and self._armed_for_click
                ):
                    spin_region = region_to_screen(
                        self._capture_bbox,
                        self.settings.regions.spin_button,
                        self._coord_mode,
                    )
                    if self.dry_run or not self.live_click:
                        x, y = jitter_point_in_region(
                            spin_region,
                            self.settings.automation.click_jitter_px,
                        )
                        self._emit(
                            SessionEventType.CLICK_DRY_RUN,
                            {"x": x, "y": y},
                        )
                        self._armed_for_click = False
                    else:
                        random_delay(
                            self.settings.automation.min_delay_sec,
                            self.settings.automation.max_delay_sec,
                        )
                        pt = self.clicker.click_spin()
                        if pt is not None:
                            self._emit(
                                SessionEventType.CLICK_LIVE,
                                {"x": pt[0], "y": pt[1]},
                            )
                            self._armed_for_click = False

                frame_index += 1
                self.capture.sleep_for_fps(self.fps, loop_start)

        except KeyboardInterrupt:
            self._emit(SessionEventType.SESSION_STOPPED, {"reason": "keyboard_interrupt"})
        finally:
            ended = _utcnow()
            if not self._session_stop_emitted:
                self._emit(
                    SessionEventType.SESSION_STOPPED,
                    {"reason": "completed" if not self._stop.is_set() else "user_stop"},
                )
            summary = build_summary(self.session_id, self._logs)
            summary.started_at = started
            summary.ended_at = ended
            if summary.duration_sec == 0 and ended and started:
                summary.duration_sec = max(0.0, (ended - started).total_seconds())

            _, reports_dir = ensure_output_dirs()
            csv_path = reports_dir / f"session_{self.session_id}.csv"
            md_path = reports_dir / f"session_{self.session_id}.md"
            write_csv_summary(csv_path, summary)
            write_markdown_report(md_path, summary)
            log.info("Wrote reports to %s and %s", csv_path, md_path)


def run_session(
    settings: BotSettings,
    live_click: bool = False,
    dry_run: bool = True,
) -> None:
    SessionRunner(settings, live_click=live_click, dry_run=dry_run).run()
