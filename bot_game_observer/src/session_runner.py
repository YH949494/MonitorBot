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
from .spin_result import DetectorStatus, SpinResult, classify_spin_result
from .templates import load_template_grayscale
from . import vision
from .metrics import build_summary
from .app_paths import logs_path, resolve_path_relative_to_app
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
        self._active_spin: SpinResult | None = None
        self._awaiting_spinning_since: float | None = None
        self._spinning_since: float | None = None
        self._awaiting_ready_since: float | None = None
        self._payout_samples: deque[tuple[float, float]] = deque(maxlen=3)
        self._bet_samples: deque[tuple[float, float]] = deque(maxlen=6)
        self._balance_samples: deque[tuple[float, float]] = deque(maxlen=3)
        self._result_evidence_saved = False
        self._finalize_on_ready = False
        self._last_near_miss = 0.0
        self._last_tease = 0.0
        self._popup_frames = 0
        self._result_recovery_stable_frames = 0
        self._result_ready_debug_frames: deque[dict[str, Any]] = deque(maxlen=10)
        self._result_spin_button_ready_evidence_saved = False
        self._post_result_animation_since: float | None = None
        self._post_result_animation_reason: str | None = None
        self._result_detected_mono: float | None = None
        self._spin_start_evidence_seen = False
        self._locked_session_bet: float | None = None
        self._bet_lock_acquired_at_spin: int | None = None
        self._bet_lock_source: str | None = None
        self._last_payout_sample_mono: float | None = None
        self._last_bet_sample_mono: float | None = None
        self._error_count = 0
        self._armed_for_click = True

        self._logs: list[dict[str, Any]] = []
        self._panic_path = self._resolve_panic_path(settings.automation.panic_stop_file)
        self._payout_evidence_dir = resolve_path_relative_to_app(settings.detection.payout_evidence_dir)
        self._session_frames_dir = logs_path("sessions", self.session_id, "frames")
        self._session_debug_dir = logs_path("sessions", self.session_id, "debug")
        self._last_symbol_capture_by_reason: dict[str, float] = {}
        self._symbol_debug_saved_spins = 0

    def _classify_post_result_visual(self, duration_sec: float, bonus_like_signal: bool) -> str:
        det = self.settings.detection
        if bonus_like_signal or duration_sec >= det.post_result_bonus_like_threshold_sec:
            return "bonus_like"
        if duration_sec >= det.post_result_long_animation_threshold_sec:
            return "long_animation"
        if duration_sec >= det.post_result_normal_threshold_sec:
            return "normal_result"
        return "none"

    def _start_spin_attempt(self, click_ts: datetime | None) -> None:
        if self._spin_counter == 0:
            self._locked_session_bet = None
            self._bet_lock_acquired_at_spin = None
            self._bet_lock_source = None
        self._spin_counter += 1
        self._active_spin = SpinResult(
            spin_index=self._spin_counter,
            ts_start=_utcnow(),
            ts_click=click_ts,
            detector_status="partial",
            reason="awaiting_spinning_after_click",
        )
        self._awaiting_spinning_since = time.monotonic()
        self._spinning_since = None
        self._awaiting_ready_since = None
        self._payout_samples.clear()
        self._bet_samples.clear()
        self._balance_samples.clear()
        self._result_evidence_saved = False
        self._finalize_on_ready = False
        self._post_result_animation_since = None
        self._post_result_animation_reason = None
        self._result_detected_mono = None
        self._spin_start_evidence_seen = False
        self._last_payout_sample_mono = None
        self._last_bet_sample_mono = None

    def _save_evidence_crop(self, frame: FramePacket, region: Region | None, tag: str) -> None:
        if not self.settings.detection.payout_evidence_mode or self._active_spin is None or region is None:
            return
        crop = vision.crop_region(frame.image_bgr, region)
        if crop.size == 0:
            return
        self._payout_evidence_dir.mkdir(parents=True, exist_ok=True)
        out = self._payout_evidence_dir / f"spin_{self._active_spin.spin_index}_{tag}_f{frame.frame_index}.png"
        cv2.imwrite(str(out), crop)

    def _detect_symbol_observations(self, frame: FramePacket) -> dict[str, Any]:
        if self._active_spin is None:
            return {}
        reels_bgr = self._crop_rel(frame, self.settings.regions.reels)
        reels_gray = vision.to_gray(reels_bgr)
        det_cfg = self.settings.detection
        scatter_spec = self.settings.templates.get("scatter_symbol")
        scatter_tmpl = self._tmpl_cache.get("scatter_symbol")
        scatter_threshold = scatter_spec.threshold if scatter_spec is not None else None
        scatter_template_shape = (
            [int(scatter_tmpl.shape[0]), int(scatter_tmpl.shape[1])]
            if scatter_tmpl is not None and hasattr(scatter_tmpl, "shape") and len(scatter_tmpl.shape) >= 2
            else None
        )
        reels_shape = (
            [int(reels_gray.shape[0]), int(reels_gray.shape[1])]
            if hasattr(reels_gray, "shape") and len(reels_gray.shape) >= 2
            else None
        )
        scatter_debug_best_score: float | None = None
        scatter_debug_best_loc: list[int] | None = None
        scatter_debug_ran = False
        scatter_debug_reason: str | None = None
        try:
            if scatter_spec is None:
                scatter_debug_reason = "missing_template_spec"
            elif scatter_tmpl is None:
                scatter_debug_reason = "missing_template_image"
            elif (
                not hasattr(scatter_tmpl, "shape")
                or len(scatter_tmpl.shape) < 2
                or int(scatter_tmpl.shape[0]) <= 0
                or int(scatter_tmpl.shape[1]) <= 0
            ):
                scatter_debug_reason = "invalid_template"
            elif (
                not hasattr(reels_gray, "shape")
                or len(reels_gray.shape) < 2
                or int(reels_gray.shape[0]) <= 0
                or int(reels_gray.shape[1]) <= 0
            ):
                scatter_debug_reason = "invalid_reels_crop"
            elif int(scatter_tmpl.shape[0]) > int(reels_gray.shape[0]) or int(scatter_tmpl.shape[1]) > int(reels_gray.shape[1]):
                scatter_debug_reason = "template_larger_than_scene"
            else:
                best_score, best_loc = vision.template_match_best(reels_gray, scatter_tmpl)
                scatter_debug_best_score = float(best_score)
                scatter_debug_best_loc = [int(best_loc[0]), int(best_loc[1])]
                scatter_debug_ran = True
                scatter_debug_reason = "match_scored"
        except Exception:
            scatter_debug_reason = "exception"

        def _detect(name: str) -> tuple[int | None, bool, list[dict[str, int]] | None, list[float] | None]:
            spec = self.settings.templates.get(name)
            if spec is None:
                return None, False, None, None
            if spec.threshold <= 0:
                return None, False, None, None
            tmpl = self._tmpl_cache.get(name)
            ok, boxes, scores = vision.template_match_locations(
                reels_gray,
                tmpl,
                spec.threshold,
                max_matches=det_cfg.symbol_max_count_cap,
                center_merge_px=det_cfg.symbol_match_center_merge_px,
            )
            if not ok:
                return None, False, None, None
            return len(boxes), True, boxes, scores

        scatter_count, scatter_ok, scatter_boxes, scatter_scores = _detect("scatter_symbol")
        bonus_count, bonus_ok, bonus_boxes, bonus_scores = _detect("bonus_symbol")
        scatter_near_miss = (
            scatter_ok is True
            and det_cfg.scatter_trigger_count is not None
            and scatter_count == det_cfg.scatter_trigger_count - 1
        )
        bonus_tease = (
            bonus_ok is True
            and det_cfg.bonus_trigger_count is not None
            and bonus_count == det_cfg.bonus_trigger_count - 1
        )
        scatter_detected = (
            scatter_ok is True
            and scatter_count is not None
            and scatter_count >= det_cfg.scatter_min_count_for_signal
            and any(s >= det_cfg.scatter_min_score_for_signal for s in (scatter_scores or []))
        )
        bonus_detected = (
            bonus_ok is True
            and bonus_count is not None
            and bonus_count >= det_cfg.bonus_min_count_for_signal
            and any(s >= det_cfg.bonus_min_score_for_signal for s in (bonus_scores or []))
        )
        reason_flags: list[str] = []
        if scatter_detected:
            reason_flags.append("scatter_detected")
        if bonus_detected:
            reason_flags.append("bonus_detected")
        if scatter_near_miss:
            reason_flags.append("scatter_near_miss")
        if bonus_tease:
            reason_flags.append("bonus_tease")

        should_capture = bool(reason_flags)
        frame_rel_path: str | None = None
        frame_ts: str | None = None
        debug_reels_path: str | None = None
        symbol_debug_mode = bool(getattr(det_cfg, "symbol_debug_mode", False))
        symbol_debug_max_spins = max(0, int(getattr(det_cfg, "symbol_debug_max_spins", 10)))
        symbol_debug_save_reels_crop = bool(getattr(det_cfg, "symbol_debug_save_reels_crop", True))
        if should_capture:
            primary = reason_flags[0]
            now = time.monotonic()
            last_ts = self._last_symbol_capture_by_reason.get(primary, 0.0)
            if (now - last_ts) >= det_cfg.symbol_capture_cooldown_sec:
                self._session_frames_dir.mkdir(parents=True, exist_ok=True)
                filename = f"spin_{self._active_spin.spin_index:04d}_{primary}.png"
                out = self._session_frames_dir / filename
                image_to_save = reels_bgr.copy()
                if scatter_boxes:
                    for idx, box in enumerate(scatter_boxes):
                        cv2.rectangle(
                            image_to_save,
                            (box["x"], box["y"]),
                            (box["x"] + box["w"], box["y"] + box["h"]),
                            (0, 255, 0),
                            2,
                        )
                        score = (scatter_scores or [None] * len(scatter_boxes))[idx]
                        lbl = "SCATTER" if score is None else f"SCATTER {score:.2f}"
                        cv2.putText(
                            image_to_save,
                            lbl,
                            (box["x"], max(10, box["y"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                if bonus_boxes:
                    for idx, box in enumerate(bonus_boxes):
                        cv2.rectangle(
                            image_to_save,
                            (box["x"], box["y"]),
                            (box["x"] + box["w"], box["y"] + box["h"]),
                            (0, 165, 255),
                            2,
                        )
                        score = (bonus_scores or [None] * len(bonus_boxes))[idx]
                        lbl = "BONUS" if score is None else f"BONUS {score:.2f}"
                        cv2.putText(
                            image_to_save,
                            lbl,
                            (box["x"], max(10, box["y"] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 165, 255),
                            1,
                            cv2.LINE_AA,
                        )
                if not out.exists():
                    cv2.imwrite(str(out), image_to_save)
                frame_rel_path = str(Path("logs") / "sessions" / self.session_id / "frames" / filename)
                frame_ts = frame.ts.isoformat()
                self._last_symbol_capture_by_reason[primary] = now
        if (
            symbol_debug_mode
            and symbol_debug_save_reels_crop
            and self._symbol_debug_saved_spins < symbol_debug_max_spins
            and reels_bgr.size > 0
        ):
            self._session_debug_dir.mkdir(parents=True, exist_ok=True)
            debug_name = f"spin_{self._active_spin.spin_index:04d}_reels_debug.png"
            debug_out = self._session_debug_dir / debug_name
            if cv2.imwrite(str(debug_out), reels_bgr):
                debug_reels_path = str(Path("logs") / "sessions" / self.session_id / "debug" / debug_name)
                self._symbol_debug_saved_spins += 1

        return {
            "scatter_count": scatter_count,
            "bonus_count": bonus_count,
            "scatter_detect_ok": scatter_ok,
            "bonus_detect_ok": bonus_ok,
            "scatter_near_miss": scatter_near_miss,
            "bonus_tease": bonus_tease,
            "scatter_trigger_count": det_cfg.scatter_trigger_count,
            "bonus_trigger_count": det_cfg.bonus_trigger_count,
            "symbol_detection_frame_path": frame_rel_path,
            "symbol_detection_frame_ts": frame_ts,
            "scatter_boxes": scatter_boxes,
            "bonus_boxes": bonus_boxes,
            "scatter_match_scores": scatter_scores,
            "bonus_match_scores": bonus_scores,
            "symbol_detection_reason_flags": reason_flags or None,
            "scatter_debug_template_present": scatter_tmpl is not None,
            "scatter_debug_template_shape": scatter_template_shape,
            "scatter_debug_reels_shape": reels_shape,
            "scatter_debug_best_score": scatter_debug_best_score,
            "scatter_debug_best_loc": scatter_debug_best_loc,
            "scatter_debug_threshold": scatter_threshold,
            "scatter_debug_frame_index": frame.frame_index,
            "scatter_debug_ran": scatter_debug_ran,
            "scatter_debug_reason": scatter_debug_reason,
            "scatter_debug_reels_path": debug_reels_path,
        }

    def _finalize_spin_result(
        self,
        *,
        detector_status: DetectorStatus,
        reason: str,
        payout: float | None = None,
        visual_win: bool | None = None,
        fallback_used: bool = False,
        probable_win_signal: bool = False,
    ) -> None:
        if self._active_spin is None:
            return
        if probable_win_signal:
            self._active_spin.win_signal_detected = True
        canonical_bet = self._locked_session_bet if self._locked_session_bet is not None else self._active_spin.bet
        payload = classify_spin_result(
            bet=canonical_bet,
            payout=payout,
            visual_win=visual_win if visual_win is not None else self._active_spin.visual_win,
            detector_status=detector_status,
            reason=reason,
            result_kind=self._active_spin.result_kind,
            ready_recovered=(
                self._active_spin.ts_ready_detected is not None and not self._active_spin.timeouts.result_to_ready
            ),
            win_signal_detected=self._active_spin.win_signal_detected,
            payout_source=self._active_spin.payout_source,
            balance_delta=(
                None
                if self._active_spin.balance_before is None or self._active_spin.balance_after is None
                else round(self._active_spin.balance_after - self._active_spin.balance_before, 2)
            ),
        )
        for k, v in payload.items():
            setattr(self._active_spin, k, v)
        if payout is None and probable_win_signal:
            self._active_spin.result_kind = "win_unreadable"
        self._active_spin.chosen_payout_source = self._active_spin.payout_source
        self._active_spin.payout = payout
        self._active_spin.payout_read_success = payout is not None
        self._active_spin.locked_session_bet = self._locked_session_bet
        self._active_spin.bet_lock_acquired_at_spin = self._bet_lock_acquired_at_spin
        self._active_spin.bet_lock_source = self._bet_lock_source
        self._active_spin.canonical_bet = canonical_bet
        self._active_spin.empty_spin = self._active_spin.result_kind == "no_win"
        if payout is None or canonical_bet is None:
            self._active_spin.visual_win_by_bet = None
            self._active_spin.big_win = None
        else:
            self._active_spin.visual_win_by_bet = payout < canonical_bet
            self._active_spin.big_win = payout > canonical_bet
        self._active_spin.fallback_used = fallback_used
        if self._active_spin.ts_result_detected is None:
            self._active_spin.ts_result_detected = _utcnow()
        self._normalize_spin_result_timestamps(self._active_spin)
        ev_payload = self._active_spin.model_dump(mode="json")
        ev_payload["type"] = SessionEventType.SPIN_RESULT_SUMMARY.value
        self._emit(SessionEventType.SPIN_RESULT_SUMMARY, ev_payload)
        self._active_spin = None
        self._awaiting_spinning_since = None
        self._spinning_since = None
        self._awaiting_ready_since = None
        self._payout_samples.clear()
        self._bet_samples.clear()
        self._balance_samples.clear()
        self._result_evidence_saved = False
        self._finalize_on_ready = False
        self._post_result_animation_since = None
        self._post_result_animation_reason = None
        self._result_detected_mono = None
        self._spin_start_evidence_seen = False
        self._last_payout_sample_mono = None
        self._last_bet_sample_mono = None

    def _normalize_spin_result_timestamps(self, spin: SpinResult) -> None:
        if spin.post_result_animation_duration_sec is not None:
            spin.post_result_animation_duration_sec = max(0.0, spin.post_result_animation_duration_sec)
        if spin.ts_result_detected and spin.ts_ready_detected and spin.ts_ready_detected < spin.ts_result_detected:
            log.warning(
                "spin timestamp normalization applied spin=%s ts_result=%s ts_ready=%s",
                spin.spin_index,
                spin.ts_result_detected.isoformat(),
                spin.ts_ready_detected.isoformat(),
            )
            spin.ts_ready_detected = spin.ts_result_detected

    def _detect_probable_win_signal(self, sig: FrameSignals) -> bool:
        if self._active_spin is None:
            return False
        if sig.win or self._active_spin.visual_win:
            return True
        if sig.bonus_trigger or self._post_result_animation_reason == "bonus_feature_animation":
            return True
        return False

    def _should_defer_ready_finalize(self, probable_win_signal: bool) -> bool:
        if self._active_spin is None or self._active_spin.payout is not None:
            return False
        if not probable_win_signal or self._result_detected_mono is None:
            return False
        elapsed = time.monotonic() - self._result_detected_mono
        return (
            elapsed < (self.settings.detection.payout_sampling_window_ms / 1000.0)
            and self._active_spin.payout_read_attempts < self.settings.detection.payout_stabilization_max_attempts
        )

    def _read_amount(self, frame: FramePacket, region: Region | None) -> tuple[float | None, float]:
        if region is None:
            return None, 0.0
        text = vision.ocr_region_text(frame.image_bgr, region, lang=self.settings.detection.ocr_lang)
        return vision.parse_numeric_amount(text)

    def _stable_amount(self, samples: deque[tuple[float, float]]) -> tuple[float | None, float]:
        if len(samples) < 2:
            return None, 0.0
        v1, c1 = samples[-1]
        v2, c2 = samples[-2]
        if abs(v1 - v2) > 0.009:
            return None, 0.0
        return float(v1), float((c1 + c2) / 2.0)

    def _stabilize_amount(
        self,
        samples: deque[tuple[float, float]],
        *,
        min_confirmations: int,
        min_attempts: int,
    ) -> tuple[float | None, float, str | None]:
        if len(samples) < max(1, min_attempts):
            return None, 0.0, None
        normalized_values = [round(v, 2) for v, _c in samples]
        last = normalized_values[-1]
        consecutive = 1
        for idx in range(len(normalized_values) - 2, -1, -1):
            if normalized_values[idx] != last:
                break
            consecutive += 1
        if consecutive >= min_confirmations:
            confidences = [c for v, c in samples if round(v, 2) == last]
            conf = (sum(confidences) / len(confidences)) if confidences else 0.0
            return float(last), float(conf), "consecutive"

        counts: dict[float, int] = {}
        conf_by_value: dict[float, list[float]] = {}
        for value, conf in samples:
            k = round(value, 2)
            counts[k] = counts.get(k, 0) + 1
            conf_by_value.setdefault(k, []).append(conf)
        top_value, top_count = max(counts.items(), key=lambda item: item[1])
        if top_count >= min_confirmations and top_count > (len(samples) // 2):
            conf = sum(conf_by_value[top_value]) / len(conf_by_value[top_value])
            return float(top_value), float(conf), "majority"
        return None, 0.0, None

    def _maybe_lock_session_bet(self, stable_bet: float, source: str) -> None:
        if self._active_spin is None or self._locked_session_bet is not None:
            return
        max_lock_spin = self.settings.detection.bet_lock_max_spins_from_session_start
        if self._active_spin.spin_index > max_lock_spin:
            return
        self._locked_session_bet = stable_bet
        self._bet_lock_acquired_at_spin = self._active_spin.spin_index
        self._bet_lock_source = source

    def _update_payout_resolution(self, frame: FramePacket, sig: FrameSignals) -> None:
        if self._active_spin is None:
            return

        regs = self.settings.regions
        self._active_spin.win_signal_detected = self._active_spin.win_signal_detected or self._detect_probable_win_signal(sig)
        now_mono = time.monotonic()
        if self._result_detected_mono is None:
            self._append_sampling_diag(
                "payout",
                {"code": "sampling_skipped_state", "ts_mono": now_mono},
                dedupe=True,
            )
            return
        payout_initial_delay = self.settings.detection.payout_sampling_initial_delay_ms / 1000.0
        payout_window = self.settings.detection.payout_sampling_window_ms / 1000.0
        payout_retry_interval = self.settings.detection.payout_sampling_retry_interval_ms / 1000.0
        if (now_mono - self._result_detected_mono) < payout_initial_delay:
            return
        if (now_mono - self._result_detected_mono) >= payout_window:
            self._append_sampling_diag(
                "payout",
                {"code": "sampling_window_expired", "ts_mono": now_mono},
                dedupe=True,
            )
            if self._active_spin.payout is None and self._active_spin.stabilization_fail_reason is None:
                self._active_spin.stabilization_fail_reason = "payout_window_expired_without_stabilization"
            return
        if self._active_spin.payout_read_attempts >= self.settings.detection.payout_stabilization_max_attempts:
            self._append_sampling_diag(
                "payout",
                {"code": "sampling_window_expired", "ts_mono": now_mono},
                dedupe=True,
            )
            if self._active_spin.payout is None and self._active_spin.stabilization_fail_reason is None:
                self._active_spin.stabilization_fail_reason = "payout_max_attempts_without_stabilization"
            return
        self._active_spin.payout_resolution_attempts += 1
        bet_window = self.settings.detection.bet_sampling_window_ms / 1000.0
        bet_retry_interval = self.settings.detection.bet_sampling_retry_interval_ms / 1000.0
        if (
            self._active_spin.bet_read_attempts < self.settings.detection.bet_sampling_max_attempts
            and (now_mono - self._result_detected_mono) < bet_window
            and (self._last_bet_sample_mono is None or (now_mono - self._last_bet_sample_mono) >= bet_retry_interval)
        ):
            self._active_spin.bet_read_attempts += 1
            self._last_bet_sample_mono = now_mono
            bet, bconf = self._attempt_sample_amount(frame, regs.bet_text, "bet")
            if bet is not None:
                self._active_spin.current_spin_raw_bet = bet
                self._bet_samples.append((bet, bconf))
                stable_bet, _stable_bet_conf, bet_source = self._stabilize_amount(
                    self._bet_samples,
                    min_confirmations=self.settings.detection.bet_stabilization_min_confirmations,
                    min_attempts=self.settings.detection.bet_stabilization_min_attempts,
                )
                if stable_bet is not None:
                    self._active_spin.bet = stable_bet
                    self._active_spin.bet_read_success = True
                    self._active_spin.chosen_bet_source = "ocr"
                    self._maybe_lock_session_bet(stable_bet, bet_source or "stabilized")

        if self._locked_session_bet is not None and self._active_spin.current_spin_raw_bet is not None:
            self._active_spin.bet_mismatch_vs_lock = abs(self._active_spin.current_spin_raw_bet - self._locked_session_bet) > 0.009

        if self._last_payout_sample_mono is not None and (now_mono - self._last_payout_sample_mono) < payout_retry_interval:
            return
        self._active_spin.payout_read_attempts += 1
        self._last_payout_sample_mono = now_mono
        payout, pconf = self._attempt_sample_amount(frame, regs.payout_text, "payout")
        if payout is not None:
            self._payout_samples.append((payout, pconf))
            stable_payout, stable_conf, payout_source = self._stabilize_amount(
                self._payout_samples,
                min_confirmations=self.settings.detection.payout_stabilization_min_confirmations,
                min_attempts=self.settings.detection.payout_stabilization_min_attempts,
            )
            if stable_payout is not None:
                self._active_spin.payout = stable_payout
                self._active_spin.payout_source = "ocr"
                self._active_spin.chosen_payout_source = "ocr"
                self._active_spin.payout_resolution_status = "confirmed"
                self._active_spin.confidence_payout = stable_conf
                self._active_spin.payout_stabilized_value_source = payout_source
                self._active_spin.detector_status = "confirmed"
                self._active_spin.reason = "payout_read_success"
                self._active_spin.payout_read_success = True

        if self.settings.detection.use_ocr_balance:
            bal, bconf = self._attempt_sample_amount(frame, regs.balance_text, "balance")
            if bal is not None:
                self._balance_samples.append((bal, bconf))
                stable_bal, stable_bal_conf = self._stable_amount(self._balance_samples)
                if stable_bal is not None:
                    if self._active_spin.balance_before is None:
                        self._active_spin.balance_before = stable_bal
                    else:
                        self._active_spin.balance_after = stable_bal
                        if self._active_spin.payout is None:
                            delta = self._active_spin.balance_after - self._active_spin.balance_before
                            if delta >= 0:
                                self._active_spin.payout = round(delta, 2)
                                self._active_spin.payout_source = "balance_delta"
                                self._active_spin.chosen_payout_source = "balance_delta"
                                self._active_spin.payout_resolution_status = "estimated"
                                self._active_spin.confidence_payout = stable_bal_conf
                                self._active_spin.detector_status = "partial"
                                self._active_spin.reason = "balance_delta_estimate"

        if self._active_spin.payout is None and sig.win is False and self._active_spin.payout_source == "unknown":
            self._active_spin.payout_resolution_status = "unresolved"

    def _append_sampling_diag(self, sample_kind: str, payload: dict[str, Any], *, dedupe: bool = False) -> None:
        if self._active_spin is None:
            return
        target: list[dict[str, Any]]
        if sample_kind == "payout":
            target = self._active_spin.payout_sampling_diagnostics
        elif sample_kind == "balance":
            target = self._active_spin.balance_sampling_diagnostics
        else:
            target = self._active_spin.bet_sampling_diagnostics
        if dedupe and target and target[-1].get("code") == payload.get("code"):
            return
        target.append(payload)

    def _attempt_sample_amount(self, frame: FramePacket, region: Region | None, sample_kind: str) -> tuple[float | None, float]:
        now_mono = time.monotonic()
        if region is None:
            self._append_sampling_diag(sample_kind, {"code": "roi_missing", "ts_mono": now_mono})
            return None, 0.0
        if frame.image_bgr is None:
            self._append_sampling_diag(sample_kind, {"code": "frame_none", "ts_mono": now_mono})
            return None, 0.0
        self._save_evidence_crop(frame, region, f"sample_{sample_kind}")
        crop = vision.crop_region(frame.image_bgr, region)
        if crop.size == 0:
            self._append_sampling_diag(sample_kind, {"code": "roi_empty", "ts_mono": now_mono})
            return None, 0.0
        try:
            text = vision.ocr_region_text(frame.image_bgr, region, lang=self.settings.detection.ocr_lang)
        except Exception:
            self._append_sampling_diag(sample_kind, {"code": "ocr_exception", "ts_mono": now_mono})
            return None, 0.0
        raw_attempt = {"ts_mono": now_mono, "raw_text": text}
        if sample_kind == "payout":
            self._active_spin.raw_payout_ocr_samples.append(text)
            self._active_spin.payout_raw_attempts.append(raw_attempt)
        elif sample_kind == "balance":
            self._active_spin.raw_balance_samples.append(text)
        else:
            self._active_spin.raw_bet_ocr_samples.append(text)
            self._active_spin.bet_raw_attempts.append(raw_attempt)
        if not text.strip():
            self._append_sampling_diag(sample_kind, {"code": "ocr_blank", "ts_mono": now_mono})
            return None, 0.0
        amount, conf = vision.parse_numeric_amount(text, hint=sample_kind)
        if amount is None:
            self._append_sampling_diag(sample_kind, {"code": "parse_failed", "ts_mono": now_mono, "raw_text": text})
            return None, 0.0
        self._append_sampling_diag(sample_kind, {"code": "sample_accepted", "ts_mono": now_mono, "confidence": conf})
        return amount, conf

    def _check_spin_timeouts(self, sig: FrameSignals, frame: FramePacket) -> None:
        det = self.settings.detection
        now = time.monotonic()
        if self._active_spin is None:
            return

        if self._awaiting_spinning_since is not None:
            has_spin_start_evidence = sig.reels_spinning or (
                sig.motion_score >= self.settings.detection.spinning_motion_threshold
            )
            if has_spin_start_evidence:
                self._spin_start_evidence_seen = True
            if self._spin_start_evidence_seen:
                self._awaiting_spinning_since = None
                if self._spinning_since is None:
                    self._spinning_since = now
                if self._active_spin.ts_spinning_detected is None:
                    self._active_spin.ts_spinning_detected = frame.ts
                    self._active_spin.detector_status = "partial"
                    self._active_spin.reason = "spinning_evidence_detected"
                return
            if (now - self._awaiting_spinning_since) >= det.click_to_spinning_timeout_sec:
                self._active_spin.timeouts.click_to_spinning = True
                self._emit(
                    SessionEventType.ERROR,
                    {
                        "type": "spin_timeout",
                        "spin_index": self._active_spin.spin_index,
                        "reason": "click_to_spinning_timeout",
                    },
                )
                self._finalize_spin_result(
                    detector_status="timeout",
                    reason="click_to_spinning_timeout",
                    payout=None,
                    visual_win=sig.win,
                    fallback_used=True,
                )
                self._armed_for_click = True
                return

        if self._spinning_since is not None:
            if (now - self._spinning_since) >= det.spinning_to_result_timeout_sec:
                self._active_spin.timeouts.spinning_to_result = True
                self._emit(
                    SessionEventType.ERROR,
                    {
                        "type": "spin_timeout",
                        "spin_index": self._active_spin.spin_index,
                        "reason": "spinning_to_result_timeout",
                    },
                )
                self._finalize_spin_result(
                    detector_status="timeout",
                    reason="spinning_to_result_timeout",
                    payout=None,
                    visual_win=sig.win,
                    fallback_used=True,
                )
                self._armed_for_click = True
                return

        if self._awaiting_ready_since is not None:
            if self.sm.state == BotState.POST_RESULT_ANIMATION:
                animation_started = self._post_result_animation_since or self._awaiting_ready_since
                if (now - animation_started) < det.result_animation_timeout_sec:
                    return
                timeout_reason = "post_result_animation_timeout"
            else:
                if (now - self._awaiting_ready_since) < det.result_to_ready_timeout_sec:
                    return
                timeout_reason = "ready_not_recovered"
            if self._active_spin is not None:
                self._save_evidence_crop(frame, self.settings.regions.spin_button, "ready_timeout_last_spin_button")
                debug_frames = list(self._result_ready_debug_frames)
                if debug_frames:
                    log.warning(
                        "result_to_ready timeout debug spin=%s state=%s reason=%s frames=%s",
                        self._active_spin.spin_index,
                        self.sm.state.value,
                        timeout_reason,
                        debug_frames,
                    )
                self._active_spin.timeouts.result_to_ready = True
                self._emit(
                    SessionEventType.ERROR,
                    {
                        "type": "spin_timeout",
                        "spin_index": self._active_spin.spin_index,
                        "reason": timeout_reason,
                        "debug_last_frames": debug_frames,
                    },
                )
                self._finalize_spin_result(
                    detector_status="timeout",
                    reason=timeout_reason,
                    payout=self._active_spin.payout,
                    visual_win=sig.win,
                    fallback_used=True,
                )
                self._armed_for_click = True

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

        post_result_recovery = False
        if self._awaiting_ready_since is not None:
            if reels_stopped and not popup:
                self._result_recovery_stable_frames += 1
            else:
                self._result_recovery_stable_frames = 0
            if self._result_recovery_stable_frames >= max(2, det.stopped_consecutive_frames):
                post_result_recovery = True
        else:
            self._result_recovery_stable_frames = 0

        confidences = {
            "motion": min(1.0, motion_val / max(1.0, det.spinning_motion_threshold * 2)),
            "popup": pconf,
            "win": wconf,
            "near_miss": nmconf,
            "bonus_tease": btconf,
            "bonus_trigger": btrconf,
            "spin_ready": srconf,
            "session_end": 0.8 if s_end else 0.0,
            "post_result_recovery": 0.45 if post_result_recovery else 0.0,
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
            post_result_ready_fallback=post_result_recovery,
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
                if self._active_spin is None:
                    self._start_spin_attempt(click_ts=None)
                if self._active_spin is not None:
                    self._active_spin.ts_spinning_detected = r.ts
                    self._active_spin.confidence_motion = r.confidence
                    self._active_spin.detector_status = "partial"
                    self._active_spin.reason = "spinning_detected"
                self._awaiting_spinning_since = None
                self._spinning_since = time.monotonic()
                self._emit(
                    SessionEventType.SPIN_STARTED,
                    {"spin_index": self._spin_counter, "confidence": r.confidence},
                )
            if r.from_state == BotState.SPINNING and r.to_state in (
                BotState.RESULT_WIN,
                BotState.RESULT_NO_WIN,
            ):
                self._emit(SessionEventType.SPIN_STOPPED, {"spin_index": self._spin_counter})
                self._spinning_since = None
                self._awaiting_ready_since = time.monotonic()
                self._result_recovery_stable_frames = 0
                self._result_ready_debug_frames.clear()
                self._result_spin_button_ready_evidence_saved = False
                self._post_result_animation_since = None
                self._post_result_animation_reason = None
                self._result_detected_mono = time.monotonic()
                if self._active_spin is not None:
                    self._active_spin.result_kind = "win" if r.to_state == BotState.RESULT_WIN else "no_win"
                    self._active_spin.ts_result_detected = r.ts
                    self._active_spin.post_result_animation_started_at = r.ts
            if r.from_state == BotState.SPINNING and r.to_state == BotState.READY_TO_SPIN:
                self._emit(SessionEventType.SPIN_STOPPED, {"spin_index": self._spin_counter})
                self._spinning_since = None
                self._awaiting_ready_since = time.monotonic()
                self._result_recovery_stable_frames = 0
                self._result_ready_debug_frames.clear()
                self._result_spin_button_ready_evidence_saved = False
                self._post_result_animation_since = None
                self._post_result_animation_reason = None
                self._result_detected_mono = time.monotonic()
                if self._active_spin is not None:
                    self._active_spin.ts_result_detected = r.ts
            if r.to_state == BotState.POST_RESULT_ANIMATION:
                now_mono = time.monotonic()
                if self._post_result_animation_since is None:
                    self._post_result_animation_since = now_mono
                if self._active_spin is not None:
                    self._post_result_animation_reason = (
                        "post_result_big_win_animation" if self._active_spin.visual_win else "post_result_animation"
                    )
                    self._active_spin.reason = self._post_result_animation_reason
                window = list(self._result_ready_debug_frames)[-5:]
                trend = [round(float(f.get("motion_score", 0.0)), 4) for f in window]
                self._emit(
                    SessionEventType.FRAME_TICK,
                    {
                        "type": "post_result_animation_entered",
                        "spin_index": self._spin_counter,
                        "duration_sec": round(now_mono - (self._awaiting_ready_since or now_mono), 3),
                        "motion_trend": trend,
                        "motion_score": r.detail.get("motion"),
                        "reason": self._post_result_animation_reason or "post_result_animation",
                    },
                )
            if r.to_state == BotState.RESULT_WIN:
                self._emit(
                    SessionEventType.WIN_DETECTED,
                    {"spin_index": self._spin_counter, "confidence": r.confidence},
                )
                if self._active_spin is not None:
                    self._active_spin.visual_win = True
                    self._active_spin.confidence_visual = r.confidence
                    self._active_spin.detector_status = "partial"
                    self._active_spin.reason = "awaiting_payout_resolution"
            if r.to_state == BotState.RESULT_NO_WIN:
                if r.reason == "near_miss_template":
                    self._emit(
                        SessionEventType.NEAR_MISS_DETECTED,
                        {"spin_index": self._spin_counter, "confidence": r.confidence},
                    )
                    if self._active_spin is not None:
                        self._active_spin.visual_win = False
                        self._active_spin.confidence_visual = r.confidence
                        self._active_spin.detector_status = "ambiguous"
                        self._active_spin.reason = "ambiguous_visual_signal"
                else:
                    if self._active_spin is not None:
                        self._active_spin.visual_win = False
                        self._active_spin.confidence_visual = r.confidence
                        self._active_spin.detector_status = "fallback"
                        self._active_spin.reason = "awaiting_payout_resolution"
            if r.to_state == BotState.BONUS_TEASE:
                self._emit(SessionEventType.BONUS_TEASE_DETECTED, {"confidence": r.confidence})
            if r.to_state == BotState.BONUS_TRIGGERED:
                self._emit(SessionEventType.BONUS_TRIGGERED, {"confidence": r.confidence})
                if self._active_spin is not None and self._awaiting_ready_since is not None:
                    self._post_result_animation_reason = "bonus_feature_animation"
                    self._active_spin.reason = "bonus_feature_animation"
                    self._emit(
                        SessionEventType.FRAME_TICK,
                        {
                            "type": "post_result_animation_bonus_feature",
                            "spin_index": self._spin_counter,
                            "duration_sec": round(
                                time.monotonic() - (self._awaiting_ready_since or time.monotonic()),
                                3,
                            ),
                            "reason": "bonus_feature_animation",
                        },
                    )
            if r.to_state == BotState.POPUP_BLOCKING:
                self._emit(SessionEventType.POPUP_DETECTED, {"confidence": r.confidence})
            if r.to_state == BotState.READY_TO_SPIN:
                self._armed_for_click = True
                if self._active_spin is not None:
                    self._active_spin.ts_ready_detected = r.ts
                    self._active_spin.confidence_ready = r.confidence
                    if self._awaiting_ready_since is not None:
                        duration_sec = max(0.0, time.monotonic() - self._awaiting_ready_since)
                        bonus_like_signal = (
                            self._post_result_animation_reason == "bonus_feature_animation"
                            or self.sm.state == BotState.BONUS_TRIGGERED
                        )
                        self._active_spin.post_result_animation_duration_sec = round(duration_sec, 3)
                        self._active_spin.post_result_visual_classification = self._classify_post_result_visual(
                            duration_sec,
                            bonus_like_signal=bonus_like_signal,
                        )
                self._finalize_on_ready = True
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
                if self._active_spin is not None and self._awaiting_ready_since is not None:
                    self._result_ready_debug_frames.append(
                        {
                            "frame_index": frame.frame_index,
                            "state": self.sm.state.value,
                            "spin_button_ready": sig.spin_button_ready,
                            "popup": sig.popup,
                            "win": sig.win,
                            "near_miss": sig.near_miss,
                            "motion_score": round(sig.motion_score, 4),
                            "confidences": dict(sig.confidences),
                        }
                    )
                    if any(
                        r.from_state == BotState.SPINNING and r.to_state in (BotState.RESULT_WIN, BotState.RESULT_NO_WIN)
                        for r in recs
                    ):
                        self._save_evidence_crop(frame, self.settings.regions.spin_button, "first_result_spin_button")
                    if sig.spin_button_ready and not self._result_spin_button_ready_evidence_saved:
                        self._save_evidence_crop(frame, self.settings.regions.spin_button, "ready_signal_spin_button")
                        self._result_spin_button_ready_evidence_saved = True
                if self._active_spin is not None and self._awaiting_ready_since is not None and not self._result_evidence_saved:
                    self._save_evidence_crop(frame, self.settings.regions.bet_text, "first_result_bet")
                    self._save_evidence_crop(frame, self.settings.regions.payout_text, "first_result_payout")
                    self._save_evidence_crop(frame, self.settings.regions.balance_text, "first_result_balance")
                    self._result_evidence_saved = True
                self._update_payout_resolution(frame, sig)
                if self._finalize_on_ready and self._active_spin is not None and self.sm.state == BotState.READY_TO_SPIN:
                    if (
                        self._result_detected_mono is not None
                        and self._active_spin.payout_read_attempts == 0
                        and (time.monotonic() - self._result_detected_mono)
                        < (self.settings.detection.payout_sampling_window_ms / 1000.0)
                    ):
                        continue
                    probable_win_signal = self._detect_probable_win_signal(sig)
                    self._active_spin.win_signal_detected = self._active_spin.win_signal_detected or probable_win_signal
                    if self._should_defer_ready_finalize(probable_win_signal):
                        continue
                    self._save_evidence_crop(frame, self.settings.regions.bet_text, "ready_bet")
                    self._save_evidence_crop(frame, self.settings.regions.payout_text, "ready_payout")
                    self._save_evidence_crop(frame, self.settings.regions.balance_text, "ready_balance")
                    finalize_reason = self._active_spin.reason if self._active_spin.payout is not None else "payout_not_readable"
                    if self._active_spin.payout is None and probable_win_signal:
                        finalize_reason = "win_unreadable"
                    if self._post_result_animation_reason:
                        finalize_reason = self._post_result_animation_reason
                    observation_payload = self._detect_symbol_observations(frame)
                    for key, value in observation_payload.items():
                        setattr(self._active_spin, key, value)
                    self._finalize_spin_result(
                        detector_status=self._active_spin.detector_status,
                        reason=finalize_reason,
                        payout=self._active_spin.payout,
                        visual_win=self._active_spin.visual_win,
                        fallback_used=self._active_spin.payout is None,
                        probable_win_signal=probable_win_signal,
                    )
                self._check_spin_timeouts(sig, frame)

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
                        click_ts = _utcnow()
                        self._start_spin_attempt(click_ts=click_ts)
                        self._save_evidence_crop(frame, self.settings.regions.bet_text, "before_click_bet")
                        self._save_evidence_crop(frame, self.settings.regions.payout_text, "before_click_payout")
                        self._save_evidence_crop(frame, self.settings.regions.balance_text, "before_click_balance")
                        if self.settings.detection.use_ocr_balance and self._active_spin is not None:
                            bal, bconf = self._read_amount(frame, self.settings.regions.balance_text)
                            if bal is not None:
                                self._active_spin.balance_before = bal
                                self._active_spin.confidence_payout = bconf
                        self._emit(
                            SessionEventType.CLICK_DRY_RUN,
                            {"x": x, "y": y, "spin_index": self._spin_counter},
                        )
                        self._armed_for_click = False
                    else:
                        random_delay(
                            self.settings.automation.min_delay_sec,
                            self.settings.automation.max_delay_sec,
                        )
                        pt = self.clicker.click_spin()
                        if pt is not None:
                            click_ts = _utcnow()
                            self._start_spin_attempt(click_ts=click_ts)
                            self._save_evidence_crop(frame, self.settings.regions.bet_text, "before_click_bet")
                            self._save_evidence_crop(frame, self.settings.regions.payout_text, "before_click_payout")
                            self._save_evidence_crop(frame, self.settings.regions.balance_text, "before_click_balance")
                            if self.settings.detection.use_ocr_balance and self._active_spin is not None:
                                bal, bconf = self._read_amount(frame, self.settings.regions.balance_text)
                                if bal is not None:
                                    self._active_spin.balance_before = bal
                                    self._active_spin.confidence_payout = bconf
                            self._emit(
                                SessionEventType.CLICK_LIVE,
                                {"x": pt[0], "y": pt[1], "spin_index": self._spin_counter},
                            )
                            self._armed_for_click = False

                frame_index += 1
                self.capture.sleep_for_fps(self.fps, loop_start)

        except KeyboardInterrupt:
            self._emit(SessionEventType.SESSION_STOPPED, {"reason": "keyboard_interrupt"})
        finally:
            ended = _utcnow()
            if self._active_spin is not None:
                self._finalize_spin_result(
                    detector_status="fallback",
                    reason="result_unknown_fallback",
                    payout=None,
                    visual_win=self._active_spin.visual_win,
                    fallback_used=True,
                )
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
