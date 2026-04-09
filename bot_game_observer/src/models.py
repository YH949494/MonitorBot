"""Typed domain models for regions, events, detection, and session summaries."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Region(BaseModel):
    """Rectangle in pixels (left, top, width, height)."""

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def clip_to(self, max_w: int, max_h: int) -> "Region":
        """Clamp size so the region stays inside a box of max_w x max_h from origin."""
        w = min(self.width, max(0, max_w - self.left))
        h = min(self.height, max(0, max_h - self.top))
        return Region(left=self.left, top=self.top, width=max(1, w), height=max(1, h))


class CoordinateMode(str, Enum):
    RELATIVE_TO_CAPTURE = "relative_to_capture"
    ABSOLUTE_SCREEN = "absolute_screen"


class CaptureMode(str, Enum):
    REGION = "region"
    WINDOW = "window"


class CaptureConfig(BaseModel):
    mode: CaptureMode = CaptureMode.REGION
    window_title_contains: str = ""
    region: Region
    fps: float = Field(default=8.0, gt=0, le=60)
    coordinate_mode: CoordinateMode = CoordinateMode.RELATIVE_TO_CAPTURE


class TemplateSpec(BaseModel):
    path: str
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class DetectionConfig(BaseModel):
    spinning_motion_threshold: float = Field(default=12.0, ge=0.0)
    stopped_consecutive_frames: int = Field(default=3, ge=1)
    state_debounce_frames: int = Field(default=2, ge=1)
    motion_history_frames: int = Field(default=3, ge=1)
    near_miss_cooldown_sec: float = Field(default=2.0, ge=0.0)
    bonus_tease_cooldown_sec: float = Field(default=2.0, ge=0.0)
    popup_persist_frames: int = Field(default=15, ge=1)
    min_confidence_for_transition: float = Field(default=0.35, ge=0.0, le=1.0)
    # Watchdog timeouts to avoid hanging spin lifecycle states.
    click_to_spinning_timeout_sec: float = Field(default=2.0, ge=0.1)
    spinning_to_result_timeout_sec: float = Field(default=6.0, ge=0.1)
    result_to_ready_timeout_sec: float = Field(default=4.0, ge=0.1)
    result_animation_timeout_sec: float = Field(default=12.0, ge=0.1)
    post_result_normal_threshold_sec: float = Field(default=1.0, ge=0.0)
    post_result_long_animation_threshold_sec: float = Field(default=3.0, ge=0.1)
    post_result_bonus_like_threshold_sec: float = Field(default=6.0, ge=0.1)
    # Deprecated legacy payout OCR knobs kept for backward compatibility.
    payout_read_delay_sec: float = Field(default=0.25, ge=0.0)
    payout_read_retry_window_sec: float = Field(default=1.0, ge=0.0)
    payout_read_max_attempts: int = Field(default=5, ge=1)
    payout_sampling_initial_delay_ms: int = Field(default=250, ge=0)
    payout_sampling_retry_interval_ms: int = Field(default=150, ge=0)
    payout_sampling_window_ms: int = Field(default=1000, ge=1)
    payout_stabilization_min_confirmations: int = Field(default=2, ge=2)
    payout_stabilization_min_attempts: int = Field(default=2, ge=1)
    payout_stabilization_max_attempts: int = Field(default=6, ge=1)
    bet_sampling_retry_interval_ms: int = Field(default=120, ge=0)
    bet_sampling_window_ms: int = Field(default=1000, ge=1)
    bet_stabilization_min_confirmations: int = Field(default=2, ge=2)
    bet_stabilization_min_attempts: int = Field(default=2, ge=1)
    bet_sampling_max_attempts: int = Field(default=6, ge=1)
    bet_lock_max_spins_from_session_start: int = Field(default=5, ge=1)
    payout_evidence_mode: bool = False
    payout_evidence_dir: str = "logs/payout_evidence"
    scatter_trigger_count: int = Field(default=4, ge=1)
    bonus_trigger_count: int = Field(default=4, ge=1)
    scatter_min_count_for_signal: int = Field(default=2, ge=1)
    bonus_min_count_for_signal: int = Field(default=2, ge=1)
    scatter_min_score_for_signal: float = Field(default=0.96, ge=0.0, le=1.0)
    bonus_min_score_for_signal: float = Field(default=0.96, ge=0.0, le=1.0)
    symbol_match_center_merge_px: int = Field(default=24, ge=0)
    symbol_max_count_cap: int = Field(default=12, ge=1)
    symbol_capture_cooldown_sec: float = Field(default=2.0, ge=0.0)
    symbol_debug_mode: bool = False
    symbol_debug_max_spins: int = Field(default=10, ge=0)
    symbol_debug_save_reels_crop: bool = True
    use_ocr_balance: bool = False
    ocr_lang: str = "eng"

    @model_validator(mode="after")
    def _bridge_legacy_payout_settings(self) -> "DetectionConfig":
        # New payout sampling/stabilization keys take precedence when explicitly set.
        # Legacy keys map into new keys only when legacy was explicitly provided and new was not.
        fields_set = self.model_fields_set
        legacy_used = False
        legacy_overridden = False
        if "payout_read_delay_sec" in fields_set and "payout_sampling_initial_delay_ms" not in fields_set:
            self.payout_sampling_initial_delay_ms = int(round(self.payout_read_delay_sec * 1000.0))
            legacy_used = True
        elif "payout_read_delay_sec" in fields_set and "payout_sampling_initial_delay_ms" in fields_set:
            legacy_overridden = True
        if "payout_read_retry_window_sec" in fields_set and "payout_sampling_window_ms" not in fields_set:
            self.payout_sampling_window_ms = int(round(self.payout_read_retry_window_sec * 1000.0))
            legacy_used = True
        elif "payout_read_retry_window_sec" in fields_set and "payout_sampling_window_ms" in fields_set:
            legacy_overridden = True
        if "payout_read_max_attempts" in fields_set and "payout_stabilization_max_attempts" not in fields_set:
            self.payout_stabilization_max_attempts = self.payout_read_max_attempts
            legacy_used = True
        elif "payout_read_max_attempts" in fields_set and "payout_stabilization_max_attempts" in fields_set:
            legacy_overridden = True
        if legacy_used:
            log.warning(
                "Legacy payout config keys applied to new payout sampling/stabilization settings."
            )
        if legacy_overridden:
            log.warning(
                "Both legacy and new payout config keys set; new payout settings took precedence."
            )
        return self


class AutomationConfig(BaseModel):
    enable_clicking: bool = False
    click_jitter_px: int = Field(default=6, ge=0)
    min_delay_sec: float = Field(default=0.8, ge=0.0)
    max_delay_sec: float = Field(default=1.8, ge=0.0)
    max_clicks_per_minute: int = Field(default=40, ge=1)
    panic_stop_file: str = "logs/STOP.txt"


class SessionLimitsConfig(BaseModel):
    max_spins: int = Field(default=300, ge=1)
    max_duration_min: float = Field(default=45.0, ge=0.1)
    max_consecutive_errors: int = Field(default=10, ge=1)


class OutputConfig(BaseModel):
    """Legacy field; session logs/reports use fixed portable folders under app root."""

    base_dir: str = "."


class GameRegions(BaseModel):
    """Named regions; coordinates per capture `coordinate_mode`."""

    spin_button: Region
    reels: Region
    popup_close: Region | None = Field(default=None)
    win_banner: Region | None = Field(default=None)
    bonus_indicator: Region | None = Field(default=None)
    bet_text: Region | None = Field(default=None)
    payout_text: Region | None = Field(default=None)
    balance_text: Region | None = Field(default=None)
    session_end: Region | None = Field(default=None)


class BotSettings(BaseModel):
    """Full validated configuration for a game profile."""

    game_profile: str = "generic_reel_game"
    calibrated: bool = False
    capture: CaptureConfig
    regions: GameRegions
    templates: dict[str, TemplateSpec] = Field(default_factory=dict)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    automation: AutomationConfig = Field(default_factory=AutomationConfig)
    session: SessionLimitsConfig = Field(default_factory=SessionLimitsConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("templates", mode="before")
    @classmethod
    def empty_templates_ok(cls, v: Any) -> Any:
        return v or {}


class DetectorKind(str, Enum):
    TEMPLATE = "template"
    MOTION = "motion"
    PIXEL_DIFF = "pixel_diff"
    OCR = "ocr"
    UNKNOWN = "unknown"


class DetectorResult(BaseModel):
    name: str
    kind: DetectorKind = DetectorKind.UNKNOWN
    active: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detail: dict[str, Any] = Field(default_factory=dict)


class BotState(str, Enum):
    IDLE = "IDLE"
    READY_TO_SPIN = "READY_TO_SPIN"
    SPINNING = "SPINNING"
    RESULT_WIN = "RESULT_WIN"
    RESULT_NO_WIN = "RESULT_NO_WIN"
    POST_RESULT_ANIMATION = "POST_RESULT_ANIMATION"
    BONUS_TEASE = "BONUS_TEASE"
    BONUS_TRIGGERED = "BONUS_TRIGGERED"
    POPUP_BLOCKING = "POPUP_BLOCKING"
    SESSION_ENDED = "SESSION_ENDED"
    ERROR = "ERROR"


class SessionEventType(str, Enum):
    SPIN_STARTED = "spin_started"
    SPIN_STOPPED = "spin_stopped"
    WIN_DETECTED = "win_detected"
    NO_WIN_DETECTED = "no_win_detected"
    NEAR_MISS_DETECTED = "near_miss_detected"
    BONUS_TEASE_DETECTED = "bonus_tease_detected"
    BONUS_TRIGGERED = "bonus_triggered"
    POPUP_DETECTED = "popup_detected"
    SESSION_STOPPED = "session_stopped"
    STATE_TRANSITION = "state_transition"
    CLICK_DRY_RUN = "click_dry_run"
    CLICK_LIVE = "click_live"
    ERROR = "error"
    FRAME_TICK = "frame_tick"
    CALIBRATION = "calibration"
    SPIN_RESULT_SUMMARY = "spin_result_summary"


class SessionEvent(BaseModel):
    session_id: str
    ts: datetime = Field(default_factory=utcnow)
    event_type: SessionEventType | str
    payload: dict[str, Any] = Field(default_factory=dict)

    def model_dump_jsonl(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        if isinstance(d.get("event_type"), str):
            pass
        return d


class SessionSummary(BaseModel):
    session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    duration_sec: float = 0.0
    total_spins: int = 0
    total_wins: int = 0
    total_no_win: int = 0
    unreadable_win_count: int = 0
    finalized_non_no_win_count: int = 0
    visual_win_count: int = 0
    any_payout_count: int = 0
    real_win_count: int = 0
    break_even_count: int = 0
    net_loss_with_payout_count: int = 0
    no_payout_count: int = 0
    result_unknown_count: int = 0
    click_to_spinning_timeout_count: int = 0
    spinning_to_result_timeout_count: int = 0
    result_to_ready_timeout_count: int = 0
    visual_win_rate: float = 0.0
    any_payout_rate: float = 0.0
    real_win_rate: float = 0.0
    first_win_spin_index: int | None = None
    first_readable_win_spin_index: int | None = None
    first_finalized_non_no_win_spin_index: int | None = None
    spins_before_first_win: int | None = None
    gaps_between_wins: list[int] = Field(default_factory=list)
    avg_spins_between_wins: float | None = None
    max_spins_between_wins: int | None = None
    near_miss_count: int = 0
    near_miss_rate: float = 0.0
    empty_spin_count: int = 0
    empty_spin_rate: float = 0.0
    visual_win_spin_indices: list[int] = Field(default_factory=list)
    big_win_count: int = 0
    big_win_rate: float = 0.0
    big_win_spin_indices: list[int] = Field(default_factory=list)
    missing_payout_count: int = 0
    missing_payout_rate: float = 0.0
    payout_truth_conflict_count: int = 0
    ocr_balance_agree_count: int = 0
    balance_delta_confirmed_count: int = 0
    payout_effective_resolved_count: int = 0
    payout_effective_unresolved_count: int = 0
    balance_backed_payout_count: int = 0
    ocr_only_payout_count: int = 0
    payout_conflict_count: int = 0
    session_trust_score: float | None = None
    session_trust_label: str | None = None
    coverage_ratio: float | None = None
    session_quality: str | None = None
    usable_spin_count: int = 0
    usable_spin_ratio: float | None = None
    session_valid_for_analysis: bool = True
    session_exclusion_reason: str | None = None
    conflict_spin_indices: list[int] = Field(default_factory=list)
    unresolved_spin_indices: list[int] = Field(default_factory=list)
    consecutive_conflict_spins_max: int = 0
    consecutive_unresolved_spins_max: int = 0
    locked_session_bet: float | None = None
    bonus_tease_count: int = 0
    bonus_trigger_count: int = 0
    end_reason: str = ""
    consecutive_no_win_streak_max: int = 0
    total_wins_currency: float | None = None
    total_losses_currency: float | None = None
    confidence_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def new_session_id() -> str:
    return str(uuid4())


class FramePacket(BaseModel):
    """Single captured frame with optional precomputed grayscale."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: datetime
    frame_index: int
    image_bgr: Any  # numpy array
    gray: Any | None = None
