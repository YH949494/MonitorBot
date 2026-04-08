from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np

if "pyautogui" not in sys.modules:
    sys.modules["pyautogui"] = types.SimpleNamespace(
        moveTo=lambda *_args, **_kwargs: None,
        click=lambda *_args, **_kwargs: None,
    )
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(
        imwrite=lambda *_args, **_kwargs: True,
        rectangle=lambda *_args, **_kwargs: None,
        putText=lambda *_args, **_kwargs: None,
        cvtColor=lambda img, *_args, **_kwargs: img[..., 0],
        matchTemplate=lambda *_args, **_kwargs: np.zeros((1, 1), dtype=np.float32),
        COLOR_BGR2GRAY=0,
        TM_CCOEFF_NORMED=0,
        FONT_HERSHEY_SIMPLEX=0,
        LINE_AA=0,
        __version__="4.0.0",
    )

from src.models import FramePacket, Region
from src.session_runner import SessionRunner
from src.spin_result import SpinResult
from src import vision


def _runner(tmp_path: Path) -> SessionRunner:
    runner = SessionRunner.__new__(SessionRunner)
    runner._active_spin = SpinResult(spin_index=7, result_kind="no_win")
    runner.session_id = "s1"
    runner._session_frames_dir = tmp_path / "frames"
    runner._session_debug_dir = tmp_path / "debug"
    runner._last_symbol_capture_by_reason = {}
    runner._symbol_debug_saved_spins = 0
    runner.settings = SimpleNamespace(
        regions=SimpleNamespace(reels=Region(left=2, top=2, width=6, height=6)),
        detection=SimpleNamespace(
            scatter_trigger_count=4,
            bonus_trigger_count=4,
            scatter_min_count_for_signal=2,
            bonus_min_count_for_signal=2,
            scatter_min_score_for_signal=0.96,
            bonus_min_score_for_signal=0.96,
            symbol_match_center_merge_px=3,
            symbol_max_count_cap=5,
            symbol_capture_cooldown_sec=2.0,
            symbol_debug_mode=False,
            symbol_debug_max_spins=10,
            symbol_debug_save_reels_crop=True,
        ),
        templates={
            "scatter_symbol": SimpleNamespace(threshold=0.90),
            "bonus_symbol": SimpleNamespace(threshold=0.90),
        },
    )
    runner._tmpl_cache = {
        "scatter_symbol": np.zeros((2, 2), dtype=np.uint8),
        "bonus_symbol": np.zeros((2, 2), dtype=np.uint8),
    }
    return runner


def _frame() -> FramePacket:
    img = np.zeros((12, 12, 3), dtype=np.uint8)
    img[2:8, 2:8] = 100
    return FramePacket(ts=datetime.now(timezone.utc), frame_index=1, image_bgr=img)


def test_weak_match_does_not_promote_reason_or_capture(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr("src.session_runner.vision.template_match_best", lambda *_args, **_kwargs: (0.95, (1, 1)))
    monkeypatch.setattr(
        "src.session_runner.vision.template_match_locations",
        lambda *_args, **_kwargs: (True, [{"x": 1, "y": 1, "w": 2, "h": 2}], [0.95]),
    )
    writes: list[np.ndarray] = []
    monkeypatch.setattr("src.session_runner.cv2.imwrite", lambda _p, img: writes.append(img.copy()) or True)
    out = runner._detect_symbol_observations(_frame())
    assert out["scatter_count"] == 1
    assert out["scatter_detect_ok"] is True
    assert out["symbol_detection_reason_flags"] is None
    assert out["symbol_detection_frame_path"] is None
    assert out["scatter_debug_ran"] is True
    assert out["scatter_debug_reason"] == "match_scored"
    assert out["scatter_debug_best_score"] == 0.95
    assert out["scatter_debug_best_loc"] == [1, 1]
    assert writes == []


def test_bonus_template_missing_is_safe(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner._tmpl_cache["bonus_symbol"] = None

    def _fake_match(_scene, tmpl, *_args, **_kwargs):
        if tmpl is None:
            return False, [], []
        return True, [], []

    monkeypatch.setattr("src.session_runner.vision.template_match_locations", _fake_match)
    out = runner._detect_symbol_observations(_frame())
    assert out["bonus_detect_ok"] is False
    assert out["bonus_count"] is None
    assert out["bonus_boxes"] is None
    assert out["bonus_match_scores"] is None
    assert out["bonus_tease"] is False
    assert out["symbol_detection_reason_flags"] is None


def test_capture_uses_reel_crop_and_overlay_and_cooldown(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner._tmpl_cache["bonus_symbol"] = None
    saved: list[np.ndarray] = []
    monkeypatch.setattr("src.session_runner.cv2.imwrite", lambda _p, img: saved.append(img.copy()) or True)

    def _rectangle(img, p1, p2, _color, _thick):
        img[p1[1] : p2[1], p1[0] : p2[0]] = 255

    monkeypatch.setattr("src.session_runner.cv2.rectangle", _rectangle)
    monkeypatch.setattr("src.session_runner.cv2.putText", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 10.0)
    def _fake_match(_scene, tmpl, *_args, **_kwargs):
        if tmpl is None:
            return False, [], []
        return True, [{"x": 1, "y": 1, "w": 2, "h": 2}, {"x": 3, "y": 1, "w": 2, "h": 2}, {"x": 5, "y": 1, "w": 2, "h": 2}], [0.99, 0.98, 0.97]

    monkeypatch.setattr("src.session_runner.vision.template_match_locations", _fake_match)

    out1 = runner._detect_symbol_observations(_frame())
    assert out1["symbol_detection_frame_path"] is not None
    assert out1["symbol_detection_frame_ts"] is not None
    assert saved[0].shape[:2] == (6, 6)
    plain_crop = _frame().image_bgr[2:8, 2:8]
    assert np.any(saved[0] != plain_crop)

    monkeypatch.setattr("src.session_runner.time.monotonic", lambda: 11.0)
    out2 = runner._detect_symbol_observations(_frame())
    assert out2["scatter_count"] == 3
    assert out2["symbol_detection_reason_flags"] == ["scatter_detected", "scatter_near_miss"]
    assert out2["symbol_detection_frame_path"] is None
    assert out2["symbol_detection_frame_ts"] is None
    assert len(saved) == 1


def test_win_without_symbol_reasons_does_not_trigger_symbol_capture(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner._active_spin = SpinResult(spin_index=7, result_kind="win")
    monkeypatch.setattr("src.session_runner.vision.template_match_best", lambda *_args, **_kwargs: (0.10, (0, 0)))
    monkeypatch.setattr("src.session_runner.vision.template_match_locations", lambda *_args, **_kwargs: (True, [], []))
    out = runner._detect_symbol_observations(_frame())
    assert out["symbol_detection_reason_flags"] is None
    assert out["symbol_detection_frame_path"] is None
    assert out["symbol_detection_frame_ts"] is None
    assert out["scatter_debug_ran"] is True
    assert out["scatter_debug_reason"] == "match_scored"


def test_missing_scatter_template_debug_case(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    del runner.settings.templates["scatter_symbol"]
    runner._tmpl_cache["scatter_symbol"] = None
    monkeypatch.setattr("src.session_runner.vision.template_match_locations", lambda *_args, **_kwargs: (True, [], []))
    out = runner._detect_symbol_observations(_frame())
    assert out["scatter_detect_ok"] is False
    assert out["scatter_count"] is None
    assert out["scatter_debug_template_present"] is False
    assert out["scatter_debug_ran"] is False
    assert out["scatter_debug_reason"] == "missing_template_spec"


def test_debug_reels_crop_saving_enabled(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.settings.detection.symbol_debug_mode = True
    runner.settings.detection.symbol_debug_max_spins = 2
    runner.settings.detection.symbol_debug_save_reels_crop = True
    monkeypatch.setattr("src.session_runner.vision.template_match_locations", lambda *_args, **_kwargs: (True, [], []))
    saved: list[tuple[str, np.ndarray]] = []
    monkeypatch.setattr(
        "src.session_runner.cv2.imwrite",
        lambda path, img: saved.append((path, img.copy())) or True,
    )
    out = runner._detect_symbol_observations(_frame())
    assert out["scatter_debug_reels_path"] is not None
    assert out["scatter_debug_reels_path"].endswith("spin_0007_reels_debug.png")
    assert saved
    assert saved[0][1].shape[:2] == (6, 6)


def test_debug_reels_crop_not_saved_when_mode_disabled(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.settings.detection.symbol_debug_mode = False
    monkeypatch.setattr("src.session_runner.vision.template_match_locations", lambda *_args, **_kwargs: (True, [], []))
    writes: list[np.ndarray] = []
    monkeypatch.setattr("src.session_runner.cv2.imwrite", lambda _path, img: writes.append(img.copy()) or True)
    out = runner._detect_symbol_observations(_frame())
    assert out["scatter_debug_reels_path"] is None
    assert writes == []


def test_template_larger_than_scene_debug_reason(monkeypatch, tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner._tmpl_cache["scatter_symbol"] = np.zeros((20, 20), dtype=np.uint8)
    monkeypatch.setattr("src.session_runner.vision.template_match_locations", lambda *_args, **_kwargs: (False, [], []))
    out = runner._detect_symbol_observations(_frame())
    assert out["scatter_debug_ran"] is False
    assert out["scatter_debug_reason"] == "template_larger_than_scene"
    assert out["scatter_debug_best_score"] is None
    assert out["scatter_debug_best_loc"] is None


def test_template_match_locations_center_merge_and_cap(monkeypatch) -> None:
    scene = np.zeros((10, 10), dtype=np.uint8)
    tmpl = np.zeros((2, 2), dtype=np.uint8)
    score_map = np.zeros((9, 9), dtype=np.float32)
    score_map[1, 1] = 0.99
    score_map[1, 2] = 0.98
    score_map[1, 7] = 0.97
    monkeypatch.setattr(vision.cv2, "matchTemplate", lambda *_args, **_kwargs: score_map)
    ok, boxes, scores = vision.template_match_locations(
        scene,
        tmpl,
        0.96,
        max_matches=2,
        center_merge_px=2,
    )
    assert ok is True
    assert len(boxes) == 2
    assert len(scores) == 2
