from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from src.models import FramePacket
from src.spin_result import SpinResult
from src.state_machine import FrameSignals


def run_spin_replay(ocr_samples: list[dict[str, str]], visual_win: bool) -> dict:
    import sys
    import types

    if "pyautogui" not in sys.modules:
        sys.modules["pyautogui"] = types.SimpleNamespace(
            moveTo=lambda *_args, **_kwargs: None,
            click=lambda *_args, **_kwargs: None,
        )
    if "cv2" not in sys.modules:
        cv2_stub = types.SimpleNamespace(imwrite=lambda *_args, **_kwargs: True, __version__="4.0.0")
        sys.modules["cv2"] = cv2_stub  # type: ignore[assignment]
    from src.session_runner import SessionRunner

    runner = SessionRunner.__new__(SessionRunner)
    runner.settings = SimpleNamespace(
        detection=SimpleNamespace(
            ocr_lang="eng",
            use_ocr_balance=True,
            payout_evidence_mode=False,
            result_to_ready_timeout_sec=1.0,
        ),
        regions=SimpleNamespace(bet_text=object(), payout_text=object(), balance_text=object()),
    )
    runner._active_spin = SpinResult(spin_index=1, visual_win=visual_win, bet=2.0)
    runner._awaiting_ready_since = 0.1
    runner._awaiting_spinning_since = None
    runner._spinning_since = None
    runner._payout_samples = deque(maxlen=3)
    runner._balance_samples = deque(maxlen=3)
    runner._result_evidence_saved = False
    runner._finalize_on_ready = False
    runner._payout_evidence_dir = None
    events: list[dict] = []
    runner._emit = lambda event_type, payload: events.append(payload)  # type: ignore[assignment]

    import src.vision as vision

    original = vision.ocr_region_text
    current_sample: dict[str, str] = {}
    try:
        def by_region(_img, region, lang="eng"):
            if region is runner.settings.regions.bet_text:
                return current_sample.get("bet", "")
            if region is runner.settings.regions.payout_text:
                return current_sample.get("payout", "")
            return current_sample.get("balance", "")
        vision.ocr_region_text = by_region  # type: ignore[assignment]
        for idx, sample in enumerate(ocr_samples):
            current_sample = sample
            frame = FramePacket(
                ts=datetime.now(timezone.utc),
                frame_index=idx,
                image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
            )
            sig = FrameSignals(
                ts=frame.ts,
                frame_index=idx,
                motion_score=0.0,
                reels_spinning=False,
                reels_stopped=True,
                popup=False,
                win=visual_win,
                no_win_hint=not visual_win,
                bonus_tease=False,
                bonus_trigger=False,
                near_miss=False,
                session_end=False,
                spin_button_ready=False,
                confidences={},
            )
            runner._update_payout_resolution(frame, sig)
    finally:
        vision.ocr_region_text = original  # type: ignore[assignment]

    runner._finalize_spin_result(
        detector_status=runner._active_spin.detector_status,
        reason=runner._active_spin.reason if runner._active_spin.payout is not None else "payout_not_readable",
        payout=runner._active_spin.payout,
        visual_win=runner._active_spin.visual_win,
        fallback_used=runner._active_spin.payout is None,
    )
    return events[-1]
