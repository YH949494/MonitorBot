from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .detector import run_auto_detection
from .overlay import draw_detection_overlay
from .profile import save_profile, validate_profile

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("calibration")


def _capture_screen() -> np.ndarray | None:
    try:
        import mss  # type: ignore

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            frame = np.array(sct.grab(monitor))[:, :, :3]
            return frame
    except Exception as exc:
        logger.error(f"[CALIBRATION][CAPTURE_FAILED] reason={exc}")
        return None



def main() -> int:
    if cv2 is None:
        logger.error("[CALIBRATION][DEPENDENCY_MISSING] opencv-python is required")
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--capture-screen", action="store_true")
    parser.add_argument("--profile-name", default="auto_detected")
    args = parser.parse_args()

    logger.info("[CALIBRATION][START]")
    frame = None
    if args.source:
        frame = cv2.imread(args.source)
        if frame is None:
            logger.error(f"[CALIBRATION][SOURCE_LOAD_FAILED] path={args.source}")
            return 1
    elif args.capture_screen:
        frame = _capture_screen()
        if frame is None:
            logger.error("[CALIBRATION][NO_FRAME] capture unavailable")
            return 1
    else:
        logger.error("Provide --source <image> or --capture-screen")
        return 1

    h, w = frame.shape[:2]
    logger.info(f"[CALIBRATION][FRAME_CAPTURED] width={w} height={h}")
    detected = run_auto_detection(frame)

    for key in ("reels", "spin_button", "popup_close"):
        item = detected[key]
        logger.info(f"[CALIBRATION][{key.upper()}_DETECTED] confidence={item['confidence']:.2f} box={item['box']} reason={item['reason']}")
        if item["confidence"] < 0.75:
            logger.info(f"[CALIBRATION][LOW_CONFIDENCE] region={key} confidence={item['confidence']:.2f} reason={item['reason']}")

    logger.info(f"[CALIBRATION][GRID_DETECTED] confidence={detected['grid']['confidence']:.2f} rows={detected['grid']['rows']} cols={detected['grid']['cols']}")

    profile = {
        "game_profile": args.profile_name,
        "calibrated": detected["reels"]["confidence"] >= 0.75 and detected["spin_button"]["confidence"] >= 0.75,
        "capture": {
            "mode": "window_or_region",
            "window_title_contains": "",
            "region": {"left": 0, "top": 0, "width": w, "height": h},
            "fps": 8.0,
            "coordinate_mode": "relative_to_capture",
        },
        "regions": {
            "reels": {**detected["reels"]["box"], "confidence": detected["reels"]["confidence"], "source": "auto"},
            "spin_button": {**detected["spin_button"]["box"], "confidence": detected["spin_button"]["confidence"], "source": "auto"},
            "popup_close": {**detected["popup_close"]["box"], "confidence": detected["popup_close"]["confidence"], "source": "auto"},
        },
        "grid": {"rows": detected["grid"]["rows"], "cols": detected["grid"]["cols"], "confidence": detected["grid"]["confidence"], "source": "auto"},
        "manual_required": any(detected[k]["confidence"] < 0.45 for k in ("reels", "spin_button")),
    }

    ok, errors = validate_profile(profile)
    if not ok:
        profile["calibrated"] = False
        logger.info(f"[CALIBRATION][PROFILE_INVALID] errors={errors}")

    preview = draw_detection_overlay(frame, detected, detected.get("grid"))
    Path("sessions").mkdir(parents=True, exist_ok=True)
    preview_path = Path("sessions") / "calibration_preview.png"
    cv2.imwrite(str(preview_path), preview)

    saved_path = save_profile(args.profile_name, profile)
    logger.info(f"[CALIBRATION][PROFILE_SAVED] path={saved_path}")
    print(f"confidence_summary reels={detected['reels']['confidence']:.2f} spin={detected['spin_button']['confidence']:.2f} popup={detected['popup_close']['confidence']:.2f} grid={detected['grid']['confidence']:.2f}")
    print(f"preview_image={preview_path}")
    print(f"profile_json={saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
