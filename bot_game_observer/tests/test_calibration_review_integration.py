from __future__ import annotations

import sys
import types

import launcher
from src.config import load_settings


def test_default_config_requires_explicit_review_confirmation() -> None:
    settings = load_settings("config/default.yaml")

    assert settings.calibrated is False


def test_launcher_dispatches_review_command(monkeypatch) -> None:
    module = types.ModuleType("src.calibration_review.review")

    def fake_main(argv: list[str]) -> int:
        assert argv == ["--config", "config/default.yaml"]
        return 17

    module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.calibration_review.review", module)

    assert launcher._dispatch(["bot_game_observer.exe", "review", "--config", "config/default.yaml"]) == 17
