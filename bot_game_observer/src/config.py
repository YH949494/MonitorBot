"""YAML/JSON configuration loading and validation (portable paths)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .app_paths import SETTINGS_JSON, get_app_root
from .models import BotSettings


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_settings(path: Path | str) -> BotSettings:
    """Load and validate ``BotSettings`` from a YAML file."""
    raw = load_yaml(path)
    try:
        return BotSettings.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(
            f"Invalid configuration in {path}:\n{e}"
        ) from e


def load_settings_json(path: Path | str) -> BotSettings:
    """Load ``BotSettings`` from JSON (portable ``config/settings.json``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    try:
        return BotSettings.model_validate_json(p.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise RuntimeError(
            f"Invalid configuration in {path}:\n{e}"
        ) from e


def resolve_config_path(config_arg: str | None) -> Path:
    """Resolve CLI config path relative to app root when not absolute."""
    if not config_arg:
        return SETTINGS_JSON
    p = Path(config_arg)
    if p.is_absolute():
        return p
    return (get_app_root() / p).resolve()


def load_settings_auto(config_arg: str | None = None) -> BotSettings:
    """
    Load settings: explicit ``config_arg`` if provided, else ``config/settings.json``.

    Supports ``.json`` or ``.yaml``/``.yml`` by extension.
    """
    path = resolve_config_path(config_arg)
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {path}. Run the app once to create defaults, "
            "or pass --config path/to/default.yaml"
        )
    suf = path.suffix.lower()
    if suf == ".json":
        return load_settings_json(path)
    return load_settings(path)


def save_settings(settings: BotSettings, path: Path | str) -> None:
    """Serialize settings to YAML or JSON depending on file extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        return
    data = settings.model_dump(mode="json")
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

