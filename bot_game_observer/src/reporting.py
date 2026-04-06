"""Write JSONL logs, CSV summaries, and markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .app_paths import exports_path, logs_path
from .models import SessionSummary


def ensure_output_dirs(base_dir: str | Path | None = None) -> tuple[Path, Path]:
    """
    Ensure portable session log and export directories exist.

    ``base_dir`` is accepted for backward compatibility and ignored (paths are fixed).
    """
    _ = base_dir  # unused — portable layout is fixed under app root
    logs = logs_path("sessions")
    reports = exports_path("reports")
    logs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return logs, reports


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def write_csv_summary(path: Path, summary: SessionSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = summary.model_dump(mode="json")
    # Flatten lists as strings for CSV
    row["gaps_between_wins"] = ";".join(str(x) for x in summary.gaps_between_wins)
    row["confidence_notes"] = " | ".join(summary.confidence_notes)
    row["warnings"] = " | ".join(summary.warnings)
    pd.DataFrame([row]).to_csv(path, index=False)


def write_markdown_report(path: Path, summary: SessionSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Session report `{summary.session_id}`",
        "",
        "## Overview",
        "",
        f"- **Started:** {summary.started_at}",
        f"- **Ended:** {summary.ended_at}",
        f"- **Duration (sec):** {summary.duration_sec:.1f}",
        f"- **End reason:** {summary.end_reason}",
        "",
        "## Spin / outcome",
        "",
        f"- **Total spins:** {summary.total_spins}",
        f"- **Total wins (detected):** {summary.total_wins}",
        f"- **Total no-win (detected):** {summary.total_no_win}",
        f"- **First win at spin index:** {summary.first_win_spin_index}",
        f"- **Spins before first win:** {summary.spins_before_first_win}",
        f"- **Max consecutive no-win streak:** {summary.consecutive_no_win_streak_max}",
        "",
        "## Between wins",
        "",
        f"- **Gaps between wins:** {summary.gaps_between_wins}",
        f"- **Avg spins between wins:** {summary.avg_spins_between_wins}",
        f"- **Max spins between wins:** {summary.max_spins_between_wins}",
        "",
        "## Events / teases",
        "",
        f"- **Near-miss count:** {summary.near_miss_count}",
        f"- **Near-miss rate (approx):** {summary.near_miss_rate:.4f}",
        f"- **Bonus tease count:** {summary.bonus_tease_count}",
        f"- **Bonus trigger count:** {summary.bonus_trigger_count}",
        "",
        "## Detector confidence",
        "",
    ]
    if summary.confidence_notes:
        for n in summary.confidence_notes:
            lines.append(f"- {n}")
    else:
        lines.append("- No low-confidence transitions recorded.")
    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    if summary.warnings:
        for w in summary.warnings:
            lines.append(f"- **{w}**")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Data quality",
            "",
            "Near-miss and bonus-tease metrics are **game-specific** and depend on templates ",
            "and thresholds. Treat rates as indicative, not ground truth.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
