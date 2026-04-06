#!/usr/bin/env python3
"""Rebuild CSV/Markdown summary from an existing JSONL session log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from src.app_paths import resolve_path_relative_to_app
from src.bootstrap import init_portable_app
from src.metrics import build_summary
from src.reporting import ensure_output_dirs, write_csv_summary, write_markdown_report

console = Console()


def load_jsonl(path: Path) -> tuple[str, list[dict]]:
    events: list[dict] = []
    session_id = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            events.append(rec)
            if not session_id:
                session_id = str(rec.get("session_id", ""))
    return session_id, events


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a session JSONL log.")
    parser.add_argument("jsonl", type=str, help="Path to session_*.jsonl")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Override export directory (default: exports/reports under app root)",
    )
    args = parser.parse_args()

    init_portable_app(create_config=False)

    p = resolve_path_relative_to_app(args.jsonl)
    if not p.is_file():
        console.print(f"[red]File not found: {p}[/red]")
        return 2

    session_id, events = load_jsonl(p)
    if not session_id:
        session_id = p.stem.replace("session_", "")
    summary = build_summary(session_id, events)

    if args.out_dir:
        base = resolve_path_relative_to_app(args.out_dir)
        reports = base / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        csv_path = reports / f"session_{session_id}.csv"
        md_path = reports / f"session_{session_id}.md"
    else:
        _, reports_dir = ensure_output_dirs()
        csv_path = reports_dir / f"session_{session_id}.csv"
        md_path = reports_dir / f"session_{session_id}.md"

    write_csv_summary(csv_path, summary)
    write_markdown_report(md_path, summary)
    console.print(f"Wrote [green]{csv_path}[/green] and [green]{md_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
