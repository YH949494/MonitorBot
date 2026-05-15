from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class GameProfile:
    game_id: str
    game_name: str
    provider: str
    reel_count: int
    row_count: int
    layout_type: str
    paylines_or_ways: str
    symbol_mappings: dict[str, str]
    wild_symbols: list[str]
    scatter_symbols: list[str]
    bonus_symbols: list[str]
    bonus_trigger_rule: dict[str, Any]
    created_at: str
    updated_at: str


class MultiSlotStore:
    def __init__(self) -> None:
        self.spins: dict[str, list[dict[str, Any]]] = {}

    def add_spin(self, spin: dict[str, Any]) -> None:
        self.spins.setdefault(spin["game_id"], []).append(spin)

    def game_spins(self, game_id: str) -> list[dict[str, Any]]:
        return list(self.spins.get(game_id, []))


class MultiSlotEngine:
    def __init__(self, profiles_dir: str | Path, store: MultiSlotStore | None = None) -> None:
        self.profiles = self._load_profiles(Path(profiles_dir))
        self.store = store or MultiSlotStore()

    def _load_profiles(self, profiles_dir: Path) -> dict[str, GameProfile]:
        profiles: dict[str, GameProfile] = {}
        for p in profiles_dir.glob("*.json"):
            data = json.loads(p.read_text(encoding="utf-8"))
            profiles[data["game_id"]] = GameProfile(**data)
        return profiles

    def ingest_spin(self, payload: dict[str, Any]) -> dict[str, Any]:
        game_id = payload["game_id"]
        profile = self.profiles[game_id]
        grid = payload["grid"]
        if len(grid) != profile.reel_count:
            raise ValueError("grid reel count mismatch")
        if any(len(col) != profile.row_count for col in grid):
            raise ValueError("grid row count mismatch")

        normalized = [[profile.symbol_mappings.get(sym, sym) for sym in col] for col in grid]
        flat = [s for col in normalized for s in col]
        wild_count = sum(1 for s in flat if s in profile.wild_symbols)
        scatter_count = sum(1 for s in flat if s in profile.scatter_symbols)
        bonus_count = sum(1 for s in flat if s in profile.bonus_symbols)
        bet = float(payload["bet_amount"])
        payout = float(payload["payout_amount"])
        bonus_triggered = scatter_count >= int(profile.bonus_trigger_rule.get("scatter_count", 3))

        event = {
            "game_id": game_id,
            "session_id": payload["session_id"],
            "spin_id": payload.get("spin_id", str(uuid4())),
            "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "bet_amount": bet,
            "payout_amount": payout,
            "payout_multiplier": (payout / bet) if bet > 0 else 0.0,
            "grid": normalized,
            "detected_symbols": flat,
            "wild_count": wild_count,
            "scatter_count": scatter_count,
            "bonus_count": bonus_count,
            "win_detected": payout > 0,
            "dead_spin": payout == 0,
            "visual_win": payout > 0 and payout < bet,
            "real_win": payout > bet,
            "bonus_triggered": bonus_triggered,
            "free_spin_mode": bool(payload.get("free_spin_mode", False)),
            "confidence": float(payload.get("confidence", 0.8)),
        }
        self.store.add_spin(event)
        return event


def _freq(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _segment_confidence(sample: int) -> str:
    if sample < 20:
        return "low"
    if sample < 100:
        return "medium"
    return "high"


def calculate_game_metrics(game_id: str, spins: list[dict[str, Any]], profile: GameProfile) -> dict[str, Any]:
    total = len(spins)
    symbol_counts: dict[int, dict[str, int]] = {i: {} for i in range(profile.reel_count)}
    wild_freq: dict[int, float] = {}
    scatter_freq: dict[int, float] = {}
    bonus_freq_reel: dict[int, float] = {}
    for sp in spins:
        for i, col in enumerate(sp["grid"]):
            for sym in col:
                symbol_counts[i][sym] = symbol_counts[i].get(sym, 0) + 1

    for i in range(profile.reel_count):
        reel_total = sum(symbol_counts[i].values())
        wild_freq[i] = _freq(sum(symbol_counts[i].get(s, 0) for s in profile.wild_symbols), reel_total)
        scatter_freq[i] = _freq(sum(symbol_counts[i].get(s, 0) for s in profile.scatter_symbols), reel_total)
        bonus_freq_reel[i] = _freq(sum(symbol_counts[i].get(s, 0) for s in profile.bonus_symbols), reel_total)

    payouts = [s["payout_amount"] for s in spins]
    multipliers = [s["payout_multiplier"] for s in spins]
    bonus_indices = [idx for idx, s in enumerate(spins, start=1) if s["bonus_triggered"]]
    dry = 0
    max_dry = 0
    for s in spins:
        if s["payout_amount"] > 0:
            dry = 0
        else:
            dry += 1
            max_dry = max(max_dry, dry)

    return {
        "game_id": game_id,
        "total_spins": total,
        "estimated_symbol_weight": symbol_counts,
        "estimated_reel_strip_distribution": symbol_counts,
        "paytable_evidence": {"observed_payout_multipliers": sorted(set(multipliers))[:20]},
        "wild_frequency_by_reel": wild_freq,
        "scatter_frequency_by_reel": scatter_freq,
        "bonus_frequency_by_reel": bonus_freq_reel,
        "hit_frequency": _freq(sum(1 for s in spins if s["payout_amount"] > 0), total),
        "bonus_frequency": _freq(sum(1 for s in spins if s["bonus_triggered"]), total),
        "dead_spin_rate": _freq(sum(1 for s in spins if s["dead_spin"]), total),
        "visual_win_rate": _freq(sum(1 for s in spins if s["visual_win"]), total),
        "real_win_rate": _freq(sum(1 for s in spins if s["real_win"]), total),
        "average_spins_between_bonus": (bonus_indices[-1] - bonus_indices[0]) / max(1, (len(bonus_indices)-1)) if len(bonus_indices) > 1 else None,
        "max_win_seen": max(payouts) if payouts else 0.0,
        "observed_max_exposure": max(payouts) if payouts else 0.0,
        "max_multiplier_seen": max(multipliers) if multipliers else 0.0,
        "longest_dry_streak": max_dry,
        "inferred_volatility": "high" if (max(multipliers) if multipliers else 0) >= 10 else "medium" if (max(multipliers) if multipliers else 0) >= 3 else "low",
        "sample_size_confidence": _segment_confidence(total),
    }


def segment_metrics(spins: list[dict[str, Any]], baseline: dict[str, Any], *, window: tuple[int, int] | None = None, bet: float | None = None) -> dict[str, Any]:
    subset = spins
    if window is not None:
        start, end = window
        subset = spins[start - 1:end]
    if bet is not None:
        subset = [s for s in subset if s["bet_amount"] == bet]
    total = len(subset)
    hit = _freq(sum(1 for s in subset if s["payout_amount"] > 0), total)
    bonus = _freq(sum(1 for s in subset if s["bonus_triggered"]), total)
    dead = _freq(sum(1 for s in subset if s["dead_spin"]), total)
    return {
        "sample_size": total,
        "confidence_label": _segment_confidence(total),
        "warning": "sample_too_small" if total < 20 else None,
        "hit_frequency": hit,
        "bonus_frequency": bonus,
        "dead_spin_rate": dead,
        "hit_frequency_diff_abs": hit - baseline["hit_frequency"],
        "hit_frequency_diff_pct": ((hit - baseline["hit_frequency"]) / baseline["hit_frequency"]) if baseline["hit_frequency"] else None,
    }


def render_game_report(profile: GameProfile, metrics: dict[str, Any], segments: dict[str, Any] | None = None) -> str:
    return (
        f"Game: {profile.game_name}\nProvider: {profile.provider}\n"
        f"Total sample spins: {metrics['total_spins']}\nConfidence: {metrics['sample_size_confidence']}\n"
        f"Reel layout: {profile.reel_count}x{profile.row_count}\n"
        f"Wild behavior: {profile.wild_symbols}\nScatter behavior: {profile.scatter_symbols}\nBonus behavior: {profile.bonus_symbols}\n"
        f"Symbol weight by reel: {metrics['estimated_symbol_weight']}\n"
        f"Hit frequency: {metrics['hit_frequency']:.4f}\nBonus frequency: {metrics['bonus_frequency']:.4f}\n"
        f"Dead spin rate: {metrics['dead_spin_rate']:.4f}\nVisual win rate: {metrics['visual_win_rate']:.4f}\n"
        f"Real win rate: {metrics['real_win_rate']:.4f}\n"
        f"Average spins between bonus: {metrics['average_spins_between_bonus']}\n"
        f"Max win observed: {metrics['max_win_seen']}\nMax multiplier observed: {metrics['max_multiplier_seen']}\n"
        f"Longest dry streak: {metrics['longest_dry_streak']}\nVolatility label: {metrics['inferred_volatility']}\n"
        "Design read summary: observed difference only; possible drift needs more spins.\n"
        + (f"Metric Drift Analysis: {segments}\n" if segments else "")
    )
