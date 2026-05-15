from src.multi_slot import MultiSlotEngine, calculate_game_metrics, render_game_report, segment_metrics


def main() -> None:
    engine = MultiSlotEngine("config/slot_profiles")
    samples = [
        {"game_id":"egypt_demo","session_id":"s1","bet_amount":1.0,"payout_amount":0.5,"grid":[["A","K","Wild"],["Scatter","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["Scatter","A","J"]]},
        {"game_id":"egypt_demo","session_id":"s1","bet_amount":1.0,"payout_amount":0.0,"grid":[["A","K","Wild"],["Q","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["A","A","J"]]},
        {"game_id":"candy_demo","session_id":"s2","bet_amount":5.0,"payout_amount":0.0,"grid":[["A","K","Wild"],["Scatter","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["Scatter","A","J"]]},
        {"game_id":"candy_demo","session_id":"s2","bet_amount":1.0,"payout_amount":4.0,"grid":[["A","K","Wild"],["Scatter","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["Scatter","A","J"]]},
    ]
    for s in samples:
        engine.ingest_spin(s)

    for gid, profile in engine.profiles.items():
        spins = engine.store.game_spins(gid)
        metrics = calculate_game_metrics(gid, spins, profile)
        seg = {"spins_1_100": segment_metrics(spins, metrics, window=(1, 100))}
        print(render_game_report(profile, metrics, seg))


if __name__ == "__main__":
    main()
