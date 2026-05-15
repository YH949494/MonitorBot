from src.multi_slot import MultiSlotEngine, calculate_game_metrics, segment_metrics, render_game_report


def _engine():
    return MultiSlotEngine("config/slot_profiles")


def _grid():
    return [["A","K","Wild"],["Scatter","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["Scatter","A","J"]]


def test_two_games_do_not_mix_metrics():
    e = _engine()
    e.ingest_spin({"game_id":"egypt_demo","session_id":"s1","bet_amount":1.0,"payout_amount":1.0,"grid":_grid()})
    e.ingest_spin({"game_id":"candy_demo","session_id":"s2","bet_amount":1.0,"payout_amount":0.0,"grid":_grid()})
    assert len(e.store.game_spins("egypt_demo")) == 1
    assert len(e.store.game_spins("candy_demo")) == 1


def test_same_symbol_can_have_different_type_per_game():
    e = _engine()
    s1 = e.ingest_spin({"game_id":"egypt_demo","session_id":"s1","bet_amount":1.0,"payout_amount":1.0,"grid":_grid()})
    s2 = e.ingest_spin({"game_id":"candy_demo","session_id":"s2","bet_amount":1.0,"payout_amount":1.0,"grid":_grid()})
    assert s1["bonus_count"] != s2["bonus_count"]


def test_classification_and_grid_validation_and_unknown_symbol_preserved():
    e = _engine()
    ev = e.ingest_spin({"game_id":"egypt_demo","session_id":"s1","bet_amount":2.0,"payout_amount":1.0,"grid":[["Unknown","K","Wild"],["Scatter","Q","A"],["J","Bonus","Wild"],["A","K","Q"],["Scatter","A","J"]]})
    assert ev["dead_spin"] is False
    assert ev["visual_win"] is True
    assert ev["real_win"] is False
    assert "Unknown" in ev["detected_symbols"]


def test_metrics_rules_and_report_fields_and_segments():
    e = _engine()
    for p in [0.0, 1.0, 5.0, 0.5, 0.0]:
        e.ingest_spin({"game_id":"egypt_demo","session_id":"s1","bet_amount":1.0,"payout_amount":p,"grid":_grid()})
    spins = e.store.game_spins("egypt_demo")
    m = calculate_game_metrics("egypt_demo", spins, e.profiles["egypt_demo"])
    assert m["hit_frequency"] == 3/5
    assert m["bonus_frequency"] == 1.0
    assert m["max_multiplier_seen"] == 5.0
    assert m["longest_dry_streak"] >= 1
    assert m["sample_size_confidence"] == "low"
    rep = render_game_report(e.profiles["egypt_demo"], m)
    assert "Game:" in rep and "Hit frequency:" in rep and "Volatility label:" in rep
    seg = segment_metrics(spins, m, bet=1.0)
    assert seg["confidence_label"] == "low"
    assert "hit_frequency_diff_pct" in seg
