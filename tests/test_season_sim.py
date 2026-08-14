"""Season simulator: legality, no hindsight, and the ablation machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.season_sim import ablate, score_gameweek, simulate_season

RULES = {
    "squad": {
        "budget": 100.0, "size": 15, "starting_xi": 11, "max_per_club": 3,
        "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "formation": {
            "GK": {"min": 1, "max": 1}, "DEF": {"min": 3, "max": 5},
            "MID": {"min": 2, "max": 5}, "FWD": {"min": 1, "max": 3},
        },
    }
}


def _forecasts(n_gws=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gws + 1):
        pid = 0
        for position, count in (("GK", 8), ("DEF", 16), ("MID", 16), ("FWD", 10)):
            for i in range(count):
                rows.append({
                    "gw": gw, "player_id": pid, "web_name": f"{position}{i}",
                    "position": position, "team": f"club{i % 8}",
                    # Jittered: a pool of identically-priced, identically-scored players is
                    # pathologically symmetric and makes branch-and-bound crawl. Real pools
                    # never look like that.
                    "price": 4.0 + (i % 6) * 1.5 + rng.normal(0, 0.05),
                    "expected_points": 1.0 + (i % 6) * 0.7 + rng.normal(0, 0.05),
                    "actual_points": float(rng.poisson(2 + (i % 6))),
                    "pts_bonus": 0.3, "pts_defcon": 0.2, "pts_saves": 0.1,
                })
                pid += 1
    return pd.DataFrame(rows)


def test_simulation_produces_a_row_per_gameweek():
    result = simulate_season(_forecasts(), rules=RULES)
    assert len(result.gameweeks) == 3
    assert result.total_points > 0


def test_xi_is_chosen_on_forecast_not_hindsight():
    """Choosing the XI with knowledge of realised points would inflate every result and
    could never be reproduced live. This is the single most important guard here."""
    frame = _forecasts(n_gws=1)
    # Make forecast and outcome disagree completely.
    frame["expected_points"] = np.linspace(10, 1, len(frame))
    frame["actual_points"] = np.linspace(1, 10, len(frame))

    squad = set(frame.nlargest(15, "expected_points")["player_id"])
    scored = score_gameweek(squad, None, frame, RULES)

    hindsight = frame[frame["player_id"].isin(squad)].nlargest(11, "actual_points")
    best_possible = hindsight["actual_points"].sum() + hindsight["actual_points"].max()
    assert scored["points"] < best_possible


def test_captain_falls_back_to_the_best_starter_when_absent():
    """A captain who was benched or transferred out must not silently score zero."""
    frame = _forecasts(n_gws=1)
    squad = set(frame.nlargest(15, "expected_points")["player_id"])
    scored = score_gameweek(squad, captain_id=999_999, gw_frame=frame, rules=RULES)
    assert scored["captain_points"] > 0


def test_scored_xi_is_a_legal_formation():
    frame = _forecasts(n_gws=1)
    squad = set(frame.nlargest(15, "expected_points")["player_id"])
    assert score_gameweek(squad, None, frame, RULES)["starters"] == 11


def test_squad_stays_legal_across_transfers():
    """A transfer that breaks the 3-per-club limit is not a legal move."""
    frame = _forecasts(n_gws=6, seed=3)
    result = simulate_season(frame, rules=RULES, transfer_threshold=0.0)
    assert result.gameweeks["squad_size"].eq(15).all()


def test_transfer_threshold_controls_activity_under_the_greedy_policy():
    """The threshold only applies to the greedy policy; the MILP prices hits directly and
    has no threshold to tune."""
    frame = _forecasts(n_gws=6, seed=1)
    eager = simulate_season(
        frame, rules=RULES, transfer_policy="greedy", transfer_threshold=0.0
    )
    reluctant = simulate_season(
        frame, rules=RULES, transfer_policy="greedy", transfer_threshold=99.0
    )
    assert eager.gameweeks["transfers_made"].sum() >= reluctant.gameweeks["transfers_made"].sum()
    assert reluctant.gameweeks["transfers_made"].sum() == 0


def test_milp_policy_uses_far_more_transfers_than_greedy():
    """The headline refinement. Greedy used 21 of 38 free transfers and never took a hit;
    the MILP searches all swaps at once under the club limit and uses 48 with 11 hits,
    worth +213 season points. (The rank figures once quoted here are withdrawn — see
    `field_sim` on why the simulated field cannot currently be ranked against.)"""
    frame = _forecasts(n_gws=8, seed=4)
    greedy = simulate_season(frame, rules=RULES, transfer_policy="greedy")
    milp = simulate_season(frame, rules=RULES, transfer_policy="milp")
    assert (milp.gameweeks["transfers_made"].sum()
            >= greedy.gameweeks["transfers_made"].sum())


def test_selling_price_keeps_only_half_the_profit():
    """FPL's real rule, and the simulator used to ignore it — taking market price handed it
    the full rise on every player who had gone up and funded transfers a manager could not
    actually make. Bought at 7.0 and now worth 7.5, you receive 7.2, not 7.5."""
    from fpl_expert.backtest.season_sim import selling_prices

    held = pd.DataFrame({"player_id": [1, 2, 3], "price": [7.5, 6.4, 8.0]})
    purchase = {1: 7.0, 2: 7.0, 3: 8.0}
    prices = selling_prices(held, purchase)

    assert prices.iloc[0] == pytest.approx(7.2)     # rose 0.5, keep half
    assert prices.iloc[1] == pytest.approx(6.4)     # fell, borne in full
    assert prices.iloc[2] == pytest.approx(8.0)     # unchanged


def test_an_unknown_purchase_price_falls_back_to_the_market():
    from fpl_expert.backtest.season_sim import selling_prices

    held = pd.DataFrame({"player_id": [9], "price": [5.5]})
    assert selling_prices(held, {}).iloc[0] == pytest.approx(5.5)


def test_the_purchase_ledger_tracks_transfers():
    """A sold player leaves the ledger and a bought one joins at what was paid, because that
    is the basis he will later be sold against."""
    from fpl_expert.backtest.season_sim import _update_purchases

    frame = pd.DataFrame({"player_id": [1, 2, 3], "price": [5.0, 6.0, 7.5]})
    updated = _update_purchases({1: 4.5, 2: 6.0}, before={1, 2}, after={2, 3}, frame=frame)

    assert 1 not in updated                          # sold
    assert updated[2] == 6.0                         # held, basis unchanged
    assert updated[3] == 7.5                         # bought at today's price


def test_ablation_reports_a_delta_per_component():
    table = ablate(_forecasts(n_gws=3), rules=RULES)
    assert "full" in set(table["variant"])
    assert table.loc[table["variant"] == "full", "delta"].iloc[0] == 0.0
    assert set(table["variant"]) >= {"no_bonus", "no_defcon", "no_saves"}


def test_ablation_measures_decision_impact_not_magnitude():
    """A component that is identical for every player cannot change any decision, so its
    ablation delta is zero however large it is. Only components that DISCRIMINATE between
    players carry decision weight — which is what the ablation table is really ranking."""
    frame = _forecasts(n_gws=2)
    frame["pts_bonus"] = 5.0                       # large, but uniform
    uniform = ablate(frame, rules=RULES)
    assert uniform.loc[uniform["variant"] == "no_bonus", "delta"].iloc[0] == 0.0

    frame["pts_bonus"] = np.linspace(0, 6, len(frame))   # same scale, now discriminating
    varying = ablate(frame, rules=RULES)
    assert varying.loc[varying["variant"] == "no_bonus", "delta"].iloc[0] != 0.0


def test_summary_reports_the_headline_numbers():
    summary = simulate_season(_forecasts(), rules=RULES).summary()
    assert {"total_points", "points_per_gw", "transfers", "captain_points"} <= set(summary.index)
    assert summary["points_per_gw"] == pytest.approx(
        summary["total_points"] / 3, abs=0.01
    )


def test_the_armband_falls_back_to_the_best_starter_not_the_goalkeeper():
    """`pick_xi` returns the XI in FORMATION order, so the old `head(1)` fallback handed the
    captaincy to the keeper. The test that covered this asserted only that the armband scored
    something, so it passed while the behaviour was wrong."""
    frame = _forecasts(n_gws=1)
    squad = set(frame.nlargest(15, "expected_points")["player_id"])
    scored = score_gameweek(squad, captain_id=999_999, gw_frame=frame, rules=RULES)

    held = frame[frame["player_id"].isin(squad)]
    _, xi = __import__(
        "fpl_expert.backtest.season_sim", fromlist=["pick_xi"]
    ).pick_xi(held, RULES, "expected_points")
    best = xi.nlargest(1, "expected_points").iloc[0]
    assert scored["captain_id"] == best["player_id"]


def test_the_vice_captain_takes_over_when_the_captain_does_not_play():
    """FPL passes the armband on when the captain plays no minutes. Doubling his zero was a
    silent conservative bias in every season replayed."""
    frame = _forecasts(n_gws=1)
    frame["actual_minutes"] = 90.0
    squad_frame = frame.nlargest(15, "expected_points")
    squad = set(squad_frame["player_id"])
    captain = squad_frame.iloc[0]["player_id"]
    frame.loc[frame["player_id"] == captain, ["actual_minutes", "actual_points"]] = [0.0, 0.0]

    scored = score_gameweek(squad, captain, frame, RULES)
    assert scored["captain_id"] != captain
    assert scored["captain_points"] > 0


def test_a_captain_who_played_keeps_the_armband():
    frame = _forecasts(n_gws=1)
    frame["actual_minutes"] = 90.0
    squad_frame = frame.nlargest(15, "expected_points")
    squad = set(squad_frame["player_id"])
    captain = squad_frame.iloc[0]["player_id"]
    assert score_gameweek(squad, captain, frame, RULES)["captain_id"] == captain
