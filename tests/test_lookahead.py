"""The two gameweeks a forecast has: the one being forecast, and the one it is made in.

A backtest that conflates them leaks the future without ever reading an outcome. Valuing a
GW10 transfer over GW10-15 used GW15's own forecast — built from rates, form and a team model
that did not exist in October. Nothing realised was read, so every no-hindsight guard in
`test_season_sim` still passed, and the numbers were still not reproducible live.

`decision_state` and `lookahead` are the two halves of the fix: the first pins what is known
about a player at a deadline, the second carries that view forward across the horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.historical_forecast import decision_state, to_gameweek_level
from fpl_expert.backtest.season_sim import (
    ChipPlanner,
    _drop_components,
    _smooth_across_decisions,
    horizon_valuations,
    simulate_season,
)

RULES = {
    "squad": {
        "budget": 100.0, "size": 15, "starting_xi": 11, "max_per_club": 3,
        "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "formation": {
            "GK": {"min": 1, "max": 1}, "DEF": {"min": 3, "max": 5},
            "MID": {"min": 2, "max": 5}, "FWD": {"min": 1, "max": 3},
        },
    },
    "chips": {"bench_boost": {"count": 1, "windows": [[1, 8]]}},
}


# --- decision_state: what one deadline knows -------------------------------------------


def _double_gameweek_features():
    """One player, two fixtures in GW7. The second row's lags contain the first match."""
    return pd.DataFrame([
        {
            "season": "2025-26", "GW": 7, "element": 42, "name": "Doubler",
            "kickoff_time": "2025-10-04T14:00:00Z",
            "minutes_lag1": 0.0, "started_ewm": 0.10, "value": 70,
        },
        {
            "season": "2025-26", "GW": 7, "element": 42, "name": "Doubler",
            "kickoff_time": "2025-10-07T19:00:00Z",
            # Built by shift(1) over the row above, so it already knows he played 90.
            "minutes_lag1": 90.0, "started_ewm": 0.55, "value": 70,
        },
    ])


def test_a_double_gameweek_state_is_the_earlier_kickoff():
    """Before the deadline neither match has been played, so the later row's lags are the
    first match's result. 1,766 of 86,765 archive rows carried that, in exactly the weeks
    chips are timed around."""
    state = decision_state(_double_gameweek_features(), "2025-26", 7)

    assert len(state) == 1
    assert state["minutes_lag1"].iloc[0] == 0.0
    assert state["started_ewm"].iloc[0] == pytest.approx(0.10)


def test_state_does_not_reach_into_the_discarded_row_for_a_null():
    """`groupby().first()` takes the first NON-NULL value per column independently, which
    would fill a missing lag from the very row being discarded. `drop_duplicates` does not."""
    features = _double_gameweek_features()
    features.loc[0, "minutes_lag1"] = np.nan

    state = decision_state(features, "2025-26", 7)
    assert np.isnan(state["minutes_lag1"].iloc[0])


def test_state_is_taken_by_kickoff_not_by_row_order():
    features = _double_gameweek_features().iloc[::-1].reset_index(drop=True)
    assert decision_state(features, "2025-26", 7)["minutes_lag1"].iloc[0] == 0.0


def test_a_gameweek_with_no_players_is_an_error_not_an_empty_frame():
    with pytest.raises(ValueError, match="no players"):
        decision_state(_double_gameweek_features(), "2025-26", 9)


# --- to_gameweek_level: one target, several vantage points -----------------------------


def _fixture_rows(as_of, gw, points):
    return pd.DataFrame([{
        "season": "2025-26", "as_of_gw": as_of, "gw": gw, "player_id": 1,
        "expected_points": points, "position": "MID", "team": "AVL", "price": 7.0,
    }])


def test_the_same_target_week_survives_from_several_decision_weeks():
    """Collapsing on season and target alone would fold six forecasts of GW15 into one."""
    table = pd.concat([
        _fixture_rows(10, 15, 4.0), _fixture_rows(11, 15, 5.0), _fixture_rows(12, 15, 6.0)
    ], ignore_index=True)

    collapsed = to_gameweek_level(table)
    assert len(collapsed) == 3
    assert sorted(collapsed["as_of_gw"]) == [10, 11, 12]
    assert collapsed.set_index("as_of_gw").loc[12, "expected_points"] == pytest.approx(6.0)


def test_a_table_without_a_decision_week_still_collapses_by_target():
    table = pd.concat([
        _fixture_rows(10, 15, 4.0), _fixture_rows(10, 15, 2.0)
    ], ignore_index=True).drop(columns="as_of_gw")

    collapsed = to_gameweek_level(table)
    assert len(collapsed) == 1
    assert collapsed["expected_points"].iloc[0] == pytest.approx(6.0)


# --- horizon_valuations: which window a decision sees ----------------------------------


def _flat(spike_gw=3, spike=50.0):
    """Three gameweeks. Player 1 is ordinary until `spike_gw`, where he explodes."""
    rows = []
    for gw in (1, 2, 3):
        for pid in (1, 2):
            points = spike if (pid == 1 and gw == spike_gw) else 1.0
            rows.append({
                "gw": gw, "player_id": pid, "expected_points": points,
                "position": "MID", "team": "AVL", "price": 7.0,
            })
    return pd.DataFrame(rows)


def _blind_lookahead(flat):
    """GW1's view: the spike is invisible, because in GW1 nothing had happened to cause it."""
    week1 = flat[flat["gw"] == 1]
    return {
        1: {
            1: week1,
            2: week1.assign(gw=2),
            3: week1.assign(gw=3),
        }
    }


def test_the_lookahead_window_replaces_the_target_weeks_own_forecast():
    flat = _flat()
    leaky = horizon_valuations(flat, horizon=3, decay=1.0, captaincy_weight=0.0)
    honest = horizon_valuations(
        flat, horizon=3, decay=1.0, captaincy_weight=0.0, lookahead=_blind_lookahead(flat)
    )

    def value(table, pid):
        return float(table.set_index("player_id").loc[pid, "horizon_points"])

    assert value(leaky[1], 1) == pytest.approx(52.0)     # 1 + 1 + the GW3 spike
    assert value(honest[1], 1) == pytest.approx(3.0)     # GW1 could not see it
    assert value(honest[1], 2) == pytest.approx(3.0)


def test_decision_weeks_the_lookahead_does_not_cover_fall_back():
    """Checked per gameweek, not once: a season where only some weeks were rebuilt must
    still simulate rather than silently valuing the rest at zero."""
    flat = _flat()
    honest = horizon_valuations(
        flat, horizon=3, decay=1.0, captaincy_weight=0.0, lookahead=_blind_lookahead(flat)
    )
    leaky = horizon_valuations(flat, horizon=3, decay=1.0, captaincy_weight=0.0)

    assert honest[2].equals(leaky[2])


def test_an_empty_lookahead_behaves_exactly_like_none():
    flat = _flat()
    assert horizon_valuations(flat, horizon=3, lookahead={})[1].equals(
        horizon_valuations(flat, horizon=3)[1]
    )


# --- the simulator ---------------------------------------------------------------------


def _season(n_gws=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gws + 1):
        pid = 0
        for position, count in (("GK", 8), ("DEF", 16), ("MID", 16), ("FWD", 10)):
            for i in range(count):
                rows.append({
                    "gw": gw, "player_id": pid, "web_name": f"{position}{i}",
                    "position": position, "team": f"club{i % 8}",
                    "price": 4.0 + (i % 6) * 1.5 + rng.normal(0, 0.05),
                    "expected_points": 1.0 + (i % 6) * 0.7 + rng.normal(0, 0.05),
                    "actual_points": float(rng.poisson(2 + (i % 6))),
                    "pts_bonus": 0.3, "pts_defcon": 0.2,
                })
                pid += 1
    return pd.DataFrame(rows)


def _windowed(frame, span=4):
    """Every decision week seeing exactly what the flat table says — a null lookahead."""
    return {
        gw: {g: frame[frame["gw"] == g] for g in range(gw, gw + span)
             if g in set(frame["gw"])}
        for gw in sorted(frame["gw"].unique())
    }


def test_a_lookahead_that_repeats_the_flat_table_changes_nothing():
    """The identity case. If this moves, the plumbing is altering decisions on its own."""
    frame = _season()
    plain = simulate_season(frame, rules=RULES)
    echoed = simulate_season(frame, rules=RULES, lookahead=_windowed(frame))
    assert echoed.total_points == pytest.approx(plain.total_points)


def test_a_squad_is_not_built_around_a_forecast_the_decision_could_not_have_seen():
    """The bug itself, end to end.

    A cheap midfielder becomes enormous in GW3. The leaky simulator buys him in GW1 —
    benched and pointless for two weeks — because GW1's horizon is assembled from GW3's own
    forecast, which in October did not exist. Given only GW1's view of GW3 he is worthless
    and stays unowned until the week he is not.
    """
    frame = _season()
    target = 39                          # a cheap midfielder nobody would otherwise buy
    frame.loc[frame["player_id"] == target, "price"] = 4.5
    frame.loc[frame["player_id"] == target, "expected_points"] = 0.1
    late = (frame["player_id"] == target) & (frame["gw"] >= 3)
    frame.loc[late, "expected_points"] = 40.0

    blind = _windowed(frame)
    for as_of, window in blind.items():
        for gw in window:
            if gw != as_of:
                # Every forward week seen as the decision week sees it: still worthless.
                block = window[gw].copy()
                block.loc[block["player_id"] == target, "expected_points"] = 0.1
                window[gw] = block

    def owned_before_the_spike(result):
        return {week["gw"] for week in result.trace
                if week["gw"] < 3 and target in week["squad_ids"]}

    assert owned_before_the_spike(simulate_season(frame, rules=RULES)) == {1, 2}
    assert owned_before_the_spike(
        simulate_season(frame, rules=RULES, lookahead=blind)
    ) == set()


# --- the chip planner ------------------------------------------------------------------


def test_every_forward_chip_valuation_is_anchored_at_the_decision_week():
    """Holding a chip is optimal stopping, and one solved with next month's forecasts would
    stop far too well. Each `frame_for` call must carry the week the decision is made in."""
    frame = _season(n_gws=6)
    calls = []

    def frame_for(gw, as_of=None):
        calls.append((gw, as_of))
        return frame[frame["gw"] == gw]

    planner = ChipPlanner(RULES, [1, 2, 3, 4, 5, 6], horizon=3, allowed=("bench_boost",))
    squad = set(frame[frame["gw"] == 2].nlargest(15, "expected_points")["player_id"])
    planner.decide(2, squad, 0.0, frame_for, RULES, "expected_points")

    assert calls, "the planner never valued the chip"
    assert {as_of for _, as_of in calls} == {2}
    assert max(gw for gw, _ in calls) > 2, "the planner never looked forward"


# --- ablation --------------------------------------------------------------------------


def test_ablating_a_component_removes_it_from_the_horizon_too():
    """Zeroing a component in the week in front but leaving it in the forward valuation
    would measure roughly a sixth of its true decision weight."""
    frame = _season(n_gws=2)
    trimmed = _drop_components(_windowed(frame, span=2), ["pts_bonus"])

    before = _windowed(frame, span=2)[1][2]["expected_points"].sum()
    after = trimmed[1][2]["expected_points"].sum()
    assert after == pytest.approx(before - 0.3 * len(trimmed[1][2]))


def test_dropping_components_from_no_lookahead_stays_none():
    assert _drop_components(None, ["pts_bonus"]) is None


# --- smoothing the valuation across decision weeks -------------------------------------


def _valuations(values):
    """One player, `values[gw]` as his forward valuation at each decision week."""
    return {
        gw: pd.DataFrame({"player_id": [1], "horizon_points": [value]})
        for gw, value in values.items()
    }


def _series(valuations):
    return [float(valuations[gw]["horizon_points"].iloc[0]) for gw in sorted(valuations)]


def test_smoothing_is_causal():
    """The whole point. An exponential mean at GW4 built from GW1-4 is as available live as
    the raw value; one that peeked at GW5 would be the very bug this module exists to fix."""
    early = _smooth_across_decisions(_valuations({1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}), 2.0)
    late = _smooth_across_decisions(
        _valuations({1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 99.0}), 2.0
    )
    assert _series(early) == pytest.approx(_series(late)[:4])


def test_smoothing_damps_a_revision_without_ignoring_it():
    smoothed = _smooth_across_decisions(_valuations({1: 0.0, 2: 0.0, 3: 12.0}), 1.0)
    values = _series(smoothed)
    assert values[0] == pytest.approx(0.0)          # nothing to average with yet
    assert 0.0 < values[2] < 12.0                   # moved toward it, not all the way


def test_smoothing_leaves_a_constant_valuation_alone():
    smoothed = _smooth_across_decisions(_valuations({1: 4.0, 2: 4.0, 3: 4.0}), 2.0)
    assert _series(smoothed) == pytest.approx([4.0, 4.0, 4.0])


def test_players_are_smoothed_independently():
    valuations = {
        1: pd.DataFrame({"player_id": [1, 2], "horizon_points": [0.0, 10.0]}),
        2: pd.DataFrame({"player_id": [1, 2], "horizon_points": [0.0, 10.0]}),
    }
    smoothed = _smooth_across_decisions(valuations, 1.0)
    assert smoothed[2].set_index("player_id")["horizon_points"].to_dict() == pytest.approx(
        {1: 0.0, 2: 10.0}
    )


def test_a_zero_halflife_is_simply_off():
    flat = _flat()
    assert horizon_valuations(flat, horizon=3, smoothing=0.0)[1].equals(
        horizon_valuations(flat, horizon=3)[1]
    )


# --- decision-level evidence: the trace records WHY, not just what --------------------


def test_every_transfer_records_the_margin_that_justified_it():
    """Season totals cannot separate a policy that traded well from one that traded often and
    got lucky. There are ~190 transfer decisions in three seasons against three season totals,
    and the forecast margin is what makes each one gradable."""
    result = simulate_season(_season(n_gws=4), rules=RULES)

    made = [week["moves"] for week in result.trace if week.get("moves")]
    assert made, "no transfer was recorded"
    for move in made:
        assert move["in"] and move["out"]
        assert len(move["in"]) == len(move["out"])
        assert isinstance(move["forecast_gain"], float)
        assert move["hits"] >= 0


def test_the_opening_gameweek_has_no_transfer_to_record():
    result = simulate_season(_season(n_gws=3), rules=RULES)
    assert result.trace[0]["moves"] is None


def test_a_higher_hit_bar_trades_less():
    """`hit_bar` is the margin demanded per hit INSIDE the optimiser, which is only the same
    as the -4 actually deducted if the forecast margin is unbiased. It is not: realised return
    regresses on forecast gain with slope 0.436."""
    frame = _season(n_gws=6)
    greedy = simulate_season(frame, rules=RULES, hit_bar=0.5)
    strict = simulate_season(frame, rules=RULES, hit_bar=50.0)

    assert greedy.gameweeks["transfers_made"].sum() > strict.gameweeks["transfers_made"].sum()
    assert strict.gameweeks["hit_cost"].sum() == 0


def test_the_hit_bar_does_not_change_what_a_hit_actually_costs():
    """The bar is a belief about the forecast; the -4 is a rule of the game. Raising the bar
    must make hits rarer, never cheaper."""
    frame = _season(n_gws=6)
    result = simulate_season(frame, rules=RULES, hit_bar=1.0)
    weeks = result.gameweeks[result.gameweeks["hit_cost"] > 0]
    assert (weeks["hit_cost"] % 4 == 0).all()
