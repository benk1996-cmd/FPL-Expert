"""Captaincy as a durable premium, not a weekly afterthought."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.optimise.captaincy import (
    captain_score,
    captaincy_probabilities,
    captaincy_reliability,
    captaincy_uplift,
)
from fpl_expert.optimise.transfers import horizon_points


def _gw(points, ids=None):
    return pd.DataFrame({
        "player_id": ids or list(range(len(points))),
        "expected_points": points,
        "points_variance": [4.0] * len(points),
    })


def test_captaincy_probabilities_sum_to_one():
    probabilities = captaincy_probabilities(_gw([8.0, 6.0, 5.0, 2.0]))
    assert probabilities.sum() == pytest.approx(1.0)


def test_the_best_forecast_is_the_most_likely_captain():
    probabilities = captaincy_probabilities(_gw([9.0, 6.0, 5.0]))
    assert probabilities.idxmax() == 0


def test_credit_is_shared_between_near_equal_candidates():
    """Winner-take-all gives a marginally-best player everything and his near-equal rival
    nothing. Forecasts are not that precise."""
    close = captaincy_probabilities(_gw([6.0, 5.9, 3.0]))
    assert close.iloc[1] > 0.3          # the rival keeps real credit


def test_temperature_controls_faith_in_small_gaps():
    confident = captaincy_probabilities(_gw([6.0, 5.5, 3.0]), temperature=0.1)
    hedged = captaincy_probabilities(_gw([6.0, 5.5, 3.0]), temperature=5.0)
    assert confident.iloc[0] > hedged.iloc[0]


def test_only_plausible_candidates_compete():
    """A fringe player is never captained; including him would drag the softmax toward
    uniform and dilute everyone."""
    probabilities = captaincy_probabilities(_gw([9.0, 8.0] + [0.1] * 50), pool=5)
    assert len(probabilities) == 5


# --- the durable premium --------------------------------------------------


def test_a_recurring_captain_accrues_more_uplift_than_a_one_week_wonder():
    """The point of the whole module. Two players with identical horizon totals are NOT
    worth the same if one wears the armband every week."""
    steady = {gw: _gw([7.0, 6.9, 1.0], ids=[1, 2, 3]) for gw in range(1, 5)}
    uplift = captaincy_uplift(steady, decay=1.0).set_index("player_id")
    assert uplift.loc[1, "captaincy_uplift"] > uplift.loc[3, "captaincy_uplift"] * 10


def test_uplift_is_discounted_like_everything_else():
    frames = {1: _gw([10.0, 1.0]), 2: _gw([10.0, 1.0])}
    full = captaincy_uplift(frames, decay=1.0).set_index("player_id").loc[0, "captaincy_uplift"]
    halved = captaincy_uplift(frames, decay=0.5).set_index("player_id").loc[0, "captaincy_uplift"]
    assert halved < full
    assert halved == pytest.approx(full * 0.75, rel=0.05)   # 1 + 0.5 out of 1 + 1


def test_reliability_reports_how_often_you_would_captain_him():
    frames = {gw: _gw([9.0, 3.0], ids=[1, 2]) for gw in range(1, 5)}
    table = captaincy_reliability(frames).set_index("player_id")
    assert table.loc[1, "captain_share"] > 0.9
    assert table.loc[2, "captain_share"] < 0.1


# --- integration with transfer valuation ----------------------------------


def test_horizon_points_prices_the_captaincy_premium():
    """Without this, a transfer valuation rates a five-week captain the same as a player who
    would never wear the armband."""
    frames = {gw: _gw([8.0, 4.0], ids=[1, 2]) for gw in range(1, 4)}

    without = horizon_points(frames, decay=1.0, captaincy_weight=0.0).set_index("player_id")
    with_captaincy = horizon_points(frames, decay=1.0).set_index("player_id")

    # The raw totals are unchanged; only the premium is added.
    assert with_captaincy.loc[1, "horizon_points"] > without.loc[1, "horizon_points"]
    gain_captain = (with_captaincy.loc[1, "horizon_points"]
                    - without.loc[1, "horizon_points"])
    gain_other = (with_captaincy.loc[2, "horizon_points"]
                  - without.loc[2, "horizon_points"])
    assert gain_captain > gain_other * 3


def test_captaincy_weight_zero_restores_previous_behaviour():
    frames = {1: _gw([8.0, 4.0], ids=[1, 2])}
    plain = horizon_points(frames, decay=1.0, captaincy_weight=0.0).set_index("player_id")
    assert plain.loc[1, "horizon_points"] == pytest.approx(8.0)


# --- risk ------------------------------------------------------------------


def test_captain_score_doubles_the_mean():
    scores = captain_score(_gw([6.0, 5.0]))
    assert scores.iloc[0] == pytest.approx(12.0)


def test_variance_weight_can_prefer_upside_over_mean():
    """The armband quadruples variance. A manager chasing a rank should be able to prefer a
    volatile pick; a manager maximising total points should not."""
    frame = pd.DataFrame({
        "player_id": [1, 2],
        "expected_points": [6.0, 5.6],
        "points_variance": [1.0, 25.0],
    })
    assert captain_score(frame, variance_weight=0.0).idxmax() == 0
    assert captain_score(frame, variance_weight=0.5).idxmax() == 1
