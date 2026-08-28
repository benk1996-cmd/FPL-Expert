"""One squad-value function, in one set of units."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.config import load_scoring_rules
from fpl_expert.optimise.valuation import deferral_gap, horizon_xi_value, squad_frames

RULES = load_scoring_rules()
POSITIONS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def _squad(ids=range(15)):
    ids = list(ids)
    return pd.DataFrame({
        "player_id": ids,
        "web_name": [f"P{i}" for i in ids],
        "position": POSITIONS[: len(ids)],
        "team": [f"C{i % 6}" for i in ids],
        "price": 5.0, "selling_price": 5.0, "p_zero": 0.1,
    })


def _per_gw(gws, points):
    return {
        gw: pd.DataFrame({"player_id": range(15), "expected_points": points})
        for gw in gws
    }


def test_the_horizon_is_a_sum_of_ELEVENS_not_of_fifteens():
    """The whole point. Four bench players must not contribute to any week's score."""
    value = horizon_xi_value(_squad(), _per_gw([2], [2.0] * 15), RULES, decay=0.84)
    assert value == pytest.approx(11 * 2.0 + 2.0)      # XI plus the armband, not 15 * 2.0


def test_later_gameweeks_are_discounted_by_the_same_decay_as_the_forecasts():
    one = horizon_xi_value(_squad(), _per_gw([2], [1.0] * 15), RULES, decay=0.5)
    three = horizon_xi_value(_squad(), _per_gw([2, 3, 4], [1.0] * 15), RULES, decay=0.5)
    assert three == pytest.approx(one * (1 + 0.5 + 0.25))


def test_a_blank_gameweek_drops_the_player_rather_than_scoring_him_zero():
    """No row means no fixture. `pick_xi` fills the eleven from whoever is available."""
    per_gw = _per_gw([2], [3.0] * 15)
    per_gw[2] = per_gw[2][per_gw[2]["player_id"] != 0]
    frames = squad_frames(_squad(), per_gw)
    assert len(frames[2]) == 14
    assert 0 not in set(frames[2]["player_id"])


def _pair():
    """Two fifteens differing in ONE forward: `now` holds the better man, `later` the worse."""
    upgraded = _squad()
    upgraded.loc[upgraded["player_id"] == 14, "player_id"] = 15      # swap in a better FWD
    return upgraded, _squad()


def _points_for(gws, better, worse):
    """Sixteen players: everyone flat, id 14 worth `worse` and id 15 worth `better`."""
    points = {i: 3.0 for i in range(16)}
    points[14], points[15] = worse, better
    return {
        gw: pd.DataFrame({"player_id": list(points), "expected_points": list(points.values())})
        for gw in gws
    }


def test_the_deferral_gap_is_only_the_weeks_before_the_squads_converge():
    """A transfer deferred one week costs one week of edge, not the whole horizon — every
    later term cancels because both branches then hold the same fifteen."""
    now, later = _pair()
    per_gw = _points_for([2, 3, 4, 5], better=9.0, worse=3.0)

    gap = deferral_gap(now, later, per_gw, RULES, weeks=1, decay=0.84)
    horizon = horizon_xi_value(now, per_gw, RULES, decay=0.84) - horizon_xi_value(
        later, per_gw, RULES, decay=0.84
    )

    assert gap > 0
    assert gap < horizon, "one week must be worth less than the whole horizon"
    assert horizon == pytest.approx(gap * (1 + 0.84 + 0.84**2 + 0.84**3))


def test_two_bought_forward_weeks_are_worth_more_than_one():
    now, later = _pair()
    per_gw = _points_for([2, 3, 4], better=9.0, worse=3.0)
    one = deferral_gap(now, later, per_gw, RULES, weeks=1, decay=0.84)
    two = deferral_gap(now, later, per_gw, RULES, weeks=2, decay=0.84)

    assert two > one
    assert two == pytest.approx(one * (1 + 0.84))
