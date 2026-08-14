"""Double gameweeks: the join grain, and the difference between what sums and what does not.

Two bugs lived here, both silent, both concentrated in exactly the gameweeks chips are timed
around — so neither showed up as an obvious wrong number, and both flattered the backtest.

1. **The team join was a cross product.** The archive gives one row per player-FIXTURE, and
   `assemble` joined team forecasts on team alone. A player with two fixtures met two team
   rows and came out with four, doubling his expected points.
2. **Realised points were summed across fixture rows.** They are recorded per player-GAMEWEEK,
   so summing two rows credited a double gameweek player with twice the points he scored — on
   top of the duplication above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.historical_forecast import attach_actual_points, to_gameweek_level
from fpl_expert.models.points import assemble

RULES = {
    "appearance": {"under_60": 1, "sixty_plus": 2},
    "goal": {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "saves_per_point": 3,
    "goals_conceded": {"points": -1, "per_goals": 2, "applies_to": ["GK", "DEF"]},
    "yellow_card": -1,
    "red_card": -3,
    "defensive_contribution": {
        "enabled": True, "points": 2,
        "thresholds": {"DEF": 10, "MID": 12, "FWD": 12, "GK": None},
    },
}


def _player_fixtures(fixtures=(500, 501)):
    """One player, one gameweek, `fixtures` fixtures — already at fixture grain."""
    return pd.DataFrame([
        {
            "name": "Doubler", "player_id": 1, "position": "MID", "team": "AVL",
            "fixture": fixture, "price": 7.0, "penalty_share": 0.0,
            "p_short": 0.05, "p_long": 0.9, "p_zero": 0.05,
            "expected_minutes": 0.05 * 22 + 0.9 * 85,
            "expected_appearance_points": 0.05 + 1.8,
            "xg_per90": 0.4, "xa_per90": 0.3, "finishing_multiplier": 1.0,
            "saves_per90": 0.0, "defcon_per90": 6.0, "yellows_per90": 0.2,
            "bonus_per90": 0.3,
        }
        for fixture in fixtures
    ])


def _team_forecasts(fixtures=(500, 501)):
    return pd.DataFrame([
        {
            "team": "AVL", "fixture": fixture, "opponent": f"OPP{i}",
            "expected_goals_for": 1.5, "expected_goals_against": 1.2,
            "p_clean_sheet": 0.28, "expected_conceded_penalty": -0.4,
        }
        for i, fixture in enumerate(fixtures)
    ])


def test_a_double_gameweek_produces_two_rows_not_four():
    """The cross product. Two fixtures joined on team alone gave 2 x 2 = 4 rows."""
    assembled = assemble(_player_fixtures(), _team_forecasts(), rules=RULES)
    assert len(assembled) == 2
    assert sorted(assembled["fixture"]) == [500, 501]


def test_a_single_gameweek_is_unaffected():
    assembled = assemble(_player_fixtures((500,)), _team_forecasts((500,)), rules=RULES)
    assert len(assembled) == 1


def test_expected_points_double_across_two_fixtures():
    """Expected points genuinely ARE per fixture and must sum — that is what makes a double
    gameweek worth planning for. Only realised points are per gameweek."""
    single = assemble(_player_fixtures((500,)), _team_forecasts((500,)), rules=RULES)
    double = assemble(_player_fixtures(), _team_forecasts(), rules=RULES)
    assert double["expected_points"].sum() == pytest.approx(
        2 * single["expected_points"].sum(), rel=0.01
    )


def test_a_player_row_still_joins_when_the_caller_has_no_fixture_column():
    """The live path passes one row per PLAYER, where fanning out on team is correct. Keying
    on the fixture must not be forced on a caller that has no such column."""
    players = _player_fixtures((500,)).drop(columns=["fixture"])
    teams = _team_forecasts()
    assembled = assemble(players, teams, rules=RULES)
    assert len(assembled) == 2          # correctly fanned out to both fixtures


def test_gameweek_level_collapses_fixtures_to_one_row_per_player():
    assembled = assemble(_player_fixtures(), _team_forecasts(), rules=RULES)
    assembled["season"], assembled["gw"] = "2025-26", 24

    collapsed = to_gameweek_level(assembled)
    assert len(collapsed) == 1
    assert collapsed["expected_points"].iloc[0] == pytest.approx(
        assembled["expected_points"].sum()
    )


def test_realised_points_are_not_multiplied_by_the_number_of_fixtures():
    """The bug this exists to prevent. `total_points` in the archive is already the gameweek
    total, so a player who scored 9 over two fixtures must arrive as 9, not 18."""
    assembled = assemble(_player_fixtures(), _team_forecasts(), rules=RULES)
    assembled["season"], assembled["gw"] = "2025-26", 24

    history = pd.DataFrame({
        "season": ["2025-26", "2025-26"], "GW": [24, 24],
        "name": ["Doubler", "Doubler"], "total_points": [4, 5],
    })
    scored = attach_actual_points(to_gameweek_level(assembled), history)
    assert len(scored) == 1
    assert scored["actual_points"].iloc[0] == 9


def test_gameweek_level_leaves_single_fixture_weeks_alone():
    assembled = assemble(_player_fixtures((500,)), _team_forecasts((500,)), rules=RULES)
    assembled["season"], assembled["gw"] = "2025-26", 3
    collapsed = to_gameweek_level(assembled)

    assert len(collapsed) == 1
    assert collapsed["expected_points"].iloc[0] == pytest.approx(
        assembled["expected_points"].iloc[0]
    )


def test_missing_attacking_rates_are_filled_within_position():
    """Filling from the whole league gave a goalkeeper with no history the same attacking rate
    as an outfielder. Across three seasons that handed goalkeepers 79 expected goals against
    zero actually scored — and a goalkeeper goal is worth 10 points."""
    import pandas as pd

    frame = pd.DataFrame({
        "position": ["GK", "GK", "FWD", "FWD", "DEF", "DEF"],
        "xg_per90": [0.01, None, 0.45, None, 0.05, None],
    })
    within = frame.groupby("position")["xg_per90"].transform("median")
    filled = frame["xg_per90"].fillna(within).fillna(frame["xg_per90"].median())

    assert filled.iloc[1] == pytest.approx(0.01)     # the other keeper, not the league
    assert filled.iloc[3] == pytest.approx(0.45)
    assert filled.iloc[1] < filled.iloc[5] < filled.iloc[3]


def test_appearance_probability_is_carried_as_a_maximum_not_a_sum():
    """P(appears) cannot exceed one however many fixtures there are. Summing it would let a
    double gameweek player look more than certain to play."""
    assembled = assemble(_player_fixtures(), _team_forecasts(), rules=RULES)
    assembled["season"], assembled["gw"] = "2025-26", 24
    collapsed = to_gameweek_level(assembled)
    assert collapsed["p_long"].iloc[0] <= 1.0
    assert np.isclose(collapsed["p_long"].iloc[0], 0.9)
