"""Fixture context, with particular attention to blanks and doubles."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.features.fixtures import (
    add_rest_days,
    team_fixtures,
    team_gameweek_grid,
)

TEAMS = pd.DataFrame({"id": [1, 2, 3], "name": ["Arsenal", "Chelsea", "Spurs"]})


def _fixtures():
    return pd.DataFrame({
        "id": [10, 11, 12, 13],
        "event": [1, 1, 2, 2],
        "team_h": [1, 3, 2, 1],
        "team_a": [2, 1, 1, 3],   # Arsenal plays twice in GW2 -> double
        "kickoff_time": [
            "2026-08-21T19:00:00Z", "2026-08-22T14:00:00Z",
            "2026-08-28T19:00:00Z", "2026-08-30T14:00:00Z",
        ],
    })


def test_team_fixtures_gives_one_row_per_team_per_fixture():
    tf = team_fixtures(_fixtures(), TEAMS)
    assert len(tf) == 8                                   # 4 fixtures x 2 teams
    arsenal_gw1 = tf[(tf["team"] == 1) & (tf["gw"] == 1)]
    assert len(arsenal_gw1) == 2                          # Arsenal plays twice in GW1 here
    assert set(arsenal_gw1["is_home"]) == {True, False}


def test_team_fixtures_orients_opponent_correctly():
    tf = team_fixtures(_fixtures(), TEAMS)
    home_row = tf[(tf["fixture_id"] == 10) & (tf["team"] == 1)].iloc[0]
    away_row = tf[(tf["fixture_id"] == 10) & (tf["team"] == 2)].iloc[0]
    assert home_row["is_home"] and home_row["opponent_name"] == "Chelsea"
    assert not away_row["is_home"] and away_row["opponent_name"] == "Arsenal"


def test_team_fixtures_requires_the_columns_it_needs():
    with pytest.raises(KeyError, match="team_h"):
        team_fixtures(pd.DataFrame({"event": [1]}))


def test_grid_makes_blank_gameweeks_explicit():
    """A blank shows up as an ABSENT row in the fixture list, which is easy to miss and
    leads to a forecast quietly reusing the previous gameweek."""
    tf = team_fixtures(_fixtures(), TEAMS)
    grid = team_gameweek_grid(tf, team_ids=[1, 2, 3], gameweeks=[1, 2, 3])

    assert len(grid) == 9                                  # 3 teams x 3 gameweeks
    gw3 = grid[grid["gw"] == 3]
    assert gw3["is_blank"].all()                           # nobody plays in GW3
    assert (gw3["fixture_count"] == 0).all()

    chelsea_gw2 = grid[(grid["team"] == 2) & (grid["gw"] == 2)].iloc[0]
    assert chelsea_gw2["fixture_count"] == 1


def test_grid_flags_double_gameweeks():
    tf = team_fixtures(_fixtures(), TEAMS)
    grid = team_gameweek_grid(tf, team_ids=[1, 2, 3], gameweeks=[1, 2])
    arsenal_gw2 = grid[(grid["team"] == 1) & (grid["gw"] == 2)].iloc[0]
    assert arsenal_gw2["fixture_count"] == 2
    assert arsenal_gw2["is_double"]
    assert not arsenal_gw2["is_blank"]


def test_rest_days_measures_the_gap_since_the_previous_fixture():
    tf = add_rest_days(team_fixtures(_fixtures(), TEAMS))
    arsenal = tf[tf["team"] == 1].sort_values("kickoff_time")
    # 21 Aug 19:00 -> 22 Aug 14:00 is 0.79 days; a genuinely congested turnaround.
    assert arsenal["rest_days"].iloc[1] == pytest.approx(0.79, abs=0.02)


def test_rest_days_are_capped():
    """An uncapped first-fixture or post-international gap would dominate a linear model."""
    tf = add_rest_days(team_fixtures(_fixtures(), TEAMS), max_days=7.0)
    assert tf["rest_days"].max() <= 7.0
    assert tf["rest_days"].notna().all()


def test_rest_days_without_kickoff_times_degrades_gracefully():
    fixtures = _fixtures().drop(columns=["kickoff_time"])
    tf = add_rest_days(team_fixtures(fixtures, TEAMS))
    assert tf["rest_days"].isna().all()
