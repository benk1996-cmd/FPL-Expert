"""Parsers and gameweek logic. No network access."""

from __future__ import annotations

import pandas as pd

from fpl_expert.data.fpl_api import (
    current_gameweek,
    next_gameweek,
    parse_events,
    parse_players,
    parse_teams,
)


def test_parse_players_converts_prices_and_positions(bootstrap):
    df = parse_players(bootstrap)
    assert df.loc[df["id"] == 1, "position"].iloc[0] == "GK"
    assert df.loc[df["id"] == 2, "position"].iloc[0] == "MID"
    # now_cost is in tenths of a million; Saka at 100 is £10.0m.
    assert df.loc[df["id"] == 2, "price"].iloc[0] == 10.0


def test_parse_players_coerces_stringly_typed_numerics(bootstrap):
    """The API sends percentages, form and xG as strings; everything downstream wants floats."""
    df = parse_players(bootstrap)
    assert pd.api.types.is_numeric_dtype(df["selected_by_percent"])
    assert pd.api.types.is_numeric_dtype(df["expected_goals"])
    assert df.loc[df["id"] == 2, "selected_by_percent"].iloc[0] == 41.2


def test_parse_events_makes_deadlines_tz_aware(bootstrap):
    df = parse_events(bootstrap)
    assert df["deadline_time"].dt.tz is not None
    assert str(df["deadline_time"].iloc[0]) == "2026-08-21 17:30:00+00:00"


def test_gameweek_helpers_before_season_start(bootstrap):
    """Pre-season: nothing is current, but GW1 is next — the gameweek we are deciding."""
    assert current_gameweek(bootstrap) is None
    assert next_gameweek(bootstrap) == 1


def test_next_gameweek_falls_back_when_no_is_next_flag(bootstrap):
    """Between a deadline and the flags updating, no event carries is_next."""
    for event in bootstrap["events"]:
        event["is_next"] = False
    bootstrap["events"][0]["finished"] = True
    assert next_gameweek(bootstrap) == 2


def test_parse_teams(bootstrap):
    assert parse_teams(bootstrap)["name"].tolist() == ["Arsenal"]
