"""Ingesting the season in progress into the archive the models train on."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.data.current_season import (
    build_current_season,
    player_identity,
    settled_gameweeks,
)


def _bootstrap(events):
    return {
        "events": events,
        "elements": [
            {"id": 1, "first_name": "Dominic", "second_name": "Calvert-Lewin",
             "element_type": 4, "team": 10},
            {"id": 2, "first_name": "João", "second_name": "Pedro",
             "element_type": 4, "team": 6},
        ],
        "teams": [{"id": 10, "name": "Leeds"}, {"id": 6, "name": "Chelsea"}],
    }


def _events(*specs):
    return [
        {"id": i, "name": f"Gameweek {i}", "deadline_time": "2026-08-21T17:30:00Z",
         "finished": f, "data_checked": d, "is_next": False, "is_current": False}
        for i, (f, d) in enumerate(specs, start=1)
    ]


class _Api:
    """Serves a fixed bootstrap and per-player history."""

    def __init__(self, bootstrap, history):
        self._bootstrap, self._history = bootstrap, history

    def bootstrap_static(self, **_):
        return self._bootstrap

    def element_summary(self, element, **_):
        return {"history": self._history.get(element, [])}


def _row(element, rnd, **over):
    row = {
        "element": element, "round": rnd, "minutes": 90, "goals_scored": 1,
        "assists": 0, "expected_goals": 0.5, "expected_assists": 0.1, "bonus": 3,
        "saves": 0, "yellow_cards": 0, "defensive_contribution": 4, "value": 60,
        "kickoff_time": "2026-08-22T14:00:00Z", "opponent_team": 3, "was_home": True,
        "fixture": 5, "total_points": 9,
    }
    row.update(over)
    return row


def test_only_finished_and_checked_gameweeks_are_settled():
    """`finished` flips before `data_checked`, while bonus is still provisional."""
    events = _events((True, True), (True, False), (False, False))
    assert settled_gameweeks({"events": events}) == [1]


def test_no_settled_gameweeks_is_an_error_not_an_empty_partition():
    """Writing an empty partition would look exactly like a season where nobody played."""
    api = _Api(_bootstrap(_events((False, False))), {})
    with pytest.raises(RuntimeError, match="no settled gameweeks"):
        build_current_season(api, season="2026-27")


def test_identity_spells_names_the_way_the_archive_does():
    """`first_name + " " + second_name` is the join key into every rate. If this drifts, the
    join silently misses and the player falls back to a league-wide prior."""
    identity = player_identity(_bootstrap(_events((True, True))))

    assert identity.loc[identity["element"] == 1, "name"].iloc[0] == "Dominic Calvert-Lewin"
    assert identity.loc[identity["element"] == 2, "name"].iloc[0] == "João Pedro"
    assert identity.loc[identity["element"] == 1, "position"].iloc[0] == "FWD"
    assert identity.loc[identity["element"] == 1, "team"].iloc[0] == "Leeds"


def test_unsettled_gameweeks_are_excluded_from_the_rows():
    """A player's history includes the live gameweek; only settled ones may be written."""
    api = _Api(
        _bootstrap(_events((True, True), (False, False))),
        {1: [_row(1, 1), _row(1, 2)], 2: [_row(2, 1)]},
    )
    df = build_current_season(api, season="2026-27")

    assert set(df["GW"]) == {1}
    assert len(df) == 2


def test_rows_carry_everything_the_rate_and_minutes_models_read():
    from fpl_expert.pipeline import RATE_STATS

    api = _Api(_bootstrap(_events((True, True))), {1: [_row(1, 1)], 2: [_row(2, 1)]})
    df = build_current_season(api, season="2026-27")

    for column in [*RATE_STATS, "name", "position", "season", "GW", "minutes",
                   "value", "kickoff_time"]:
        assert column in df.columns, column
    assert df["season"].unique().tolist() == ["2026-27"]
    assert pd.api.types.is_datetime64_any_dtype(df["kickoff_time"])


def test_a_double_gameweek_keeps_both_fixtures_as_separate_rows():
    """`element-summary` is per FIXTURE, which is why it is used instead of `event/N/live` —
    that endpoint collapses a double into one entry with no per-fixture expected goals."""
    api = _Api(
        _bootstrap(_events((True, True))),
        {1: [_row(1, 1, fixture=5), _row(1, 1, fixture=6)], 2: [_row(2, 1)]},
    )
    df = build_current_season(api, season="2026-27")

    assert len(df[df["element"] == 1]) == 2
    assert sorted(df[df["element"] == 1]["fixture"]) == [5, 6]


def test_columns_the_live_feed_cannot_supply_are_absent_not_zero():
    """The archive's discipline: a missing stat must read as NaN, never as a real zero."""
    api = _Api(_bootstrap(_events((True, True))), {1: [_row(1, 1)], 2: [_row(2, 1)]})
    df = build_current_season(api, season="2026-27")

    assert df["xP"].isna().all()


def test_string_typed_stats_are_coerced_to_the_archive_dtype():
    """The API sends expected goals as "0.63". The archive stores float64.

    Concatenating the two gave an object column that survived the write and then failed on the
    first multiplication, inside the rate calculation, with a traceback pointing at pandas.
    """
    from fpl_expert.data.current_season import STRING_TYPED_NUMERICS

    api = _Api(
        _bootstrap(_events((True, True))),
        {1: [_row(1, 1, expected_goals="0.63", expected_assists="0.10", ict_index="7.4")],
         2: [_row(2, 1, expected_goals="0.13", expected_assists="0.00", ict_index="2.1")]},
    )
    df = build_current_season(api, season="2026-27")

    for column in STRING_TYPED_NUMERICS:
        if column in df:
            assert pd.api.types.is_numeric_dtype(df[column]), column
    assert df["expected_goals"].sum() == pytest.approx(0.76)
    assert (df["expected_goals"] * 2.0).sum() == pytest.approx(1.52)   # arithmetic works


def test_the_written_partition_matches_the_archive_schema():
    """The guard that would have caught the dtype bug at ingestion instead of at use.

    Skipped when the partitions are absent, so a fresh checkout does not fail on data it has
    not downloaded yet.
    """
    from fpl_expert.data.current_season import schema_mismatches
    from fpl_expert.data.storage import read_table

    try:
        new = read_table("interim", "history", season="2026-27")
        archive = read_table("interim", "history", season="2025-26")
    except FileNotFoundError:
        pytest.skip("history partitions not ingested in this checkout")

    assert schema_mismatches(new, archive) == []
