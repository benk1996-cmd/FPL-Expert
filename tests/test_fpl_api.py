"""Parsers and gameweek logic. No network access."""

from __future__ import annotations

import pandas as pd
import pytest

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


# --- the HTTP cache must never be able to fail a command --------------------


def test_a_truncated_cache_entry_is_a_miss_not_a_crash(tmp_path, caplog):
    """A killed `fpl results` left a half-written `bootstrap-static` entry, and every command
    in the project then died with a bare "Aborted." — an EOFError from inside gzip, surfacing
    through Typer with no message and no indication of which file was at fault.

    The cache is disposable, so corruption may cost one refetch and nothing more.
    """
    import gzip
    import logging

    from fpl_expert.data.http import HttpClient

    client = HttpClient(cache_dir=tmp_path)
    url = "https://example.invalid/thing"
    client._write_cache(url, {"ok": True})
    assert client._read_cache(url) == {"ok": True}

    # truncate it the way an interrupted write does
    path = client._cache_path(url)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(EOFError), gzip.open(path, "rt", encoding="utf-8") as fh:
        fh.read()

    with caplog.at_level(logging.WARNING):
        assert client._read_cache(url) is None
    assert "corrupt cache entry" in caplog.text
    assert not path.exists(), "a damaged entry should be removed, not read again next time"


def test_cache_writes_are_atomic(tmp_path):
    """No temporary file may survive a write, and a half-written one must never be readable
    under the real name."""
    from fpl_expert.data.http import HttpClient

    client = HttpClient(cache_dir=tmp_path)
    client._write_cache("https://example.invalid/a", {"n": list(range(1000))})

    assert client._read_cache("https://example.invalid/a")["n"][-1] == 999
    assert not list(tmp_path.glob("*.tmp")), "temporary files must be renamed or cleaned up"
