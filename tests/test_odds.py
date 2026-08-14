"""Odds parsing, de-vigging and bookmaker selection. No network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.data.odds import _pick_source, _read_csv, devig_proportional, normalise, season_code
from fpl_expert.data.teams import UnknownTeamError, normalise_series


def test_season_code():
    assert season_code("2025-26") == "2526"
    assert season_code("2019-20") == "1920"


def test_devig_removes_the_margin():
    """Probabilities must sum to 1 after de-vigging, and the margin must be recorded."""
    # 1/2 + 1/4 + 1/2 = 1.25, i.e. a 25% overround.
    odds = pd.DataFrame({"home": [2.0], "draw": [4.0], "away": [2.0]})
    out = devig_proportional(odds)
    assert out[["home", "draw", "away"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
    assert out["overround"].iloc[0] == pytest.approx(1.25)
    assert out["home"].iloc[0] == pytest.approx(0.4)   # 0.50 / 1.25
    assert out["draw"].iloc[0] == pytest.approx(0.2)   # 0.25 / 1.25


def test_devig_is_a_no_op_on_a_fair_book():
    """Odds already summing to 1.0 carry no margin and must pass through unchanged."""
    odds = pd.DataFrame({"home": [2.0], "draw": [4.0], "away": [4.0]})
    out = devig_proportional(odds)
    assert out["overround"].iloc[0] == pytest.approx(1.0)
    assert out["home"].iloc[0] == pytest.approx(0.5)


def test_devig_returns_nan_for_incomplete_rows():
    """A missing price must yield NaN, not a zero overround.

    pandas sums an all-NaN row to 0.0 by default, which produced an 'overround' of 0 —
    an impossible free arbitrage — and silently corrupted the season average.
    """
    odds = pd.DataFrame({"home": [np.nan], "draw": [np.nan], "away": [np.nan]})
    out = devig_proportional(odds)
    assert np.isnan(out["overround"].iloc[0])

    partial = pd.DataFrame({"home": [2.0], "draw": [np.nan], "away": [4.0]})
    assert np.isnan(devig_proportional(partial)["overround"].iloc[0])


def test_pick_source_rejects_a_sharp_book_with_poor_coverage():
    """Pinnacle is sharpest but football-data carries it for only part of some seasons.

    Preferring it blindly would drop 45% of 2025-26's fixtures.
    """
    df = pd.DataFrame({
        "PSCH": [2.0] * 10, "PSCD": [3.0] * 5 + [np.nan] * 5, "PSCA": [4.0] * 10,
        "AvgCH": [2.0] * 10, "AvgCD": [3.0] * 10, "AvgCA": [4.0] * 10,
    })
    prefix, label, cov = _pick_source(df, [("PSC", "Pinnacle"), ("AvgC", "market average")])
    assert prefix == "AvgC" and label == "market average"
    assert cov == 1.0


def test_pick_source_prefers_sharpest_when_coverage_is_good():
    df = pd.DataFrame({
        "PSCH": [2.0] * 10, "PSCD": [3.0] * 10, "PSCA": [4.0] * 10,
        "AvgCH": [2.0] * 10, "AvgCD": [3.0] * 10, "AvgCA": [4.0] * 10,
    })
    prefix, label, _ = _pick_source(df, [("PSC", "Pinnacle"), ("AvgC", "market average")])
    assert prefix == "PSC" and label == "Pinnacle"


def test_read_csv_handles_bom():
    """fixtures.csv is UTF-8 with a BOM; latin-1 decoding welds 'ï»¿' onto the first header,
    so the division filter matches nothing and upcoming fixtures vanish."""
    raw = "﻿Div,Date,HomeTeam,AwayTeam\nE0,21/08/2026,Arsenal,Chelsea\n".encode("utf-8-sig")
    df = _read_csv(raw)
    assert "Div" in df.columns
    assert df["Div"].iloc[0] == "E0"


def test_read_csv_falls_back_to_latin1():
    """Season files are latin-1 and would raise on a strict utf-8 decode."""
    raw = "Div,Date,HomeTeam,AwayTeam,Referee\nE0,21/08/2026,Arsenal,Chelsea,M\xfcller\n".encode(
        "latin-1"
    )
    df = _read_csv(raw)
    assert df["Referee"].iloc[0] == "Müller"


def test_normalise_maps_team_names_and_dates():
    df = pd.DataFrame({
        "Date": ["21/08/2026"], "HomeTeam": ["Man United"], "AwayTeam": ["Tottenham"],
        "FTHG": [2], "FTAG": [1],
        "AvgCH": [2.0], "AvgCD": [4.0], "AvgCA": [2.0],   # 25% overround
    })
    out = normalise(df, "2026-27")
    assert out["home_team"].iloc[0] == "Man Utd"     # FPL short name, not football-data's
    assert out["away_team"].iloc[0] == "Spurs"
    assert out["date"].iloc[0] == pd.Timestamp("2026-08-21")  # dd/mm/yyyy, not mm/dd
    assert out["p_home_close"].iloc[0] == pytest.approx(0.4)


def test_unknown_team_raises_rather_than_silently_dropping_odds():
    """A rename upstream must fail loudly — otherwise that fixture just loses its odds
    and the match model quietly falls back to a prior."""
    with pytest.raises(UnknownTeamError, match="Real Madrid"):
        normalise_series(pd.Series(["Arsenal", "Real Madrid"]))
