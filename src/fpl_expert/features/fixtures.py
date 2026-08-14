"""Fixture context: opponent, venue, congestion, and blank/double gameweeks.

The natural grain for player features is one row per team per fixture, not one per fixture,
because a player's gameweek is defined by what his club does. Two FPL-specific wrinkles make
this more than a reshape:

*Blank gameweeks.* A club can have no fixture in a gameweek at all (cup competitions
displace league games). Its players score nothing. A blank is easy to miss because it shows
up as an *absent row* rather than a zero — the classic silent failure is a forecast that
quietly reuses the previous gameweek's fixture. So a complete team x gameweek grid is built
explicitly, with `fixture_count = 0` marking blanks.

*Double gameweeks.* A club can also play twice, and its players then accumulate points from
both matches. Double gameweeks are when Bench Boost and Triple Captain earn their keep, so
they must be first-class rather than an edge case.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def team_fixtures(fixtures: pd.DataFrame, teams: pd.DataFrame | None = None) -> pd.DataFrame:
    """Explode fixtures to one row per team per fixture, from that team's point of view."""
    required = {"event", "team_h", "team_a"}
    if missing := required - set(fixtures.columns):
        raise KeyError(f"fixtures missing {sorted(missing)}")

    frames = []
    for venue, own, opp in (("H", "team_h", "team_a"), ("A", "team_a", "team_h")):
        block = pd.DataFrame({
            "gw": fixtures["event"],
            "team": fixtures[own],
            "opponent": fixtures[opp],
            "is_home": venue == "H",
            "fixture_id": fixtures.get("id", pd.Series(range(len(fixtures)))),
        })
        if "kickoff_time" in fixtures:
            block["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], utc=True)
        if "finished" in fixtures:
            block["finished"] = fixtures["finished"]
        frames.append(block)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["gw", "team"])
    out["gw"] = out["gw"].astype(int)
    if teams is not None:
        names = teams.set_index("id")["name"]
        out["team_name"] = out["team"].map(names)
        out["opponent_name"] = out["opponent"].map(names)
    return out.sort_values(["team", "gw"]).reset_index(drop=True)


def add_rest_days(tf: pd.DataFrame, max_days: float = 14.0) -> pd.DataFrame:
    """Days since each team's previous fixture — the congestion signal behind rotation.

    Capped, because a first fixture of the season or a post-international gap says nothing
    useful about fatigue and an uncapped value would dominate any linear model.
    """
    if "kickoff_time" not in tf.columns:
        tf = tf.copy()
        tf["rest_days"] = np.nan
        return tf

    out = tf.sort_values(["team", "kickoff_time"]).copy()
    delta = out.groupby("team")["kickoff_time"].diff().dt.total_seconds() / 86400
    out["rest_days"] = delta.clip(upper=max_days)
    # First fixture of the sample has no predecessor; treat as fully rested.
    out["rest_days"] = out["rest_days"].fillna(max_days)
    return out.sort_index()


def team_gameweek_grid(
    tf: pd.DataFrame, team_ids: list[int] | None = None, gameweeks: list[int] | None = None
) -> pd.DataFrame:
    """Complete team x gameweek grid with fixture counts.

    Every team appears in every gameweek, so blanks are explicit zeros rather than missing
    rows. Returns `is_blank` and `is_double` alongside the count.
    """
    team_ids = sorted(tf["team"].unique()) if team_ids is None else team_ids
    gameweeks = sorted(tf["gw"].unique()) if gameweeks is None else gameweeks

    grid = pd.MultiIndex.from_product([team_ids, gameweeks], names=["team", "gw"]).to_frame(False)
    counts = tf.groupby(["team", "gw"]).size().rename("fixture_count")
    grid = grid.merge(counts, on=["team", "gw"], how="left")
    grid["fixture_count"] = grid["fixture_count"].fillna(0).astype(int)
    grid["is_blank"] = grid["fixture_count"] == 0
    grid["is_double"] = grid["fixture_count"] >= 2
    return grid


def attach_odds(tf: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join pre-deadline odds onto team-fixtures, oriented to each team's perspective.

    Odds are quoted home/draw/away; a team's own win probability depends on its venue, so
    they are flipped for away sides. Only `_open` columns are consumed — closing odds are
    not reachable from the feature path (see `data/snapshot.py`).
    """
    if "team_name" not in tf.columns:
        raise KeyError("attach_odds needs team_name/opponent_name — pass `teams` to team_fixtures")

    cols = ["home_team", "away_team", "p_home_open", "p_draw_open", "p_away_open"]
    available = [c for c in cols if c in odds.columns]
    if len(available) < len(cols):
        log.warning("odds missing %s — skipping odds features", sorted(set(cols) - set(available)))
        return tf

    slim = odds[available + [c for c in ("p_over25_open",) if c in odds.columns]]
    home = slim.rename(columns={"home_team": "team_name", "away_team": "opponent_name"})
    away = slim.rename(columns={"away_team": "team_name", "home_team": "opponent_name"})

    out = tf.merge(home, on=["team_name", "opponent_name"], how="left", suffixes=("", "_h"))
    out = out.merge(away, on=["team_name", "opponent_name"], how="left", suffixes=("", "_a"))

    # Orient to "this team wins" regardless of venue.
    p_home = out["p_home_open"].where(out["is_home"], out.get("p_away_open_a"))
    p_away = out["p_away_open"].where(out["is_home"], out.get("p_home_open_a"))
    out["p_win"] = p_home
    out["p_lose"] = p_away
    out["p_draw"] = out["p_draw_open"].where(out["is_home"], out.get("p_draw_open_a"))
    return out.drop(columns=[c for c in out.columns if c.endswith(("_h", "_a"))], errors="ignore")
