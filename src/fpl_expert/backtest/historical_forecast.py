"""Forecast a PAST gameweek using only what was knowable before its deadline.

The live pipeline reads through `PointInTime`, but those snapshots only begin in August 2026.
To evaluate anything before that, point-in-time state has to be reconstructed from the
archive instead. The archive supports this for most of what we need — it carries per-gameweek
prices (`value`), positions, teams and fixtures — with one important exception.

**Availability data does not exist historically.** `chance_of_playing_next_round`, `status`
and `news` are overwritten in place and archived by nobody, which is the entire reason Item 3
exists. So a historical simulation runs WITHOUT the availability gate that the live system
applies. Everything measured here therefore *understates* live performance, and by an unknown
amount. That is the right direction to be wrong, but it must be stated whenever these numbers
are quoted.

Two gameweeks, not one
----------------------

A forecast has two dates attached to it, and conflating them is a subtle way to leak the
future. There is the gameweek being FORECAST, and there is the gameweek the forecast is being
made AT. For scoring the week in front of you they coincide. For valuing a transfer over the
next six they do not: a manager planning at GW10 knows GW15's fixture list, but not the form,
price or fitness a player will carry into it.

`forecast_horizon` separates them. Everything derived from past results — the EWM minutes
features, the shrunk rates, the team model — is cut at the DECISION week, while the fixture
list, kickoff times and gameweek number come from the TARGET week, because FPL publishes the
schedule months ahead and revises it weeks ahead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..features.rates import season_gw_index
from ..models.match_sim import DixonColes, build_match_forecasts
from ..models.points import assemble
from ..pipeline import player_rates

log = logging.getLogger(__name__)

# The half of a minutes-feature row that is built from PAST RESULTS. When the target gameweek
# is not the decision gameweek these must come from the decision week instead, or the forecast
# reads form that had not happened yet.
#
# Deliberately NOT all of `FEATURE_COLUMNS`. `gw`, `rest_days` and `position_code` describe the
# published schedule and the player's registration, both knowable well in advance, so taking
# those from the target row is not lookahead.
STATE_COLUMNS = (
    "played_ewm", "started_ewm", "minutes_ewm",
    "minutes_lag1", "minutes_lag2", "started_lag1",
    "appearances", "career_games", "value",
    "season_games", "season_started_rate",
    "prev_season_started_rate", "prev_season_games",
)


def fixtures_for_gameweek(history: pd.DataFrame, season: str, gw: int) -> pd.DataFrame:
    """Recover the fixture list for a past gameweek from player rows.

    Both clubs in a match share a `fixture` id, so grouping by it and reading `was_home`
    recovers home and away without needing a team-id lookup — which matters because FPL
    reassigns team ids between seasons.
    """
    block = history[(history["season"] == season) & (history["GW"] == gw)]
    rows = []
    for fixture_id, group in block.groupby("fixture"):
        home = group.loc[group["was_home"], "team"].dropna().unique()
        away = group.loc[~group["was_home"].astype(bool), "team"].dropna().unique()
        if len(home) != 1 or len(away) != 1:
            continue     # incomplete fixture in the archive
        rows.append({
            "fixture": fixture_id, "home_team": home[0], "away_team": away[0],
            "event": gw, "kickoff_time": group["kickoff_time"].iloc[0],
        })
    return pd.DataFrame(rows)


def attach_opening_odds(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Join OPENING odds onto a fixture list, so the match model can blend with the market.

    Without this `build_match_forecasts` never sees a `p_home_open` column and silently falls
    back to Dixon-Coles alone — which is what every backtested number in this project actually
    described, despite `market_weight=0.8` and a README promising a blend. The data was always
    there; nothing joined it.

    Only `_open` columns exist in `odds_features` by construction: the physical split in
    `data/odds.py` keeps closing odds in a separate table that the feature path cannot reach,
    because closing odds postdate the FPL deadline. So this join cannot leak even by accident.
    """
    columns = [c for c in (
        "p_home_open", "p_draw_open", "p_away_open", "p_over25_open"
    ) if c in odds.columns]
    if not columns or fixtures.empty:
        return fixtures

    slim = odds.drop_duplicates(["home_team", "away_team", "season"])[
        ["season", "home_team", "away_team", *columns]
    ]
    keys = ["home_team", "away_team"]
    if "season" in fixtures.columns:
        keys.append("season")
    else:
        slim = slim.drop(columns=["season"])

    merged = fixtures.merge(slim, on=keys, how="left")
    matched = merged["p_home_open"].notna().mean() if len(merged) else 0.0
    if matched < 0.5:
        log.warning(
            "only %.0f%% of fixtures matched opening odds — the blend will mostly not fire",
            matched * 100,
        )
    return merged


def decision_state(features: pd.DataFrame, season: str, as_of_gw: int) -> pd.DataFrame:
    """Everything known about each player at one gameweek's deadline, one row per player.

    A double gameweek gives a player two feature rows, and the second is built by `shift(1)`
    over the FIRST — so it already contains that match's minutes. At the deadline neither
    match has been played, so the earlier kickoff's row is the state and the later one is
    discarded. That removes a leak affecting 1,766 of 86,765 rows, landing precisely on the
    weeks chips are timed around.

    `drop_duplicates`, not `groupby().first()`: the latter takes the first NON-NULL value
    per column independently, which would reach into the discarded row for exactly the
    lagged columns this exists to protect.
    """
    block = features[(features["season"] == season) & (features["GW"] == as_of_gw)]
    if block.empty:
        raise ValueError(f"no players found for {season} GW{as_of_gw}")
    order = pd.to_datetime(block["kickoff_time"], utc=True, errors="coerce")
    return (
        block.assign(_kickoff=order)
        .sort_values("_kickoff", kind="stable")
        .drop_duplicates("element", keep="first")
        .drop(columns="_kickoff")
    )


def forecast_horizon(
    history: pd.DataFrame,
    results: pd.DataFrame,
    minutes_model,
    features: pd.DataFrame,
    season: str,
    as_of_gw: int,
    targets: Iterable[int],
    *,
    market_weight: float = 0.8,
    rules: dict | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[int, pd.DataFrame]:
    """Forecast several gameweeks from ONE decision point, keyed by target gameweek.

    This is the honest way to value a horizon. `targets` are the gameweeks being forecast;
    `as_of_gw` is the week the forecast is made in, and everything derived from results is
    cut there — the shrunk rates, the Dixon-Coles fit, and the minutes model's form features.
    Only the fixture list and its kickoff times come from the target week.

    Batched rather than looped from outside because the expensive half of the work — fitting
    the team model and rebuilding league-wide rates — depends only on `as_of_gw`. Six targets
    cost barely more than one.
    """
    # The index must be built over the SAME frame it is compared against: it ranks seasons
    # densely, so computing it on a one-row Series makes every season rank 0 and collapses
    # `as_of` to the first gameweek — leaving almost no history and no xG columns.
    index = season_gw_index(history["season"], history["GW"])
    decision = index[(history["season"] == season) & (history["GW"] == as_of_gw)]
    if decision.empty:
        raise ValueError(f"{season} GW{as_of_gw} is not in the archive")
    as_of = decision.iloc[0]

    past = history[index < as_of]
    if past.empty:
        raise ValueError(f"no history before {season} GW{as_of_gw}")

    state = decision_state(features, season, as_of_gw)
    # --- rates from everything before the DECISION gameweek
    rates = player_rates(past, as_of).reset_index()

    # --- team model, fitted only on matches played before the DECISION gameweek kicked off
    decision_fixtures = fixtures_for_gameweek(history, season, as_of_gw)
    if decision_fixtures.empty:
        raise ValueError(f"no fixtures recovered for {season} GW{as_of_gw}")
    cutoff = pd.to_datetime(
        decision_fixtures["kickoff_time"], utc=True, errors="coerce"
    ).min()
    prior_results = results[pd.to_datetime(results["date"], utc=True) < cutoff]
    match_model = DixonColes().fit(prior_results, as_of=cutoff.tz_localize(None))

    out: dict[int, pd.DataFrame] = {}
    for target in targets:
        try:
            out[target] = _forecast_target(
                history, features, minutes_model, state, rates, match_model,
                season, as_of_gw, target,
                market_weight=market_weight, rules=rules, odds=odds,
            )
        except (ValueError, KeyError):
            # A horizon runs off the end of the season and over gameweeks the archive
            # records incompletely. Losing one of those shortens a window; losing the
            # DECISION week means the caller has no frame to act on at all, so that one
            # is never swallowed.
            if target == as_of_gw:
                raise
            log.info("%s GW%d: no forecast from GW%d", season, target, as_of_gw)
    return out


def _forecast_target(
    history, features, minutes_model, state, rates, match_model, season, as_of_gw, target,
    *, market_weight, rules, odds,
) -> pd.DataFrame:
    """One target gameweek, forecast from an already-built decision-week view of the league."""
    current = features[(features["season"] == season) & (features["GW"] == target)]
    if current.empty:
        raise ValueError(f"no players found for {season} GW{target}")

    # Players with no row at the decision week cannot be forecast from it — a January signing
    # is not knowable in October. Dropping them is right: they are equally absent from that
    # week's decision frame, so nothing can be transferred in that this excludes.
    known = state.set_index("element")
    current = current[current["element"].isin(known.index)].reset_index(drop=True).copy()
    if current.empty:
        raise ValueError(f"no {season} GW{target} players were known at GW{as_of_gw}")

    missing = [c for c in STATE_COLUMNS if c not in current.columns or c not in known.columns]
    if missing:
        raise KeyError(f"minutes features are missing state columns: {missing}")
    for col in STATE_COLUMNS:
        current[col] = current["element"].map(known[col])

    # --- minutes. No availability gate: the data to apply one does not exist historically.
    # Attached positionally, not merged on name: a player has one row per fixture, so a
    # double gameweek gives duplicate names and a name-keyed merge would explode.
    minutes = minutes_model.predict(current).reset_index(drop=True)

    fixtures = fixtures_for_gameweek(history, season, target)
    if fixtures.empty:
        raise ValueError(f"no fixtures recovered for {season} GW{target}")
    if odds is not None:
        fixtures = attach_opening_odds(fixtures.assign(season=season), odds)
    team_forecasts = build_match_forecasts(fixtures, match_model, market_weight=market_weight)

    # `fixture` travels with the player rows so that `assemble` can key the team join on it.
    # Archive rows are already one per player-fixture, so a team-only join would cross-product
    # a double gameweek into four rows per player.
    base = current[["name", "position", "team", "value", "element", "fixture"]].rename(
        columns={"element": "player_id"}
    ).reset_index(drop=True)
    base["price"] = base["value"] / 10.0
    base = pd.concat([base, minutes], axis=1)
    base = base.merge(rates, on="name", how="left")
    base["penalty_share"] = 0.0      # no set-piece order exists in the archive
    # Fill missing rates WITHIN POSITION. Filling from the whole league gave every player
    # without rate history the same attacking rate, which is nonsense for a goalkeeper: it
    # left GKs with a median xG per 90 of 0.111, HIGHER than defenders, and the allocator
    # duly handed goalkeepers 79 expected goals across three seasons against zero scored.
    # At 10 points a goalkeeper goal that is not a rounding error.
    for col in ("xg_per90", "xa_per90", "finishing_multiplier"):
        within_position = base.groupby("position")[col].transform("median")
        base[col] = base[col].fillna(within_position).fillna(base[col].median())

    forecasts = assemble(
        base,
        team_forecasts[[
            # `opponent` is required: without it the bonus model cannot identify which two
            # squads competed for a match's six bonus points and silently falls back to
            # predicting each player independently.
            "team", "fixture", "opponent", "expected_goals_for", "expected_goals_against",
            "p_clean_sheet", "expected_conceded_penalty",
        ]],
        rules=rules,
    )
    forecasts["season"], forecasts["gw"] = season, target
    # Carried so a stored horizon table says which decision point each row belongs to.
    forecasts["as_of_gw"] = as_of_gw

    # Realised minutes are attached ONLY when the target IS the decision week, where the
    # simulator needs them for the vice-captain rule. Putting them on a lookahead row would
    # place an outcome inside a forward valuation — the precise thing this module separates
    # the two gameweeks to prevent.
    #
    # Mapped, not assigned positionally: in a DOUBLE gameweek a team appears twice in
    # `team_forecasts`, so the merge inside `assemble` legitimately duplicates player rows
    # and `forecasts` is longer than `current`. Double gameweeks are precisely the weeks
    # chips are timed around, so silently dropping them would hide the cases that matter.
    if target == as_of_gw:
        actual = current.groupby("name")["minutes"].sum()
        forecasts["actual_minutes"] = forecasts["name"].map(actual)
    else:
        forecasts["actual_minutes"] = np.nan
    return forecasts


def forecast_past_gameweek(
    history: pd.DataFrame,
    results: pd.DataFrame,
    minutes_model,
    features: pd.DataFrame,
    season: str,
    gw: int,
    *,
    as_of_gw: int | None = None,
    market_weight: float = 0.8,
    rules: dict | None = None,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expected points for every player in a past gameweek.

    `features` is the full minutes-feature frame, built once and filtered here — every one
    of its columns is already `shift(1)`-based, so a row cannot see its own result.
    `results` is the match table used to fit the team model, cut to matches before the
    decision gameweek's first kickoff.

    `as_of_gw` defaults to `gw`, which is the scoring case: forecast the week in front of
    you. Pass an earlier week to get the forecast a manager would have made then.
    """
    decision = gw if as_of_gw is None else as_of_gw
    return forecast_horizon(
        history, results, minutes_model, features, season, decision, [gw],
        market_weight=market_weight, rules=rules, odds=odds,
    )[gw]


def to_gameweek_level(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-fixture forecast rows to one row per player-gameweek.

    The live pipeline already does this (`points.aggregate_gameweek`) before optimising; the
    backtest did not, and the omission was not cosmetic. Realised points are recorded per
    player-GAMEWEEK, so leaving two fixture rows in place made the simulator count a double
    gameweek player's actual score twice — crediting points he never scored, in precisely the
    weeks Bench Boost and Triple Captain are timed around.

    Expected points, by contrast, are genuinely per fixture and SHOULD sum across a double
    gameweek. That asymmetry is the whole reason this has to happen before actuals are joined
    on rather than after.
    """
    from ..models.points import aggregate_gameweek

    # `as_of_gw` joins the grouping key when present. A horizon table holds the same target
    # gameweek forecast from several decision points, and collapsing on season and target
    # alone would fold six different forecasts of GW15 into one player row.
    keys = [c for c in ("season", "as_of_gw", "gw") if c in forecasts.columns]
    pieces = []
    for values, block in forecasts.groupby(keys, sort=True):
        aggregated = aggregate_gameweek(block.reset_index(drop=True))
        for key, value in zip(keys, values if isinstance(values, tuple) else (values,),
                              strict=True):
            aggregated[key] = value
        pieces.append(aggregated)
    return pd.concat(pieces, ignore_index=True)


def attach_actual_points(forecasts: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Join realised FPL points onto a forecast frame."""
    key = ["season", "GW", "name"]
    actual = history[[*key, "total_points"]].rename(
        columns={"GW": "gw", "total_points": "actual_points"}
    )
    actual = actual.groupby(["season", "gw", "name"], as_index=False)["actual_points"].sum()
    return forecasts.merge(actual, on=["season", "gw", "name"], how="left")
