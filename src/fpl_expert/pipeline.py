"""End-to-end forecast pipeline: snapshot -> rates -> models -> expected points.

Every read of current-season state goes through `PointInTime`, so the same code path serves
live decisions and point-in-time backtests. Nothing here reaches for a table directly.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import project_root
from .data.historical import load_history
from .data.snapshot import PointInTime
from .data.storage import read_table
from .features.rates import decayed_totals, season_gw_index
from .models.attack import attacking_rates, penalty_shares
from .models.match_sim import DixonColes, build_match_forecasts
from .models.minutes import MinutesModel, apply_availability_gate, build_features
from .models.points import assemble

log = logging.getLogger(__name__)

POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Rate statistics pulled from the archive. xG-based stats only exist from 2022-23, which is
# why the rate window starts there rather than at the beginning of the archive.
RATE_STATS = [
    "expected_goals", "expected_assists", "goals_scored", "assists",
    "saves", "bonus", "yellow_cards", "defensive_contribution",
]
RATE_HISTORY_FROM = "2022-23"


def _full_name(players: pd.DataFrame) -> pd.Series:
    return players["first_name"].fillna("") + " " + players["second_name"].fillna("")


def _stat_seasons(history: pd.DataFrame, stat: str) -> list[str]:
    """Seasons in which a statistic was actually recorded.

    Essential because `decayed_totals` zero-fills missing values. A stat introduced recently
    — `defensive_contribution` exists only from 2025-26 — would otherwise accumulate three
    seasons of exposure against zero counts and come out diluted by a factor of several.
    This is the same failure `historical.coverage_report` exists to expose.
    """
    if stat not in history.columns:
        return []
    seen = history.groupby("season")[stat].apply(lambda s: s.notna().any())
    return sorted(seen[seen].index)


def player_rates(history: pd.DataFrame, as_of: float) -> pd.DataFrame:
    """Shrunk per-90 rates for every stat the assembler needs, keyed by player name.

    Rates are computed per availability window rather than in one pass, so a recently
    introduced statistic is measured only over the seasons that actually recorded it.
    """
    available = [s for s in RATE_STATS if s in history.columns]
    windows: dict[tuple[str, ...], list[str]] = {}
    for stat in available:
        windows.setdefault(tuple(_stat_seasons(history, stat)), []).append(stat)

    frames = []
    for seasons, stats in windows.items():
        if not seasons:
            continue
        subset = history[history["season"].isin(seasons)]
        block = decayed_totals(subset, as_of=as_of, stats=stats, key="name")
        block = block.rename(columns={f"_c_{s}": s for s in stats})
        # Exposure is window-specific: a rate must be divided by the minutes played inside
        # the same window that produced its counts.
        block = block.rename(columns={"_exposure": f"exposure_{len(frames)}"})
        frames.append(block[[*stats, f"exposure_{len(frames)}"]])
        log.info("rates for %s over %d season(s): %s", stats, len(seasons), seasons[0])

    if not frames:
        raise ValueError("no rate statistics available in the supplied history")
    totals = pd.concat(frames, axis=1)
    totals["exposure_90s"] = totals["exposure_0"]
    missing = {"expected_goals", "goals_scored"} - set(totals.columns)
    if missing:
        raise ValueError(
            f"history lacks {sorted(missing)} — xG-based columns only exist from 2022-23, "
            f"so gameweeks before then cannot be forecast by this pipeline"
        )
    totals = totals.rename(columns={
        "expected_goals": "xg", "expected_assists": "xa", "goals_scored": "goals",
    })

    # Position is attached ONLY so the shrinkage can group by it, then dropped again below.
    # Without it `attacking_rates` finds no grouping column, quietly falls back to a single
    # league-wide prior, and shrinks goalkeepers toward an outfielder's attacking rate. That
    # gave GKs a median 0.111 xG per 90 — higher than defenders — and the allocator handed
    # them 79 expected goals across three seasons against zero scored, at 10 points each.
    last_position = (
        history.sort_values(["season", "GW"]).groupby("name")["position"].last()
        if "position" in history.columns else None
    )
    if last_position is not None:
        totals = totals.join(last_position.rename("position"))

    rates = attacking_rates(totals)
    out = totals.join(rates)
    # Historical `team`/`position` describe where the player WAS. The live snapshot is
    # authoritative for both — transfers happen — so drop them rather than let them collide
    # on the merge and silently win.
    out = out.drop(columns=[c for c in ("team", "position", "tier", "web_name") if c in out])

    # Secondary components: each divided by the exposure from ITS OWN availability window.
    for source, name in (("saves", "saves_per90"), ("bonus", "bonus_per90"),
                         ("yellow_cards", "yellows_per90"),
                         ("defensive_contribution", "defcon_per90")):
        if source not in out:
            out[name] = 0.0
            continue
        window = next(
            (i for i, cols in enumerate(f.columns for f in frames) if source in cols), 0
        )
        exposure = out[f"exposure_{window}"].replace(0, pd.NA)
        out[name] = (out[source] / exposure).fillna(0.0)
    return out


def forecast_gameweek(
    gw: int, *, market_weight: float = 0.8, planning: bool = False
) -> pd.DataFrame:
    """Expected points for every player in a gameweek, one row per player-fixture.

    `planning=True` allows a FUTURE gameweek to be forecast from the latest known state, which
    is what valuing a squad over a horizon requires — there is no pre-deadline snapshot for a
    gameweek that has not happened. Left False, the strict point-in-time accessor is used and a
    missing snapshot is an error, which is what any evaluation of the past needs.
    """
    pit = PointInTime.for_planning(gw) if planning else PointInTime.for_gameweek(gw)
    players = pit.players()
    players["name"] = _full_name(players)
    players["position"] = players["element_type"].map(POSITION_BY_TYPE)

    teams = read_table("interim", "teams", season="2026-27").set_index("id")["name"]
    players["team_name"] = players["team"].map(teams)

    # --- rates from history
    history = load_history()
    history = history[history["season"] >= RATE_HISTORY_FROM]
    as_of = season_gw_index(history["season"], history["GW"]).max() + 1
    rates = player_rates(history, as_of)

    # --- minutes
    placeholder = pd.DataFrame({
        "name": players["name"], "season": "2026-27", "GW": gw, "minutes": 0.0,
        "position": players["position"], "value": players["now_cost"],
    })
    features = build_features(pd.concat([load_history(), placeholder], ignore_index=True))
    current = features[features["season"] == "2026-27"].reset_index(drop=True)
    model = MinutesModel.load(project_root() / "data" / "processed" / "models" / "minutes.txt")
    minutes = model.predict(current).reset_index(drop=True)
    minutes["name"] = current["name"]
    meta = players.set_index("name").reindex(minutes["name"])
    minutes = apply_availability_gate(
        minutes,
        meta["chance_of_playing_next_round"].reset_index(drop=True),
        status=meta["status"].reset_index(drop=True),
    )
    minutes["name"] = current["name"]

    # --- team-level match forecasts
    fixtures = pit.fixtures()
    fixtures = fixtures[fixtures["event"] == gw].assign(
        home_team=lambda d: d["team_h"].map(teams), away_team=lambda d: d["team_a"].map(teams)
    ).dropna(subset=["home_team", "away_team"])
    match_model = DixonColes().fit(read_table("external", "odds_eval"))
    team_forecasts = build_match_forecasts(fixtures, match_model, market_weight=market_weight)

    # --- combine
    base = players[[
        "id", "web_name", "name", "position", "team_name", "now_cost", "penalties_order",
    ]].rename(columns={"id": "player_id", "team_name": "team"})
    base["price"] = base["now_cost"] / 10.0
    base = base.merge(minutes, on="name", how="left")
    base = base.merge(rates.reset_index(), on="name", how="left")
    base["penalty_share"] = penalty_shares(base)
    # Within position, not across the league — see `backtest/historical_forecast.py` for what
    # the league-wide version did to goalkeepers. New signings and promoted players are the
    # ones this touches, and they are exactly the players a manager is unsure about.
    for col in ("xg_per90", "xa_per90", "finishing_multiplier"):
        within_position = base.groupby("position")[col].transform("median")
        base[col] = base[col].fillna(within_position).fillna(base[col].median())

    forecasts = assemble(base, team_forecasts[[
        "team", "expected_goals_for", "expected_goals_against", "p_clean_sheet",
        "expected_conceded_penalty", "opponent", "is_home",
    ]])
    forecasts["gw"] = gw
    return forecasts
