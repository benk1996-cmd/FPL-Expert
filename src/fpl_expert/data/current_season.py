"""The season in progress, written into the archive the models actually train on.

The gap this closes
-------------------

`fpl history` downloads COMPLETED seasons from the community archive; `fpl update` pulls the
bootstrap, which is current *state* and carries no per-gameweek results. So nothing wrote a
`season=2026-27` partition, and `load_history()` stopped at last May — every rate in every
forecast was built from prior seasons alone. A striker's first month at a new club counted for
nothing, and the gap widened by one gameweek a week.

Why `element-summary` rather than `event/{gw}/live`
---------------------------------------------------

`event/{gw}/live` is one call per gameweek instead of one per player, but it aggregates a
double gameweek into a single entry and exposes per-fixture detail only for scoring
identifiers — no per-fixture expected goals. `element-summary/{id}/history` returns one row per
FIXTURE with the full stat line including xG, plus `value` (the player's price at the time) and
`kickoff_time`, which the minutes model needs for rest days. It matches the archive's shape
almost exactly, so the two sources concatenate without harmonisation.

The cost is ~610 polite requests, a few minutes, once a week after the deadline.

Only settled gameweeks
----------------------

A gameweek is ingested only when FPL reports `finished` AND `data_checked`. Before that, bonus
points are provisional and fixtures may be outstanding — and a half-finished gameweek written
into the archive would be indistinguishable from a real one, quietly telling every rate that a
striker played 0 minutes that week.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import load_config
from .fpl_api import POSITION_BY_TYPE, FplApi, parse_events
from .storage import write_raw, write_table

log = logging.getLogger(__name__)

SOURCE = "current_season"

# Columns the archive carries that a live per-fixture row does not. Left ABSENT rather than
# zero-filled, exactly as `historical.py` treats stats that predate their own introduction.
ARCHIVE_ONLY = ("xP",)

# The API returns these as STRINGS ("0.63"), while the archive stores float64. Concatenating
# the two silently produced an object column, and the first arithmetic on it failed with
# "can't multiply sequence by non-int of type 'float'" — deep inside the rate calculation,
# nowhere near the cause. `expected_goals` and `expected_assists` are in this list, so the two
# most important rate inputs are exactly the ones that would have broken.
STRING_TYPED_NUMERICS = (
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat", "ict_index",
)


def settled_gameweeks(bootstrap: dict) -> list[int]:
    """Gameweeks whose results are final: played, checked, and bonus awarded.

    `finished` alone is not enough — it flips before `data_checked`, while bonus is still
    provisional. Ingesting then would bake provisional bonus into the rates permanently.
    """
    events = parse_events(bootstrap)
    if "data_checked" not in events:
        return []
    settled = events[events["finished"].fillna(False) & events["data_checked"].fillna(False)]
    return sorted(int(g) for g in settled["id"])


def player_identity(bootstrap: dict) -> pd.DataFrame:
    """`element` -> the name, position and team spelling the archive uses.

    `name` must be `first_name + " " + second_name`, which is how the community archive spells
    it and how `pipeline._full_name` rebuilds it from the bootstrap. If this drifts, the rate
    join silently misses and every affected player falls back to a league prior.
    """
    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"]).set_index("id")["name"]
    return pd.DataFrame({
        "element": players["id"].astype(int),
        "name": (
            players["first_name"].fillna("") + " " + players["second_name"].fillna("")
        ).str.strip(),
        "position": players["element_type"].map(POSITION_BY_TYPE),
        "team": players["team"].map(teams),
    })


def fetch_player_history(api: FplApi, element: int) -> pd.DataFrame:
    """One row per fixture this player has featured in, or an empty frame."""
    payload = api.element_summary(element, use_cache=False)
    rows = (payload or {}).get("history") or []
    return pd.DataFrame(rows)


def build_current_season(
    api: FplApi | None = None, *, season: str | None = None, gameweeks: list[int] | None = None
) -> pd.DataFrame:
    """Assemble archive-shaped rows for every settled gameweek so far."""
    api = api or FplApi()
    cfg = load_config()
    season = season or str(cfg.project.get("season", "")).replace("/", "-")

    bootstrap = api.bootstrap_static(use_cache=False)
    settled = gameweeks if gameweeks is not None else settled_gameweeks(bootstrap)
    if not settled:
        raise RuntimeError(
            "no settled gameweeks yet — a gameweek is ingested only once FPL reports it "
            "finished AND data_checked, so bonus is final"
        )
    log.info("settled gameweeks: %s", settled)

    identity = player_identity(bootstrap)
    frames = []
    for element in identity["element"]:
        block = fetch_player_history(api, int(element))
        if not block.empty:
            frames.append(block)
    if not frames:
        raise RuntimeError("no player history returned — has the season started?")

    df = pd.concat(frames, ignore_index=True)
    df["GW"] = df["round"].astype(int)
    df = df[df["GW"].isin(settled)]
    if df.empty:
        raise RuntimeError(f"no rows for settled gameweeks {settled}")

    df["season"] = season
    df = df.merge(identity, on="element", how="left")
    if df["name"].isna().any():
        missing = int(df["name"].isna().sum())
        log.warning("%d rows have no bootstrap identity and will not join to rates", missing)

    if "kickoff_time" in df:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    for col in STRING_TYPED_NUMERICS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ARCHIVE_ONLY:
        if col not in df:
            df[col] = float("nan")      # float, not pd.NA: the archive holds these as float64

    log.info(
        "%d rows, %d players, GW %s-%s", len(df), df["element"].nunique(),
        df["GW"].min(), df["GW"].max(),
    )
    return df


def ingest_current_season(
    api: FplApi | None = None, *, season: str | None = None, gameweeks: list[int] | None = None
) -> pd.DataFrame:
    """Build and write the current season's partition of `interim/history`.

    Overwrites that one partition and leaves the others alone — the same discipline as
    `ingest_history`, and what lets this be re-run after every deadline.
    """
    df = build_current_season(api, season=season, gameweeks=gameweeks)
    season = df["season"].iloc[0]
    write_raw(df.to_dict("records"), SOURCE, "history", stamp=None)
    write_table(df, "interim", "history", season=season)
    log.info("wrote interim/history season=%s (%d rows)", season, len(df))
    return df


def schema_mismatches(new: pd.DataFrame, archive: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Columns whose dtype differs from the archive's, as (column, archive, new).

    The partitions are concatenated by `load_history`, so a mismatch here does not fail on
    write — it produces an object column that breaks arithmetic much later, in a traceback
    that points at pandas rather than at the ingestion that caused it.
    """
    shared = sorted(set(new.columns) & set(archive.columns))
    return [
        (c, str(archive[c].dtype), str(new[c].dtype))
        for c in shared
        if str(new[c].dtype) != str(archive[c].dtype)
    ]
