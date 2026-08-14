"""Loader for the `vaastav/Fantasy-Premier-League` archive of past seasons.

Gameweek-level player data back to 2016/17. This is the training set — the live API only
exposes the current season, so without this there is nothing to fit on.

Schema drifts across seasons as FPL adds stats (expected goals arrived mid-archive;
`defensive_contribution` and its components only exist from 2025/26). We harmonise to the
union of columns and record what is genuinely missing rather than silently zero-filling —
a model trained on a zero-filled DefCon column would learn that nobody ever tackles.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests

from ..config import load_config
from .http import HttpClient
from .storage import read_table, write_table

log = logging.getLogger(__name__)

ARCHIVE_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
SOURCE = "history"

# Files we take per season. `merged_gw` is the one that matters; the rest give context.
SEASON_FILES = {
    "merged_gw": "gws/merged_gw.csv",
    "fixtures": "fixtures.csv",
    "teams": "teams.csv",
    "players_raw": "players_raw.csv",
}

# Columns that only exist in later seasons — absent means "not recorded", not zero.
LATE_ADDITION_COLUMNS = [
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "starts",
    "clearances_blocks_interceptions", "recoveries", "tackles", "defensive_contribution",
]


def _read_csv(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def fetch_season_file(season: str, key: str, client: HttpClient) -> pd.DataFrame | None:
    """Download one archive file. Returns None if that season doesn't publish it."""
    url = f"{ARCHIVE_BASE}/{season}/{SEASON_FILES[key]}"
    try:
        return _read_csv(client.get_text(url))
    except requests.RequestException as exc:  # 404 for seasons predating a file, or transport
        log.warning("could not fetch %s for %s: %s", key, season, exc)
        return None
    except (pd.errors.ParserError, ValueError) as exc:  # malformed CSV in the archive
        log.warning("could not parse %s for %s: %s", key, season, exc)
        return None


def load_season_gws(season: str, client: HttpClient | None = None) -> pd.DataFrame | None:
    """Per-player, per-gameweek rows for one season, with a `season` column added."""
    client = client or HttpClient(min_interval=0.5)
    df = fetch_season_file(season, "merged_gw", client)
    if df is None:
        return None
    df["season"] = season
    # Some seasons name the gameweek column `round` only.
    if "GW" not in df.columns and "round" in df.columns:
        df["GW"] = df["round"]
    if "kickoff_time" in df.columns:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per season, the share of rows with a real value in each drift-prone column.

    Read this before training: it tells you which seasons can legitimately contribute to
    which models. A season showing 0.0 for `defensive_contribution` must be excluded from
    the DefCon model rather than imputed.
    """
    rows = []
    for season, block in df.groupby("season", sort=True):
        record = {"season": season, "rows": len(block)}
        for col in LATE_ADDITION_COLUMNS:
            record[col] = round(block[col].notna().mean(), 3) if col in block else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def load_history(seasons: list[str] | None = None) -> pd.DataFrame:
    """Read previously ingested seasons back off disk, concatenated.

    Columns are unioned across partitions, so a season that predates a stat reads back as
    NaN for it rather than 0 — `coverage_report` depends on that distinction.
    """
    df = read_table("interim", "history")
    return df[df["season"].isin(seasons)] if seasons else df


def ingest_history(seasons: list[str] | None = None) -> pd.DataFrame:
    """Download seasons, harmonise, and write one parquet partition per season.

    Partitioned by season deliberately: refreshing a single season with
    `fpl history -s 2025-26` must not delete the other six. The coverage report is then
    rebuilt from everything on disk, not just what this run downloaded.
    """
    cfg = load_config()
    seasons = seasons or cfg.data.history_seasons
    client = HttpClient(min_interval=0.5)

    written = 0
    for season in seasons:
        df = load_season_gws(season, client)
        if df is None or df.empty:
            log.warning("no gameweek data for %s — skipping", season)
            continue
        log.info("%s: %d rows, GW %s-%s", season, len(df), df["GW"].min(), df["GW"].max())
        write_table(df, "interim", "history", season=season)
        written += 1

    if not written:
        raise RuntimeError("no seasons downloaded — check connectivity or season names")

    report = coverage_report(load_history())
    write_table(report, "interim", "history_coverage")
    return report
