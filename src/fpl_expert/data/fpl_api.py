"""Client for the official Fantasy Premier League API.

All endpoints below are public and unauthenticated. Verified working 2026-08-09.

    bootstrap-static/                     players, teams, events, scoring config
    fixtures/[?event=N]                   fixture list, difficulty, kickoff times
    element-summary/{player_id}/          a player's per-fixture history + future fixtures
    event/{gw}/live/                      live per-player stats for a gameweek
    entry/{id}/ , entry/{id}/history/     a manager's profile and season history
    entry/{id}/event/{gw}/picks/          a manager's 15 picks  (404s BEFORE the deadline)
    leagues-classic/{id}/standings/       paged standings, 50 per page (Overall = 314)

The picks 404 is the important one: the field's teams for the gameweek you are deciding
are never visible in time, which is why effective ownership has to be forecast (Item 2b).
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import load_config
from .http import HttpClient
from .storage import write_raw, write_table

log = logging.getLogger(__name__)

SOURCE = "fpl_api"

# `now_cost` and friends are in tenths of a million.
PRICE_DIVISOR = 10.0

POSITION_BY_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class FplApi:
    """Thin wrapper over the FPL endpoints, returning parsed JSON."""

    def __init__(self, client: HttpClient | None = None) -> None:
        cfg = load_config()
        self.base = cfg.data.fpl_api_base.rstrip("/")
        self.client = client or HttpClient(
            cache_dir=cfg.path("raw") / ".cache",
            cache_ttl_hours=cfg.data.cache_ttl_hours,
            min_interval=cfg.data.request_delay_seconds,
        )

    def _get(self, endpoint: str, **kwargs):
        return self.client.get_json(f"{self.base}/{endpoint}", **kwargs)

    # --- core static feeds -------------------------------------------------

    def bootstrap_static(self, *, use_cache: bool = True) -> dict:
        return self._get("bootstrap-static/", use_cache=use_cache)

    def fixtures(self, event: int | None = None, *, use_cache: bool = True) -> list[dict]:
        endpoint = "fixtures/" if event is None else f"fixtures/?event={event}"
        return self._get(endpoint, use_cache=use_cache)

    def element_summary(self, player_id: int, *, use_cache: bool = True) -> dict:
        return self._get(f"element-summary/{player_id}/", use_cache=use_cache)

    def event_live(self, gw: int, *, use_cache: bool = True) -> dict | None:
        return self._get(f"event/{gw}/live/", use_cache=use_cache, allow_404=True)

    # --- manager / league feeds -------------------------------------------

    def entry(self, entry_id: int, *, use_cache: bool = True) -> dict | None:
        return self._get(f"entry/{entry_id}/", use_cache=use_cache, allow_404=True)

    def entry_history(self, entry_id: int, *, use_cache: bool = True) -> dict | None:
        return self._get(f"entry/{entry_id}/history/", use_cache=use_cache, allow_404=True)

    def entry_picks(self, entry_id: int, gw: int, *, use_cache: bool = True) -> dict | None:
        """A manager's picks. Returns None before the gameweek deadline (endpoint 404s)."""
        return self._get(
            f"entry/{entry_id}/event/{gw}/picks/", use_cache=use_cache, allow_404=True
        )

    def league_standings(self, league_id: int, page: int = 1, *, use_cache: bool = True) -> dict:
        return self._get(
            f"leagues-classic/{league_id}/standings/?page_standings={page}", use_cache=use_cache
        )


# --- parsers: JSON -> tidy frames -----------------------------------------


def parse_players(bootstrap: dict) -> pd.DataFrame:
    """One row per player, with prices in £m and position labels attached."""
    df = pd.DataFrame(bootstrap["elements"])
    df["position"] = df["element_type"].map(POSITION_BY_TYPE)
    for col in ("now_cost", "cost_change_start", "cost_change_event"):
        if col in df:
            df[col.replace("now_cost", "price")] = df[col] / PRICE_DIVISOR
    # These arrive as strings; they are numeric everywhere downstream.
    numeric = [
        "selected_by_percent", "form", "points_per_game", "value_form", "value_season",
        "expected_goals", "expected_assists", "expected_goal_involvements",
        "expected_goals_conceded", "expected_goals_per_90", "expected_assists_per_90",
        "influence", "creativity", "threat", "ict_index",
    ]
    for col in numeric:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_teams(bootstrap: dict) -> pd.DataFrame:
    return pd.DataFrame(bootstrap["teams"])


def parse_events(bootstrap: dict) -> pd.DataFrame:
    """Gameweeks, with deadlines parsed to tz-aware UTC timestamps."""
    df = pd.DataFrame(bootstrap["events"])
    df["deadline_time"] = pd.to_datetime(df["deadline_time"], utc=True)
    return df


def parse_fixtures(fixtures: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(fixtures)
    if "kickoff_time" in df:
        df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df


def next_gameweek(bootstrap: dict) -> int | None:
    """The gameweek currently being decided, i.e. the next one with an open deadline."""
    for event in bootstrap["events"]:
        if event.get("is_next"):
            return event["id"]
    # Past the final deadline of the season.
    unfinished = [e["id"] for e in bootstrap["events"] if not e.get("finished")]
    return min(unfinished) if unfinished else None


def current_gameweek(bootstrap: dict) -> int | None:
    """The gameweek in progress or most recently completed; None before the season starts."""
    for event in bootstrap["events"]:
        if event.get("is_current"):
            return event["id"]
    return None


# --- ingestion ------------------------------------------------------------


def ingest_core(api: FplApi | None = None, *, stamp: str | None = None) -> dict[str, int]:
    """Pull bootstrap + fixtures, archive the raw JSON, and write typed parquet.

    Idempotent in the sense that re-running is harmless: raw gains a new timestamped
    partition (deliberately — that is the point-in-time record) and interim is rebuilt.
    """
    api = api or FplApi()
    bootstrap = api.bootstrap_static(use_cache=False)
    fixtures = api.fixtures(use_cache=False)

    write_raw(bootstrap, SOURCE, "bootstrap", stamp=stamp)
    write_raw(fixtures, SOURCE, "fixtures", stamp=stamp)

    tables = {
        "players": parse_players(bootstrap),
        "teams": parse_teams(bootstrap),
        "events": parse_events(bootstrap),
        "fixtures": parse_fixtures(fixtures),
    }
    season = str(load_config().project.get("season", "unknown")).replace("/", "-")
    for name, df in tables.items():
        write_table(df, "interim", name, season=season)

    counts = {name: len(df) for name, df in tables.items()}
    log.info("ingested core feeds: %s", counts)
    return counts
