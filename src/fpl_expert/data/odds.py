"""Bookmaker odds from football-data.co.uk — free, no key, no rate limit published.

    mmz4281/{code}/E0.csv   one season of Premier League results + odds  (code: 2526 = 2025-26)
    fixtures.csv            upcoming fixtures across leagues, with current odds

Why odds are worth the trouble: the closing line is the single strongest freely available
estimate of team strength. It aggregates injury news, rotation intelligence and market
money, and beating it consistently is hard. Item 6 uses it as a prior on the match model
rather than trying to out-predict it from scratch.

Two subtleties handled here:

*Closing vs opening.* Columns suffixed `C` (PSCH, AvgCH, ...) are CLOSING odds, priced
immediately before kick-off. Non-suffixed columns are opening odds, set days earlier. Only
closing odds embed late team news — but for a point-in-time backtest that is precisely the
problem, because we decide at the FPL deadline, which falls BEFORE the closing line exists.
Both are kept and labelled; Item 3 decides which is legitimate for a given use.

*Overround.* Quoted odds imply probabilities summing to >1 (the bookmaker's margin). They
must be de-vigged before use or every probability is biased upward.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import requests

from ..config import load_config
from .http import HttpClient
from .storage import write_raw, write_table
from .teams import normalise_series

log = logging.getLogger(__name__)

BASE = "https://www.football-data.co.uk"
SOURCE = "odds"
DIVISION = "E0"  # Premier League

# Bookmaker preference, sharpest first. Pinnacle runs the lowest margin and is the usual
# benchmark for a "fair" line; the market average is a robust fallback; Bet365 is last
# because it is a single soft book.
CLOSING_PREFERENCE = [("PSC", "Pinnacle"), ("AvgC", "market average"), ("B365C", "Bet365")]
OPENING_PREFERENCE = [("PS", "Pinnacle"), ("Avg", "market average"), ("B365", "Bet365")]


def season_code(season: str) -> str:
    """'2025-26' -> '2526', the code football-data uses in its paths."""
    start, end = season.split("-")
    return f"{start[-2:]}{end[-2:]}"


def _read_csv(raw: bytes) -> pd.DataFrame:
    """Parse a football-data CSV, coping with its inconsistent encodings.

    The season files are latin-1. `fixtures.csv` is UTF-8 with a BOM — decoding that as
    latin-1 leaves 'ï»¿' welded to the first header, so `Div` silently goes missing and the
    league filter matches nothing. Try the stricter codec first; only it can fail cleanly.
    """
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(subset=["HomeTeam", "AwayTeam"])


def fetch_season(season: str, client: HttpClient) -> pd.DataFrame | None:
    """One season of results and odds. Returns None if that season isn't published."""
    url = f"{BASE}/mmz4281/{season_code(season)}/{DIVISION}.csv"
    try:
        return _read_csv(client.session.get(url, timeout=60).content)
    except requests.RequestException as exc:
        log.warning("could not fetch odds for %s: %s", season, exc)
        return None
    except (pd.errors.ParserError, KeyError, ValueError) as exc:
        log.warning("could not parse odds for %s: %s", season, exc)
        return None


def fetch_upcoming(client: HttpClient) -> pd.DataFrame:
    """Upcoming fixtures with current odds. Pre-deadline this is the only forward-looking
    source; it carries opening-style odds only (no closing columns exist yet)."""
    df = _read_csv(client.session.get(f"{BASE}/fixtures.csv", timeout=60).content)
    return df[df["Div"] == DIVISION].copy()


def devig_proportional(odds: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds to probabilities and remove the bookmaker margin.

    Proportional (multiplicative) de-vigging: divide each raw implied probability by the
    overround. It is the standard first approximation. It slightly over-corrects
    longshots — bookmakers load more margin onto unlikely outcomes — so Item 6 may want
    Shin's method instead. Kept simple and explicit here so the bias is visible rather
    than buried.
    """
    implied = 1.0 / odds
    # min_count forces a row with ANY missing price to yield NaN. Without it pandas sums
    # all-NaN rows to 0.0, which silently produces an overround of zero (an impossible
    # free arbitrage) and NaN probabilities that look like ordinary missing data.
    overround = implied.sum(axis=1, min_count=implied.shape[1])
    out = implied.div(overround, axis=0)
    out["overround"] = overround
    return out


def _coverage(df: pd.DataFrame, prefix: str) -> float:
    cols = [f"{prefix}{s}" for s in ("H", "D", "A")]
    if not all(c in df.columns for c in cols):
        return 0.0
    return float(df[cols].notna().all(axis=1).mean())


def _pick_source(
    df: pd.DataFrame, preference: list[tuple[str, str]], min_coverage: float = 0.95
) -> tuple[str, str, float] | None:
    """Sharpest bookmaker that actually prices most of the season.

    Preference order alone is not enough: Pinnacle is the sharpest book but football-data
    only carries it for part of some seasons (210 of 380 matches in 2025-26). Taking it
    regardless would throw away 45% of the fixtures, so we fall back to a book with real
    coverage and record which one was used.
    """
    scored = [(prefix, label, _coverage(df, prefix)) for prefix, label in preference]
    for prefix, label, cov in scored:
        if cov >= min_coverage:
            return prefix, label, cov
    best = max(scored, key=lambda s: s[2])
    return best if best[2] > 0 else None


def normalise(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Tidy one football-data frame: FPL team names, real dates, de-vigged probabilities."""
    out = pd.DataFrame(
        {
            "season": season,
            "date": pd.to_datetime(df["Date"], dayfirst=True, errors="coerce"),
            "home_team": normalise_series(df["HomeTeam"]),
            "away_team": normalise_series(df["AwayTeam"]),
        }
    )
    if "Time" in df.columns:
        out["time"] = df["Time"]
    for src, dst in (("FTHG", "home_goals"), ("FTAG", "away_goals")):
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")

    for preference, tag in ((CLOSING_PREFERENCE, "close"), (OPENING_PREFERENCE, "open")):
        picked = _pick_source(df, preference)
        if picked is None:
            continue
        prefix, label, cov = picked
        if cov < 0.95:
            log.warning("%s %s odds: best book is %s at %.0f%% coverage", season, tag, label, cov * 100)
        odds = df[[f"{prefix}{s}" for s in ("H", "D", "A")]].apply(pd.to_numeric, errors="coerce")
        probs = devig_proportional(odds.set_axis(["home", "draw", "away"], axis=1))
        for outcome in ("home", "draw", "away"):
            out[f"p_{outcome}_{tag}"] = probs[outcome]
        out[f"overround_{tag}"] = probs["overround"]
        out[f"book_{tag}"] = label

    # Over/under 2.5 pins the expected total goals, which 1X2 alone cannot: it separates
    # "close match between two good attacks" from "close match between two bad ones".
    for prefix, tag in (("AvgC", "close"), ("Avg", "open")):
        over, under = f"{prefix}>2.5", f"{prefix}<2.5"
        if over in df.columns and under in df.columns:
            ou = df[[over, under]].apply(pd.to_numeric, errors="coerce")
            probs = devig_proportional(ou.set_axis(["over", "under"], axis=1))
            out[f"p_over25_{tag}"] = probs["over"]

    return out


ID_COLUMNS = ["season", "date", "time", "home_team", "away_team"]


def split_admissible(tidy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate pre-deadline features from post-deadline evaluation data.

    The split is physical rather than conventional: the two halves go to different tables,
    so a feature-building join cannot accidentally drag a closing-odds column along. See
    `snapshot.PointInTime`, which can reach `odds_features` and has no route to
    `odds_eval` at all.

    Opening odds are admissible — set days ahead of the deadline. Closing odds and final
    scores are not: both postdate the moment the squad was locked.
    """
    ids = [c for c in ID_COLUMNS if c in tidy.columns]
    feature_cols = [c for c in tidy.columns if c.endswith("_open")]
    eval_cols = [c for c in tidy.columns if c.endswith("_close")]
    eval_cols += [c for c in ("home_goals", "away_goals") if c in tidy.columns]
    return tidy[ids + feature_cols].copy(), tidy[ids + eval_cols].copy()


def ingest_odds(seasons: list[str] | None = None, *, include_upcoming: bool = True) -> pd.DataFrame:
    """Download odds for each season plus upcoming fixtures; write parquet per season."""
    cfg = load_config()
    seasons = seasons or cfg.data.history_seasons
    client = HttpClient(min_interval=1.0)

    summary = []
    for season in seasons:
        raw = fetch_season(season, client)
        if raw is None or raw.empty:
            continue
        tidy = normalise(raw, season)
        features, evaluation = split_admissible(tidy)
        write_table(features, "external", "odds_features", season=season)
        write_table(evaluation, "external", "odds_eval", season=season)
        closing = tidy.get("p_home_close", pd.Series(dtype=float))
        overround = tidy.get("overround_close", pd.Series([np.nan]))
        summary.append(
            {
                "season": season,
                "matches": len(tidy),
                "with_closing": int(closing.notna().sum()),
                "book": tidy["book_close"].iloc[0] if "book_close" in tidy else "-",
                "mean_overround": round(float(overround.mean()), 4),
            }
        )
        log.info("%s: %d matches", season, len(tidy))

    if include_upcoming:
        try:
            upcoming = normalise(fetch_upcoming(client), str(cfg.project.get("season", "")))
            write_raw(upcoming.to_dict("records"), SOURCE, "upcoming")
            write_table(split_admissible(upcoming)[0], "external", "odds_upcoming")
            log.info("upcoming fixtures with odds: %d", len(upcoming))
        except (requests.RequestException, pd.errors.ParserError, KeyError, ValueError) as exc:
            log.warning("could not fetch upcoming odds: %s", exc)

    if not summary:
        raise RuntimeError("no odds downloaded — check connectivity")
    return pd.DataFrame(summary)
