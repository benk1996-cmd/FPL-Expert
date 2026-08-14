"""Effective ownership — what the field owns, which is what rank is measured against.

Needed because the season target is top 10k, so the optimiser scores points *relative to
the field* (Item 13). Two tiers of data:

*Overall ownership* is free and live in bootstrap-static (`selected_by_percent`). It
describes all ~11M managers, most of whom are not the competition for a top-10k finish.

*Top-10k ownership* has to be assembled: page the Overall league (id 314) for entry IDs,
then pull `entry/{id}/event/{gw}/picks/` for a sample. Sampling 1,000 of 10,000 gives a
standard error near 1.5% at p=0.5 — far tighter than the decisions it feeds — for about ten
minutes of polite requests, so a full census is not worth it.

Two constraints shape the design:

1. **Picks are only exposed after the deadline.** So ownership for the gameweek being
   decided is always a forecast: last gameweek's picks carried forward (squads are >90%
   sticky) and adjusted by transfer flow. Captaincy is the volatile half and the half that
   drives rank, so it needs its own model rather than a carry-forward.

2. **Current-season rank is meaningless early.** After GW1 the "top 10k" is whoever got
   lucky once. Rather than waiting until it stabilises around GW6, we exploit the fact that
   **entry IDs persist across seasons** and `entry/{id}/history/` exposes each past season's
   final rank. Filtering candidates by *prior-season* rank yields a proven-skill cohort from
   GW1 onward, which is what we actually wanted — "managers who are good", not "managers who
   started well".
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import load_config
from .fpl_api import FplApi
from .storage import write_table

log = logging.getLogger(__name__)

SOURCE = "ownership"
PAGE_SIZE = 50  # fixed by the API


def overall_entry_ids(api: FplApi, n: int, league_id: int = 314) -> list[int]:
    """Entry IDs from the top of the Overall league, in rank order."""
    ids: list[int] = []
    for page in range(1, (n // PAGE_SIZE) + 2):
        payload = api.league_standings(league_id, page=page, use_cache=False)
        results = payload.get("standings", {}).get("results", [])
        if not results:
            break
        ids.extend(r["entry"] for r in results)
        if len(ids) >= n or not payload.get("standings", {}).get("has_next"):
            break
    return ids[:n]


def past_ranks(api: FplApi, entry_ids: list[int]) -> pd.DataFrame:
    """Each entry's finishing rank in previous seasons.

    Entry IDs persist across seasons, so this is how a manager's track record is read.
    """
    rows = []
    for entry_id in entry_ids:
        history = api.entry_history(entry_id, use_cache=True)
        if not history:
            continue
        for past in history.get("past", []):
            rows.append(
                {
                    "entry": entry_id,
                    "season": past.get("season_name"),
                    "rank": past.get("rank"),
                    "total_points": past.get("total_points"),
                }
            )
    return pd.DataFrame(rows)


def skilled_cohort(past: pd.DataFrame, max_rank: int = 100_000, min_seasons: int = 1) -> list[int]:
    """Entries that finished inside `max_rank` in at least `min_seasons` past seasons.

    This is the workaround for early-season rank being noise. A manager who finished top
    100k last season is far better evidence of skill than one sitting high after GW1, and
    the cohort is stable rather than churning weekly.
    """
    if past.empty:
        return []
    good = past[past["rank"].notna() & (past["rank"] <= max_rank)]
    counts = good.groupby("entry").size()
    return sorted(counts[counts >= min_seasons].index.tolist())


def sample_entries(entry_ids: list[int], k: int, seed: int = 0) -> list[int]:
    """Uniform sample without replacement; returns everything if the pool is small."""
    if len(entry_ids) <= k:
        return list(entry_ids)
    return (
        pd.Series(entry_ids).sample(n=k, random_state=seed).sort_values().tolist()
    )


def picks_to_frame(entry_id: int, payload: dict) -> pd.DataFrame:
    """Flatten one manager's picks. `multiplier` is 0 bench, 1 starting, 2 captain, 3 TC."""
    picks = payload.get("picks", [])
    return pd.DataFrame(
        {
            "entry": entry_id,
            "element": [p["element"] for p in picks],
            "position": [p["position"] for p in picks],
            "multiplier": [p["multiplier"] for p in picks],
            "is_captain": [p.get("is_captain", False) for p in picks],
            "active_chip": payload.get("active_chip"),
        }
    )


def fetch_picks(api: FplApi, entry_ids: list[int], gw: int) -> pd.DataFrame:
    """Picks for a completed gameweek. Entries that 404 (never set a team) are skipped."""
    frames = []
    for entry_id in entry_ids:
        payload = api.entry_picks(entry_id, gw, use_cache=True)
        if payload:
            frames.append(picks_to_frame(entry_id, payload))
    if not frames:
        return pd.DataFrame(columns=["entry", "element", "position", "multiplier", "is_captain"])
    return pd.concat(frames, ignore_index=True)


def effective_ownership(picks: pd.DataFrame) -> pd.DataFrame:
    """Per-player ownership within the sampled cohort.

    `effective_ownership` is the mean points multiplier, which is the quantity that matters
    for rank: a player owned by 50% and captained by 30% contributes as if owned by 80%,
    because captaincy doubles his score for those managers. Raw ownership alone understates
    the premiums everyone captains — precisely the players where a differential decision has
    the most leverage.
    """
    if picks.empty:
        return pd.DataFrame(columns=["element", "owned_pct", "start_pct", "captain_pct", "eo"])

    n = picks["entry"].nunique()
    grouped = picks.groupby("element")
    out = pd.DataFrame(
        {
            "owned_pct": grouped.size() / n * 100,
            "start_pct": grouped["multiplier"].apply(lambda m: (m >= 1).sum()) / n * 100,
            "captain_pct": grouped["is_captain"].sum() / n * 100,
            "eo": grouped["multiplier"].sum() / n * 100,
        }
    )
    out.index.name = "element"
    return out.reset_index().sort_values("eo", ascending=False)


def ingest_ownership(gw: int, *, max_past_rank: int = 100_000) -> pd.DataFrame:
    """Build the cohort, pull its picks for a completed gameweek, and write ownership.

    Only valid for a gameweek whose deadline has passed — picks 404 before then.
    """
    cfg = load_config()
    api = FplApi()

    candidates = overall_entry_ids(api, cfg.ownership.top_n_managers, cfg.ownership.overall_league_id)
    if not candidates:
        raise RuntimeError("Overall league standings are empty — has the season started?")

    cohort = skilled_cohort(past_ranks(api, candidates), max_rank=max_past_rank)
    if not cohort:
        log.warning("no entries passed the skill filter; falling back to raw standings order")
        cohort = candidates

    sampled = sample_entries(cohort, cfg.ownership.sample_size)
    log.info("cohort %d of %d candidates; sampling %d", len(cohort), len(candidates), len(sampled))

    picks = fetch_picks(api, sampled, gw)
    eo = effective_ownership(picks)
    eo["gw"], eo["n_managers"] = gw, picks["entry"].nunique()
    write_table(eo, "interim", "ownership", gw=gw)
    return eo
