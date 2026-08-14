"""The serving bundle: small, self-describing artefacts a front end can read without models.

Why this exists
---------------

`forecast_gameweek` loads the 185,000-row archive twice, rebuilds minutes features over all of
it and refits Dixon-Coles — on every call. `fpl squad` then does that once more per horizon
week. That is tens of seconds and hundreds of megabytes of working set, which is fine for a
weekly command and hopeless for a web page that someone opens.

So the front end never computes. `publish` runs the pipeline once, writes a few megabytes of
parquet plus a manifest, and the UI reads that. The split also keeps the point-in-time
discipline honest: the bundle records the gameweek it was built FOR and the moment it was
built AT, so a stale page is visibly stale rather than quietly wrong.

What is deliberately NOT in here
--------------------------------

Anything requiring the user's entry id. Transfer plans and held-squad chip advice depend on the
squad you own, which is a live lookup and personal data; `fpl myteam --brief` remains the place
for that. The bundle describes the game, not your team, so it is safe to commit and to publish.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Columns worth serving. The forecast frame carries ~60, most of them intermediate quantities
# that would triple the bundle and explain nothing to a reader.
PLAYER_COLUMNS = [
    "player_id", "web_name", "name", "position", "team_name", "team", "price",
    "expected_points", "horizon_points", "points_variance",
    "p_appear", "p_long", "expected_minutes",
    "pts_appearance", "pts_goals", "pts_assists", "pts_clean_sheet",
    "pts_bonus", "pts_defcon", "pts_saves", "pts_cards",
    "expected_goals", "expected_assists", "selected_by_percent", "status", "news",
]


def _present(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Intersect, loudly. A silently missing column here becomes a missing UI section."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        log.info("serving bundle omits absent columns: %s", missing)
    return [c for c in columns if c in frame.columns]


def fixture_grid(fixtures: pd.DataFrame, teams: pd.Series, gw: int, span: int) -> pd.DataFrame:
    """One row per team-gameweek over the window: who they play, home or away.

    Long rather than wide, because a team can have two fixtures in a gameweek or none, and a
    wide grid has nowhere to put either. The UI pivots it and handles the duplicates.
    """
    window = fixtures[fixtures["event"].between(gw, gw + span - 1)].copy()
    if window.empty:
        return pd.DataFrame(
            columns=["team", "gw", "opponent", "is_home", "kickoff_time"]
        )

    window["home_team"] = window["team_h"].map(teams)
    window["away_team"] = window["team_a"].map(teams)
    home = window.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(
        is_home=True
    )
    away = window.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(
        is_home=False
    )
    grid = pd.concat([home, away], ignore_index=True)
    return grid[["team", "event", "opponent", "is_home", "kickoff_time"]].rename(
        columns={"event": "gw"}
    ).sort_values(["team", "gw"]).reset_index(drop=True)


def write_bundle(
    directory: Path | str,
    *,
    gw: int,
    span: int,
    players: pd.DataFrame,
    solution,
    brief: str,
    fixtures: pd.DataFrame | None = None,
    risers: pd.DataFrame | None = None,
    fallers: pd.DataFrame | None = None,
    points_col: str = "horizon_points",
) -> Path:
    """Write everything a front end needs, and a manifest saying what it is.

    Overwrites in place. The bundle is a snapshot of one gameweek's advice, not an accumulating
    archive — history lives in `data/raw/`, which is the record that cannot be rebuilt.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    served = players[_present(players, PLAYER_COLUMNS)].copy()
    starters = set(solution.starting_xi["player_id"])
    squad_ids = set(solution.squad["player_id"])
    captain = solution.captain.get("player_id") if hasattr(solution.captain, "get") else None
    vice = (
        solution.vice_captain.get("player_id")
        if hasattr(solution.vice_captain, "get") else None
    )
    served["in_squad"] = served["player_id"].isin(squad_ids)
    served["is_starter"] = served["player_id"].isin(starters)
    served["is_captain"] = served["player_id"] == captain
    served["is_vice"] = served["player_id"] == vice
    served.to_parquet(out / "players.parquet", index=False)

    if fixtures is not None and not fixtures.empty:
        fixtures.to_parquet(out / "fixtures.parquet", index=False)

    moves = []
    for label, frame in (("rise", risers), ("fall", fallers)):
        if frame is not None and not frame.empty:
            block = frame[_present(frame, [
                "player_id", "web_name", "team_name", "price",
                "expected_change", "p_rise", "p_fall",
            ])].copy()
            block["direction"] = label
            moves.append(block)
    if moves:
        pd.concat(moves, ignore_index=True).to_parquet(out / "prices.parquet", index=False)

    (out / "brief.md").write_text(brief, encoding="utf-8")

    manifest = {
        "gameweek": int(gw),
        "horizon": int(span),
        "points_col": points_col,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "players": len(served),
        "squad_cost": round(float(solution.total_cost), 1),
        # This gameweek's XI plus armband — NOT the horizon objective the squad was chosen on.
        # Conflating the two is exactly how `fpl squad` once reported 221 points for one week.
        "expected_points": round(float(solution.expected_points), 2),
        "captain": str(solution.captain.get("web_name", "")) if captain else "",
        "vice_captain": str(solution.vice_captain.get("web_name", "")) if vice else "",
        "has_prices": bool(moves),
        "has_fixtures": fixtures is not None and not fixtures.empty,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("serving bundle written to %s (%d players)", out, len(served))
    return out


def read_bundle(directory: Path | str) -> dict:
    """Load a bundle. Returns the manifest plus whatever optional pieces are present.

    Raises rather than returning empty frames when the bundle is missing: a front end showing
    a blank page is far worse than one showing "run `fpl publish` first".
    """
    out = Path(directory)
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no serving bundle at {out} — run `fpl publish --gw N` to build one"
        )

    bundle = {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "players": pd.read_parquet(out / "players.parquet"),
        "brief": (out / "brief.md").read_text(encoding="utf-8")
        if (out / "brief.md").exists() else "",
    }
    for name in ("fixtures", "prices"):
        path = out / f"{name}.parquet"
        bundle[name] = pd.read_parquet(path) if path.exists() else None
    return bundle
