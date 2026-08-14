"""Points assembler: composes every component model into a per-player forecast.

This is where the decomposition pays off. Nothing here learns anything — it applies the
scoring rules from `config/scoring_rules.yaml` to the outputs of the minutes, match,
attacking, secondary and bonus models. Because the rules are configuration rather than
constants, a backtest can score each season under the rules that were actually in force.

The ordering matters and is not arbitrary. Minutes gate everything: a clean sheet needs 60+
minutes, appearance points need any minutes at all, and every rate is scaled by expected
minutes. Team totals come next, then player allocation. Assembling in any other order
produces components that quietly contradict each other.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import load_scoring_rules
from . import bonus as bonus_model
from . import secondary
from .attack import allocate_team_goals, attacking_points

log = logging.getLogger(__name__)

MINUTES_PER_90 = 90.0

COMPONENT_COLUMNS = [
    "pts_appearance", "pts_goals", "pts_assists", "pts_clean_sheet",
    "pts_saves", "pts_defcon", "pts_cards", "pts_bonus",
]


def assemble(
    players: pd.DataFrame,
    team_forecasts: pd.DataFrame,
    rules: dict | None = None,
) -> pd.DataFrame:
    """Expected points per player per fixture.

    `players` needs one row per player-fixture, carrying the minutes-model outputs
    (`p_short`, `p_long`, `expected_minutes`, `expected_appearance_points`), the shrunk
    rates (`xg_per90`, `xa_per90`, `finishing_multiplier`, `saves_per90`, `defcon_per90`,
    `yellows_per90`, `bonus_per90`), `position`, `team`, and `penalty_share`.

    `team_forecasts` is the match model's team-level output, joined on `team` — or on
    `team` AND `fixture` when the caller's player rows are already one per fixture.

    That distinction is load-bearing in a double gameweek. The live path passes one row per
    PLAYER, so joining on team correctly fans a player out to his two fixtures. The backtest
    path reconstructs rows from the archive, which are already one per player-FIXTURE, and
    joining those on team alone is a cross product: two fixtures became four rows, doubling
    the player's expected points in exactly the weeks chips are timed around. Keying on the
    fixture too collapses it back to the right one-to-one join.
    """
    rules = rules or load_scoring_rules()
    keys = ["team"]
    if "fixture" in players.columns and "fixture" in team_forecasts.columns:
        keys.append("fixture")
    df = players.merge(team_forecasts, on=keys, how="left", suffixes=("", "_team"))

    # --- attacking: allocate each team's expected goals across its players, PER FIXTURE.
    # Grouping by team alone is wrong in a double gameweek for a second, subtler reason than
    # the join above: the group then holds two fixtures' worth of rows but only one fixture's
    # expected goals (`iloc[0]`), so that single total is divided among twice as many rows.
    # The effect ran opposite to the duplication — a double gameweek player's attacking
    # returns were credited once rather than twice, while his appearance, clean-sheet and
    # bonus points were credited twice. Netting two errors is not the same as having none.
    allocation_keys = [k for k in ("team", "fixture") if k in df.columns]
    allocated = []
    for _, squad in df.groupby(allocation_keys, dropna=False):
        team_goals = float(squad["expected_goals_for"].iloc[0])
        allocated.append(allocate_team_goals(squad, team_goals))
    df = pd.concat(allocated).reindex(df.index)

    df["pts_appearance"] = df["expected_appearance_points"]
    df["pts_goals_assists"] = attacking_points(
        df["expected_goals"], df["expected_assists"], df["position"],
        goal_points=rules["goal"], assist_points=rules["assist"],
    )
    # Split back out so the report can show where points come from.
    per_goal = df["position"].map(rules["goal"]).astype(float)
    df["pts_goals"] = df["expected_goals"] * per_goal
    df["pts_assists"] = df["expected_assists"] * rules["assist"]

    df["pts_clean_sheet"] = secondary.defensive_points(
        df["p_clean_sheet"], df["p_long"], df["expected_conceded_penalty"],
        df["position"], clean_sheet_points=rules["clean_sheet"],
    )

    saves = secondary.expected_saves(
        df["saves_per90"].fillna(0.0), df["expected_minutes"], df["expected_goals_against"]
    )
    df["expected_saves"] = saves.where(df["position"] == "GK", 0.0)
    df["pts_saves"] = secondary.expected_save_points(
        df["expected_saves"], per_point=rules["saves_per_point"]
    )

    defcon = rules.get("defensive_contribution", {})
    if defcon.get("enabled", False):
        thresholds = {k: v for k, v in defcon["thresholds"].items() if v is not None}
        df["pts_defcon"] = secondary.defensive_contribution_points(
            df["defcon_per90"].fillna(0.0), df["expected_minutes"], df["position"],
            thresholds=thresholds, points=defcon["points"],
            # Bucket probabilities, so the threshold is evaluated conditional on how long the
            # player is actually on the pitch rather than at his average.
            p_short=df["p_short"], p_long=df["p_long"],
        )
    else:
        df["pts_defcon"] = 0.0

    df["pts_cards"] = secondary.expected_card_points(
        df["yellows_per90"].fillna(0.0), df["expected_minutes"],
        yellow_points=rules["yellow_card"], red_points=rules["red_card"],
    )

    # TRIED AND REJECTED: allocating exactly six bonus points per fixture (see
    # `bonus.allocate_match_bonus`). It is more faithful to the rules — only six points
    # exist per match — but it cost 160 season points in simulation (2086 -> 1926). The
    # reason is that normalising within a match destroys cross-match comparability, and the
    # optimiser ranks players ACROSS fixtures: a good player surrounded by other good
    # players was penalised for their presence. Kept independent per player.
    df["pts_bonus"] = bonus_model.expected_bonus(
        df["bonus_per90"].fillna(0.0), df["expected_minutes"],
        df["expected_goals_for"], df["p_clean_sheet"],
    )

    # Coerce explicitly. A single object-dtype component (an all-NA rate column survives as
    # object through the merges) silently makes the total object too, and the failure then
    # surfaces far downstream in the optimiser rather than here.
    for column in COMPONENT_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["expected_points"] = df[COMPONENT_COLUMNS].sum(axis=1)
    df["points_variance"] = _variance(df, rules)
    return df


def _variance(df: pd.DataFrame, rules: dict) -> pd.Series:
    """Approximate variance of a player's score.

    Needed for captaincy and, later, the rank-aware objective — the upside tail is what
    those decisions turn on, and an expected value alone cannot express it. Goals and assists
    are treated as Poisson and the clean sheet as Bernoulli, ignoring covariance between
    them. That understates variance slightly (a clean sheet and a win tend to coincide) and
    is flagged as a SPRINT simplification.
    """
    per_goal = df["position"].map(rules["goal"]).astype(float)
    goal_var = df["expected_goals"] * per_goal**2
    assist_var = df["expected_assists"] * rules["assist"] ** 2
    cs_prob = (df["p_clean_sheet"] * df["p_long"]).clip(0, 1)
    cs_value = df["position"].map(rules["clean_sheet"]).fillna(0).astype(float)
    cs_var = cs_prob * (1 - cs_prob) * cs_value**2
    return goal_var + assist_var + cs_var


def aggregate_gameweek(fixture_level: pd.DataFrame, key: str = "player_id") -> pd.DataFrame:
    """Sum a player's fixtures within a gameweek.

    Double gameweeks are the reason this exists rather than a one-row-per-player forecast:
    a player with two fixtures accumulates from both, which is what makes Bench Boost and
    Triple Captain worth timing.
    """
    numeric = [*COMPONENT_COLUMNS, "expected_points", "points_variance",
               "expected_minutes", "expected_goals", "expected_assists"]
    present = [c for c in numeric if c in fixture_level.columns]
    # `actual_minutes` is carried, NOT summed. Like realised points it is recorded per
    # player-GAMEWEEK already, so adding it across a double gameweek's fixture rows would
    # report a player as having played 180 minutes and break every minutes calibration.
    carry = [c for c in ("web_name", "name", "position", "team", "price", "team_name",
                         "actual_minutes")
             if c in fixture_level.columns]
    # Probabilities must NOT be summed across a double gameweek — P(appears) cannot exceed
    # 1. Carried as the maximum across the player's fixtures instead.
    probabilities = [c for c in ("p_appear", "p_long", "p_short", "p_zero")
                     if c in fixture_level.columns]

    grouped = fixture_level.groupby(key)
    out = grouped[present].sum()
    for col in carry:
        out[col] = grouped[col].first()
    for col in probabilities:
        out[col] = grouped[col].max()
    out["fixtures"] = grouped.size()
    return out.reset_index()


def captain_value(expected_points: pd.Series, points_variance: pd.Series) -> pd.Series:
    """Ranking score for captaincy: the armband doubles both mean and spread.

    Kept separate from `expected_points` because the best captain is not always the highest
    scorer — with a rank-aware objective (Item 13) the variance term earns its own weight.
    """
    return 2 * expected_points + 0.0 * np.sqrt(points_variance)
