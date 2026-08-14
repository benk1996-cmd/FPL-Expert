"""Player attacking returns: goals and assists, allocated from the team's expected goals.

Built top-down rather than bottom-up. A player's raw per-90 scoring rate says nothing about
*this* fixture; the match model (Item 6) already knows how many goals the team is expected
to score against this opponent. So the job here is allocation: split the team's expected
goals among its players in proportion to their shrunk rates and expected minutes.

Two properties follow for free, and both would have to be engineered if we went bottom-up:

1. **Team totals stay consistent with the match model.** The sum of players' expected goals
   equals the team's expected goals, so clean sheets and attacking returns cannot contradict
   each other.
2. **Fixture difficulty is inherited automatically.** Nobody has to build a separate
   "opponent strength" feature for players — it is already priced into the team total.

Finishing skill is modelled as a per-player multiplier on xG, shrunk hard toward the league
ratio. Shrinkage is doing real work here: over 2022-23 onward the league converts xG at 1.054,
but individual players' raw ratios are mostly noise, and an unshrunk multiplier would simply
chase whoever ran hot.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..features.rates import NOISE_SCALE, fit_gamma_prior, shrink

log = logging.getLogger(__name__)

# Share of a team's penalties taken by the 1st, 2nd, 3rd designated taker. Rough but far
# better than ignoring duty: the first-choice taker takes the overwhelming majority.
# SPRINT: estimated from typical usage rather than fitted, because the archive has no
# set-piece order column to fit against.
PENALTY_ORDER_SHARE = {1: 0.85, 2: 0.11, 3: 0.03}

# League-average penalties awarded per team per match, and conversion rate.
PENALTIES_PER_TEAM_MATCH = 0.11
PENALTY_CONVERSION = 0.79

# FPL assists per FPL goal. Not every goal is assisted (solo goals, rebounds, penalties), so
# team expected assists is less than team expected goals — but by far less than the 0.72 this
# previously assumed. Measured over all seven archived seasons, and remarkably stable:
#
#     2019-20  0.900   2020-21  0.918   2021-22  0.896   2022-23  0.893
#     2023-24  0.896   2024-25  0.905   2025-26  0.934        overall  0.905
#
# The old 0.72 was an unmeasured estimate and it was the entire cause of the 27% assist
# under-prediction recorded in `models/distribution.py`: 0.72 x 1.268 = 0.913. Note this is
# FPL's assist definition, which is what we forecast — broader than a strict final pass, since
# FPL also credits winning a penalty and some deflected balls.
ASSIST_RATE = 0.905

MINUTES_PER_90 = 90.0


def player_totals_from_history(
    history: pd.DataFrame, *, as_of: float, half_life_gws: float = 20.0, key: str = "name"
) -> pd.DataFrame:
    """Decayed xG, xA, goals and exposure per player, ready for `attacking_rates`.

    Keyed by name rather than `element` for the same reason as the minutes model: FPL
    reassigns element ids each season.
    """
    from ..features.rates import decayed_totals

    totals = decayed_totals(
        history,
        as_of=as_of,
        stats=["expected_goals", "expected_assists", "goals_scored", "assists"],
        half_life_gws=half_life_gws,
        key=key,
    )
    return totals.rename(
        columns={
            "_exposure": "exposure_90s",
            "_c_expected_goals": "xg",
            "_c_expected_assists": "xa",
            "_c_goals_scored": "goals",
            "_c_assists": "assist_count",
        }
    )


def attacking_rates(
    player_totals: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("position",),
) -> pd.DataFrame:
    """Shrunk per-90 xG and xA rates, plus a shrunk finishing multiplier.

    `player_totals` must carry decayed, exposure-weighted totals per player:
    `exposure_90s`, `xg`, `xa`, `goals`. Producing those is the job of
    `features.rates.rate_features`; this function only does the shrinkage and the
    finishing ratio.
    """
    df = player_totals.copy()
    out = pd.DataFrame(index=df.index)

    groups = [c for c in group_cols if c in df.columns]
    if group_cols and not groups:
        # Loudly, because the fallback is not a harmless default: without a grouping column
        # every player shrinks toward ONE league-wide prior, which drags goalkeepers toward an
        # outfielder's attacking rate. That went unnoticed for weeks and was worth 79 phantom
        # goalkeeper goals at 10 points each.
        log.warning(
            "none of %s present in player_totals — shrinking every player toward a single "
            "league-wide prior. Goalkeepers will be given outfield attacking rates.",
            list(group_cols),
        )
    # xG and xA are sums of probabilities, not counts, so they carry far less sampling noise
    # than Poisson would imply — measured at 0.29 and 0.20 of it. Passing the wrong scale
    # here collapses every prior onto its floor.
    for stat, noise in (("xg", NOISE_SCALE["expected_goals"]),
                        ("xa", NOISE_SCALE["expected_assists"])):
        pieces = []
        for _, block in (df.groupby(groups, dropna=False) if groups else [(None, df)]):
            prior = fit_gamma_prior(block[stat], block["exposure_90s"], noise)
            pieces.append(shrink(block[stat], block["exposure_90s"], prior))
        shrunk = pd.concat(pieces).reindex(df.index)
        out[f"{stat}_per90"] = shrunk["rate"]
        out[f"{stat}_confidence"] = shrunk["confidence"]

    # Finishing: goals as the count, xG as the exposure. The posterior mean is then exactly
    # "goals per unit of xG", shrunk toward the population ratio. A player with little xG
    # accumulated lands on the league rate rather than on his own noisy ratio. Goals ARE a
    # count, so the full Poisson noise scale applies here.
    pieces = []
    for name, block in (df.groupby(groups, dropna=False) if groups else [(None, df)]):
        prior = fit_gamma_prior(block["goals"], block["xg"], NOISE_SCALE["goals_scored"])
        if not prior.skill_detected:
            # Not a failure: the spread in goals-per-xG is fully explained by chance, so
            # every player is correctly estimated at the group conversion rate. Logged
            # because a collapsed prior and a real signal look identical downstream.
            log.info("no finishing skill detectable for %s — using group rate %.3f",
                     name, prior.mean)
        pieces.append(shrink(block["goals"], block["xg"], prior))
    finishing = pd.concat(pieces).reindex(df.index)
    out["finishing_multiplier"] = finishing["rate"]
    out["finishing_confidence"] = finishing["confidence"]
    return out


def penalty_shares(players: pd.DataFrame, order_col: str = "penalties_order") -> pd.Series:
    """Share of the team's penalties each player is expected to take.

    Read from the live API at inference. The archive has no set-piece order column, so this
    cannot be a training feature — the same train/serve split as the availability gate.
    Players with no listed order take no penalties, which is very nearly true.
    """
    if order_col not in players.columns:
        return pd.Series(0.0, index=players.index)
    order = pd.to_numeric(players[order_col], errors="coerce")
    return order.map(PENALTY_ORDER_SHARE).fillna(0.0)


def allocate_team_goals(
    players: pd.DataFrame,
    team_expected_goals: float,
    *,
    penalty_share_col: str = "penalty_share",
) -> pd.DataFrame:
    """Split one team's expected goals for one fixture among its players.

    Open play is allocated in proportion to each player's expected xG contribution
    (rate x expected minutes x finishing skill), rescaled so the total matches the match
    model. Penalties are handled separately because they are lumpy, assignable, and
    concentrated on one player — averaging them across the squad would systematically
    underrate designated takers, who are often the best value in the game.
    """
    df = players.copy()
    expected_90s = df["expected_minutes"] / MINUTES_PER_90

    # Expected penalties for this team in this fixture, scaled by how strong its attack is.
    league_goals = 1.43   # roughly the per-team average; only the ratio matters
    team_penalties = PENALTIES_PER_TEAM_MATCH * (team_expected_goals / league_goals)
    df["expected_penalty_goals"] = (
        team_penalties * df[penalty_share_col] * PENALTY_CONVERSION
        * (df["expected_minutes"] > 0).astype(float)
    )

    # Subtract only the penalty goals actually HANDED TO SOMEBODY, not the theoretical team
    # total. The two differ whenever no penalty taker is identified — which is every row of
    # the backtest, because the archive has no set-piece order and `penalty_share` is 0
    # throughout. Subtracting the theoretical total there deleted 0.11/1.43 x 0.79 = 6.1% of
    # every team's expected goals and gave it to nobody, which was the whole of the measured
    # 7% goal under-prediction. Written this way the allocation conserves the team total by
    # construction: with a full set of takers it subtracts the lot, with none it subtracts
    # nothing, and the penalty mass is simply spread by open-play share instead.
    allocated_penalty_goals = float(df["expected_penalty_goals"].sum())
    open_play_total = max(team_expected_goals - allocated_penalty_goals, 0.0)
    raw = df["xg_per90"] * expected_90s * df["finishing_multiplier"]
    share = raw / raw.sum() if raw.sum() > 0 else pd.Series(0.0, index=df.index)
    df["expected_open_play_goals"] = share * open_play_total
    df["expected_goals"] = df["expected_open_play_goals"] + df["expected_penalty_goals"]

    # Assists track team goals, but not every goal is assisted.
    raw_assists = df["xa_per90"] * expected_90s
    assist_share = (
        raw_assists / raw_assists.sum() if raw_assists.sum() > 0 else pd.Series(0.0, index=df.index)
    )
    df["expected_assists"] = assist_share * team_expected_goals * ASSIST_RATE
    return df


def goal_involvement_pmf(expected: float, max_events: int = 4) -> np.ndarray:
    """Poisson pmf over event counts for one player in one fixture.

    Poisson rather than a point estimate because captaincy and differentials are decided by
    the upside tail, not the mean — `P(scores 2+)` is a different question from
    `E[goals]` and the optimiser needs both.
    """
    from scipy.stats import poisson

    pmf = poisson.pmf(np.arange(max_events + 1), max(expected, 0.0))
    pmf[-1] += max(1.0 - pmf.sum(), 0.0)   # fold the tail into the last bucket
    return pmf


def attacking_points(
    expected_goals: pd.Series, expected_assists: pd.Series, position: pd.Series,
    goal_points: dict[str, int], assist_points: int = 3,
) -> pd.Series:
    """Expected points from goals and assists, using the configured scoring table."""
    per_goal = position.map(goal_points).astype(float)
    return expected_goals * per_goal + expected_assists * assist_points
