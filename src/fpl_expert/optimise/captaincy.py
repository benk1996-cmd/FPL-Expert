"""Captaincy as a durable squad property, not a weekly afterthought.

The squad optimiser already doubles the captain's points for the gameweek it is solving. What
it does not price is that captaincy is *recurring*: over a 38-week season the armband is
awarded 38 times, and a player who is your best option most weeks earns close to 2x his raw
output while a player who never wears it earns 1x. Judged on expected points alone those two
look identical, which is why a reliable captain is systematically underpriced.

Two refinements matter.

**Credit is shared, not winner-take-all.** Assigning the armband to the forecast argmax gives
a player who is marginally best in every week full credit and his near-equal rival none.
Forecasts are not that precise. A softmax over the leading candidates spreads credit in
proportion to how plausibly each is the right pick, and the temperature controls how much
faith is placed in small forecast gaps.

**The armband doubles risk as well as return.** `Var(2X) = 4·Var(X)`, so captaincy quadruples
a player's variance contribution. A risk-neutral objective is indifferent between a steady 6
and a coin-flip between 0 and 12; a manager is not, and a chasing manager should actively
prefer the coin-flip. `variance_weight` exposes that and stays at 0 until the simulator can
measure it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Only realistic armband candidates compete. Anyone outside the top of a gameweek's board is
# never captained, and including them would dilute the softmax toward uniform.
DEFAULT_CANDIDATE_POOL = 20
DEFAULT_TEMPERATURE = 1.0


def _per_player(frame: pd.DataFrame, points_col: str, key: str) -> pd.DataFrame:
    """Collapse to one row per player.

    Required because a player has one row per fixture, so in a DOUBLE gameweek he appears
    twice — and the armband applies to his combined score, not to each fixture separately.
    """
    return frame.groupby(key, as_index=False)[points_col].sum()


def captaincy_probabilities(
    frame: pd.DataFrame,
    *,
    points_col: str = "expected_points",
    temperature: float = DEFAULT_TEMPERATURE,
    pool: int = DEFAULT_CANDIDATE_POOL,
    key: str = "player_id",
) -> pd.Series:
    """P(this player wears the armband) for one gameweek, over the plausible candidates."""
    contenders = _per_player(frame, points_col, key).nlargest(pool, points_col)
    if contenders.empty:
        return pd.Series(dtype=float)

    scores = contenders[points_col].to_numpy(dtype=float)
    # Subtracting the max before exponentiating keeps this stable for large point totals.
    weights = np.exp((scores - scores.max()) / max(temperature, 1e-6))
    return pd.Series(weights / weights.sum(), index=contenders[key].to_numpy())


def captaincy_uplift(
    forecasts_by_gw: dict[int, pd.DataFrame],
    *,
    decay: float = 0.84,
    points_col: str = "expected_points",
    temperature: float = DEFAULT_TEMPERATURE,
    pool: int = DEFAULT_CANDIDATE_POOL,
    key: str = "player_id",
) -> pd.DataFrame:
    """Extra points expected from the armband across a planning horizon.

    This is the premium a reliable captain commands. It is added to, not substituted for,
    a player's own expected points: the armband grants a SECOND copy of his score, so the
    uplift is `P(captain) * expected_points`, discounted like everything else.
    """
    totals: dict = {}
    for step, (_, frame) in enumerate(sorted(forecasts_by_gw.items())):
        weight = decay**step
        probabilities = captaincy_probabilities(
            frame, points_col=points_col, temperature=temperature, pool=pool, key=key
        )
        if probabilities.empty:
            continue
        points = _per_player(frame, points_col, key).set_index(key)[points_col]
        for player_id, probability in probabilities.items():
            uplift = probability * float(points.get(player_id, 0.0)) * weight
            totals[player_id] = totals.get(player_id, 0.0) + uplift

    return pd.DataFrame(
        {key: list(totals), "captaincy_uplift": list(totals.values())}
    )


def captaincy_reliability(
    forecasts_by_gw: dict[int, pd.DataFrame], **kwargs
) -> pd.DataFrame:
    """Share of horizon gameweeks in which a player is the likely armband.

    Reported rather than optimised. It answers the question a manager actually asks — "is
    this a captain I can keep going back to?" — which raw uplift blurs, because a single
    enormous double gameweek can match six steady weeks.
    """
    key = kwargs.get("key", "player_id")
    frames = sorted(forecasts_by_gw.items())
    counts: dict = {}
    for _, frame in frames:
        probabilities = captaincy_probabilities(frame, **kwargs)
        for player_id, probability in probabilities.items():
            counts[player_id] = counts.get(player_id, 0.0) + probability

    n = max(len(frames), 1)
    return pd.DataFrame({
        key: list(counts),
        "captain_weeks": [v for v in counts.values()],
        "captain_share": [v / n for v in counts.values()],
    }).sort_values("captain_share", ascending=False)


def captain_score(
    frame: pd.DataFrame,
    *,
    points_col: str = "expected_points",
    variance_col: str = "points_variance",
    variance_weight: float = 0.0,
) -> pd.Series:
    """Ranking score for the armband this week.

    The armband doubles the mean and quadruples the variance contribution. At
    `variance_weight = 0` this is pure expected points, which is correct when maximising
    total points; a positive weight prefers upside, which is what chasing a rank requires.
    """
    score = 2.0 * frame[points_col]
    if variance_weight and variance_col in frame.columns:
        score = score + variance_weight * np.sqrt(4.0 * frame[variance_col].clip(lower=0))
    return score
