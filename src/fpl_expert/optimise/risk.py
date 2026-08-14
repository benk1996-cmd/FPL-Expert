"""Rank-aware objective: scoring points against the field rather than in absolute terms.

Required because the season target is top 10k. Maximising expected points and maximising
P(high rank) are different problems: rank depends on `X - Y`, your score minus the field's,
and the template largely cancels out of that difference. A player owned by 80% of the field
contributes almost nothing to your rank when he hauls — you rise with everyone — but costs
you enormously when you do not own him.

The objective becomes

    effective = expected_points - lambda_rank * effective_ownership/100 * expected_points

which at `lambda_rank = 0` is pure expected points, and at higher values progressively
discounts players the field already owns. Differentials keep their full value.

Two things this deliberately does NOT do:

*It does not chase variance for its own sake.* Being contrarian with no edge is pure noise:
it raises P(top 1k) while lowering expected points. The discount above only pays off where
our forecast genuinely disagrees with ownership, which is where any real edge lives.

*It is not tuned here.* `lambda_rank` stays 0 until the season simulator can measure it.
Tuning a rank objective without a rank-aware backtest would be guessing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def effective_points(
    expected_points: pd.Series,
    effective_ownership: pd.Series,
    lambda_rank: float = 0.0,
) -> pd.Series:
    """Discount expected points by how much of the field already owns the player."""
    if lambda_rank <= 0:
        return expected_points
    eo = pd.to_numeric(effective_ownership, errors="coerce").fillna(0.0).clip(0, 300) / 100.0
    return expected_points * (1.0 - lambda_rank * eo)


def differential_score(
    expected_points: pd.Series, effective_ownership: pd.Series
) -> pd.Series:
    """Expected points per unit of field exposure — a ranking of who moves your rank most.

    Useful as a report column even when `lambda_rank` is 0: it surfaces where our forecast
    and the crowd disagree, which is where a decision is worth thinking about.
    """
    eo = pd.to_numeric(effective_ownership, errors="coerce").fillna(0.0).clip(lower=0)
    return expected_points / np.sqrt(1.0 + eo / 100.0)


def captain_choice(
    candidates: pd.DataFrame,
    *,
    lambda_rank: float = 0.0,
    points_col: str = "expected_points",
    variance_col: str = "points_variance",
    eo_col: str | None = "eo",
) -> pd.Series:
    """Pick the armband, accounting for the field when a rank target is set.

    Captaincy is where the two objectives diverge most sharply. Captaining a 70%-owned
    premium is nearly rank-neutral — most of the field does the same — so a chasing manager
    may rationally prefer a slightly lower-mean, lower-owned alternative.
    """
    scored = candidates.copy()
    base = 2 * scored[points_col]
    if lambda_rank > 0 and eo_col and eo_col in scored.columns:
        scored["captain_score"] = effective_points(base, scored[eo_col], lambda_rank)
    else:
        scored["captain_score"] = base
    if variance_col in scored.columns:
        # Upside matters for the armband: two players with the same mean are not equivalent
        # if one is a 12-point-or-nothing striker and the other a steady 5.
        scored["captain_score"] += 0.05 * lambda_rank * np.sqrt(scored[variance_col].clip(lower=0))
    return scored.nlargest(1, "captain_score").iloc[0]
