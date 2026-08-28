"""One definition of what a squad is worth, in one set of units.

The inconsistency this resolves
-------------------------------

`recommend_transfers` maximised `gain - 4 * hits`, where `gain` is a SIX-WEEK decayed sum and
the hit is a ONE-OFF. With `decay=0.84` over six gameweeks the multiplier is 4.054, so a player
worth +1 a week carries a horizon gain of 4.05 and clears a nominal 4. The effective bar for
taking a -4 hit was therefore **0.99 points per week** — which is why, with the `max_transfers`
cap lifted, the policy took seven transfers and six hits to reach the ideal fifteen in a single
gameweek and scored it as a gain.

Two further inconsistencies rode along with it:

* the objective summed all FIFTEEN players, though only eleven score (plus autosubs), so bench
  value counted at full weight and was never realised;
* `select_squad` discounted a bench place at a flat guessed 0.10 while `recommend_transfers`
  valued it at 1.0, so the two optimisers priced the same squad differently.

The consistent quantity
-----------------------

    V(S) = sum_k  decay^k * xi_value(S, gw + k)

The horizon of XI VALUES, not the horizon of fifteen-player sums. Every term is points actually
scored by an eleven, and the decay is the same one the forecasts use.

Comparing states rather than player sums then makes the hit arithmetic fall out. Taking a
transfer now versus deferring it to next week's free transfer gives two branches that hold an
IDENTICAL squad from the moment the deferred move lands, so every later term cancels and what
survives is the gap over the weeks before it lands:

    take now iff  sum over the deferred weeks of the XI gap  >  the hit

Both sides are in per-week points. No multiplier mismatch, no phantom bench value, and one
value function for both optimisers.
"""

from __future__ import annotations

import logging

import pandas as pd

from .bench import xi_value

log = logging.getLogger(__name__)


def squad_frames(
    squad: pd.DataFrame, per_gw: dict[int, pd.DataFrame], points_col: str = "expected_points"
) -> dict[int, pd.DataFrame]:
    """This squad's players, carrying each gameweek's own points.

    Identity and position come from the squad; only the points column varies by week. A player
    absent from a gameweek's frame has no fixture — a blank — and is dropped rather than
    zero-filled, because `pick_xi` fills the XI from whoever is actually available.
    """
    keep = [c for c in ("player_id", "web_name", "position", "team", "price") if c in squad]
    base = squad[keep]
    out = {}
    for gw, frame in per_gw.items():
        block = frame[["player_id", points_col]]
        merged = base.merge(block, on="player_id", how="inner")
        if not merged.empty:
            out[gw] = merged
    return out


def horizon_xi_value(
    squad: pd.DataFrame,
    per_gw: dict[int, pd.DataFrame],
    rules: dict,
    *,
    decay: float = 0.84,
    points_col: str = "expected_points",
) -> float:
    """`V(S)` above: the decayed sum of the XI this squad would field each gameweek."""
    frames = squad_frames(squad, per_gw, points_col)
    total = 0.0
    for step, gw in enumerate(sorted(frames)):
        total += decay**step * xi_value(frames[gw], rules, points_col)
    return float(total)


def deferral_gap(
    now: pd.DataFrame,
    later: pd.DataFrame,
    per_gw: dict[int, pd.DataFrame],
    rules: dict,
    *,
    weeks: int,
    decay: float = 0.84,
    points_col: str = "expected_points",
) -> float:
    """What holding `now` instead of `later` is worth over the `weeks` before they converge.

    `weeks` is the number of transfers being brought forward: each needs its own free transfer,
    so h extra transfers take h weeks to make for nothing. Deliberately assumes the deferred
    branch holds the OLD squad throughout those weeks, when in truth it would converge one
    transfer at a time — so this OVERSTATES the value of acting now, erring toward the
    incumbent policy rather than toward the change.
    """
    a = squad_frames(now, per_gw, points_col)
    b = squad_frames(later, per_gw, points_col)
    gws = sorted(set(a) & set(b))[: max(1, weeks)]
    return float(
        sum(
            decay**step * (xi_value(a[gw], rules, points_col) - xi_value(b[gw], rules, points_col))
            for step, gw in enumerate(gws)
        )
    )
