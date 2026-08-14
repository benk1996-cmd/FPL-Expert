"""Squad flexibility: the option value of being able to pivot.

The hypothesis this exists to test: holding the most expensive player in a position is worth
more than his expected points alone, because it preserves the ability to react. Downgrading
is always affordable; upgrading often is not. Hold a £15.5m forward and a £7m forward starts
outperforming, and you can switch and bank £8m. Hold the £7m and the premium may simply be
out of reach.

The measurable quantity is **positional reach** — the most you could spend on a single
replacement in a position:

    reach[q] = bank + max(selling price of the players you hold in q)

A squad holding a £15m striker can buy any forward in the game. A squad of three £6m
strikers cannot reach one without a multi-transfer chain that costs points.

Two counterweights, both real:

1. **Bank dominates premiums on flexibility alone.** Cash reaches every position; a premium
   only reaches his own. The premium's advantage is that it scores points while sitting
   there, and bank does not. That trade-off is the whole question.
2. **The sell-on fee erodes reach.** You keep only half of any price rise, so a premium that
   appreciates gives you less reach than his market price suggests — the flexibility decays
   precisely when the player is doing well.

Nothing here is assumed to help. `flexibility_weight` defaults to 0 and is only worth raising
if the season simulator says so.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

PRICE_COLUMN_PREFERENCE = ("selling_price", "price")


def _price_column(squad: pd.DataFrame) -> str:
    """Selling price where known, market price otherwise.

    Using market price overstates reach: you receive the selling price, which is lower for
    any player who has risen.
    """
    for column in PRICE_COLUMN_PREFERENCE:
        if column in squad.columns:
            return column
    raise KeyError("squad needs a selling_price or price column")


def positional_reach(squad: pd.DataFrame, bank: float = 0.0) -> pd.Series:
    """Most spendable on one replacement per position, without a multi-transfer chain."""
    column = _price_column(squad)
    return squad.groupby("position")[column].max() + bank


def flexibility_score(squad: pd.DataFrame, bank: float = 0.0) -> float:
    """Total reach across positions — a single number for comparing squad shapes."""
    return float(positional_reach(squad, bank).sum())


def pivot_cost(
    squad: pd.DataFrame, target: pd.Series, bank: float = 0.0
) -> float:
    """Extra money needed to acquire `target` with one transfer. Zero means reachable now.

    A positive value is the funding gap, which in practice means a second transfer and a
    -4 hit, or selling someone you wanted to keep.
    """
    column = _price_column(squad)
    same_position = squad[squad["position"] == target["position"]]
    if same_position.empty:
        return float("inf")
    reach = float(same_position[column].max()) + bank
    return max(0.0, float(target["price"]) - reach)


def reachable_share(
    squad: pd.DataFrame, candidates: pd.DataFrame, bank: float = 0.0, top_n: int = 30
) -> float:
    """Fraction of the best available players you could sign with a single transfer.

    A more decision-relevant reading of flexibility than raw reach: what matters is not how
    much you *could* spend but how much of the genuinely useful market it opens.
    """
    best = candidates.nlargest(top_n, "expected_points")
    if best.empty:
        return 0.0
    reachable = sum(pivot_cost(squad, row, bank) <= 0 for _, row in best.iterrows())
    return reachable / len(best)


def summarise(squad: pd.DataFrame, bank: float = 0.0) -> pd.DataFrame:
    """Per-position reach and who currently provides it — a report-friendly view."""
    column = _price_column(squad)
    rows = []
    for position, block in squad.groupby("position"):
        anchor = block.loc[block[column].idxmax()]
        rows.append({
            "position": position,
            "reach": float(block[column].max()) + bank,
            "anchor": anchor.get("web_name", "?"),
            "anchor_price": float(anchor[column]),
        })
    return pd.DataFrame(rows).sort_values("reach", ascending=False)
