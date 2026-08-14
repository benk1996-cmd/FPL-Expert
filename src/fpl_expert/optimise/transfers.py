"""Transfer recommendation: what to do with the squad you actually have.

Distinct from `squad.py`, which builds an opening 15 from nothing. From GW2 onward the
problem changes shape entirely: you hold 15 players, you may sell them only at their
selling price, and every change beyond your free transfers costs 4 points. That makes it a
*marginal* decision, not a fresh optimisation — the right question is never "what is the
best squad?" but "is this specific change worth what it costs?"

Solved exactly as a MILP so the four things that make it awkward are handled together
rather than in sequence:

  * selling prices differ from market prices (you keep half of any profit);
  * a hit is -4 points, so a transfer must clear that bar to be worth taking;
  * the 3-per-club and positional quotas must hold AFTER the change, not before;
  * the best single transfer and the best pair are often disjoint — greedy picks the wrong
    first move and then cannot afford the second.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import pulp

log = logging.getLogger(__name__)

HIT_COST = 4.0


@dataclass
class TransferPlan:
    transfers_in: pd.DataFrame
    transfers_out: pd.DataFrame
    n_transfers: int
    hits: int
    hit_cost: float
    gain: float
    net_gain: float
    bank_after: float
    status: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        if self.n_transfers == 0:
            return "Recommendation: ROLL the transfer. No move clears its cost."
        lines = [
            (f"Recommendation: {self.n_transfers} transfer(s), {self.hits} hit(s) "
             f"costing {self.hit_cost:.0f} pts"),
            (f"Expected gain over the horizon: {self.gain:+.2f}  "
             f"net {self.net_gain:+.2f}  bank after £{self.bank_after:.1f}m"),
        ]
        return "\n".join(lines)


def recommend_transfers(
    squad: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    bank: float,
    free_transfers: int,
    max_per_club: int = 3,
    squad_quota: dict[str, int] | None = None,
    max_transfers: int = 3,
    hit_cost: float = HIT_COST,
    points_col: str = "horizon_points",
) -> TransferPlan:
    """Choose the transfers worth making, if any.

    `squad` needs `player_id`, `position`, `team`, `selling_price` and the points column.
    `candidates` is every purchasable player with `price` and the same points column.
    Points should be summed over the planning horizon, not a single gameweek — a transfer
    is a durable change, so judging it on one week systematically over-trades.

    **The hit threshold has not been re-derived since the horizon was made point-in-time
    (2026-08-12), and the evidence says it should be.** Against a myopic policy the horizon
    now pays 72 / 96 / 76 more points in hits every season — a reliable cost — for extra
    transfer value of +280 / -4 / +77, which is not reliable at all. In 2024-25 it made 24
    more transfers whose realised six-week return was identical to the myopic policy's, for
    96 points of hits. `hit_cost` was implicitly calibrated against a forward valuation that
    barely moved between gameweeks, and honest forecasts move. See DECISIONS.
    """
    squad = squad.reset_index(drop=True)
    pool = candidates[~candidates["player_id"].isin(squad["player_id"])].reset_index(drop=True)
    if pool.empty:
        raise ValueError("no candidates outside the current squad")

    problem = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)
    sell = pulp.LpVariable.dicts("sell", squad.index.tolist(), cat="Binary")
    buy = pulp.LpVariable.dicts("buy", pool.index.tolist(), cat="Binary")
    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")

    n_out = pulp.lpSum(sell.values())
    n_in = pulp.lpSum(buy.values())

    # Squad size is fixed: every sale must be matched by a purchase.
    problem += n_out == n_in
    problem += n_out <= max_transfers
    problem += hits >= n_out - free_transfers
    problem += hits >= 0

    # Budget: proceeds are SELLING prices, not market prices.
    proceeds = pulp.lpSum(squad.at[i, "selling_price"] * sell[i] for i in squad.index)
    outlay = pulp.lpSum(pool.at[j, "price"] * buy[j] for j in pool.index)
    problem += outlay <= proceeds + bank

    # Positional quotas must hold after the change: a sold defender needs a defender back.
    quota = squad_quota or squad["position"].value_counts().to_dict()
    for position in quota:
        out_pos = pulp.lpSum(
            sell[i] for i in squad.index if squad.at[i, "position"] == position
        )
        in_pos = pulp.lpSum(buy[j] for j in pool.index if pool.at[j, "position"] == position)
        problem += out_pos == in_pos

    # Club limit, evaluated on the post-transfer squad.
    for club in set(squad["team"]) | set(pool["team"]):
        held = squad[squad["team"] == club].index
        incoming = pool[pool["team"] == club].index
        problem += (
            len(held)
            - pulp.lpSum(sell[i] for i in held)
            + pulp.lpSum(buy[j] for j in incoming)
        ) <= max_per_club

    gain = (
        pulp.lpSum(pool.at[j, points_col] * buy[j] for j in pool.index)
        - pulp.lpSum(squad.at[i, points_col] * sell[i] for i in squad.index)
    )
    problem += gain - hit_cost * hits

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))
    status = pulp.LpStatus[problem.status]
    if status not in {"Optimal", "Not Solved"}:
        raise RuntimeError(f"transfer solve failed (status: {status})")

    out_idx = [i for i in squad.index if sell[i].value() and sell[i].value() > 0.5]
    in_idx = [j for j in pool.index if buy[j].value() and buy[j].value() > 0.5]
    n = len(out_idx)
    taken_hits = max(0, n - free_transfers)

    raw_gain = float(
        pool.loc[in_idx, points_col].sum() - squad.loc[out_idx, points_col].sum()
    )
    bank_after = float(
        bank + squad.loc[out_idx, "selling_price"].sum() - pool.loc[in_idx, "price"].sum()
    )

    return TransferPlan(
        transfers_in=pool.loc[in_idx],
        transfers_out=squad.loc[out_idx],
        n_transfers=n,
        hits=taken_hits,
        hit_cost=taken_hits * hit_cost,
        gain=raw_gain,
        net_gain=raw_gain - taken_hits * hit_cost,
        bank_after=bank_after,
        status=status,
        meta={"free_transfers": free_transfers, "candidates": len(pool)},
    )


def horizon_points(
    forecasts_by_gw: dict[int, pd.DataFrame],
    *,
    decay: float = 0.84,
    points_col: str = "expected_points",
    key: str = "player_id",
    captaincy_weight: float = 1.0,
) -> pd.DataFrame:
    """Discounted expected points over a planning horizon, including the captaincy premium.

    Future gameweeks are discounted because forecasts decay in reliability and because
    squads churn — a player you would sell in three weeks is worth less than his raw total
    suggests. Judging a transfer on a single gameweek over-trades badly; judging it on an
    undiscounted horizon under-reacts to current form.

    The captaincy term matters more than it looks. Without it a player who would wear the
    armband for the next five gameweeks is valued identically to one who never would, even
    though the armband grants a second copy of his score. That systematically underprices
    exactly the reliable premium captains the game rewards. Set `captaincy_weight = 0` to
    recover the old behaviour.
    """
    frames = []
    for step, (gw, frame) in enumerate(sorted(forecasts_by_gw.items())):
        weight = decay**step
        block = frame[[key, points_col]].copy()
        block["weighted"] = block[points_col] * weight
        block["gw"] = gw
        frames.append(block)

    combined = pd.concat(frames, ignore_index=True)
    out = (
        combined.groupby(key, as_index=False)["weighted"].sum()
        .rename(columns={"weighted": "horizon_points"})
    )

    if captaincy_weight:
        from .captaincy import captaincy_uplift

        uplift = captaincy_uplift(
            forecasts_by_gw, decay=decay, points_col=points_col, key=key
        )
        out = out.merge(uplift, on=key, how="left")
        out["captaincy_uplift"] = out["captaincy_uplift"].fillna(0.0)
        out["horizon_points"] += captaincy_weight * out["captaincy_uplift"]
    return out
