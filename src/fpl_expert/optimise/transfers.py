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


def _solve_transfers(
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
    exact_transfers: int | None = None,
) -> TransferPlan:
    """Choose the transfers worth making, if any.

    `squad` needs `player_id`, `position`, `team`, `selling_price` and the points column.
    `candidates` is every purchasable player with `price` and the same points column.
    Points should be summed over the planning horizon, not a single gameweek — a transfer
    is a durable change, so judging it on one week systematically over-trades.

    **The hit threshold was re-derived after the point-in-time fix, and raising it is
    rejected.** This docstring previously called for that work; it has since been done
    (2026-08-13) and the answer was no. The motivating bias is real and about as stable as
    anything in this project — the forecast margin behind a transfer is overstated ~2.3x,
    slope 0.436 in every season — but the implied bar of ~9 measures **-38 / -42 / +45**
    across the three seasons, and every `hit_bar` variant swept (6, 8, 9, 10, 12) flips sign
    between seasons on 10 perturbed decision paths each. None is adoptable.

    So `hit_cost` stays at the nominal 4, not because it is proven but because nothing beats
    it reliably. Do not re-open this on the strength of the 2.3x measurement alone: that is
    ground rule 3, and the hit bar is the fifth principled bias fix to measure to nothing.
    See DECISIONS, "Decision-layer parameter sweep (2026-08-13)".
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
    if exact_transfers is not None:
        # Used by the bench-aware wrapper, which needs the best plan AT each size so it can
        # rescore them against one another on a value the MILP cannot express.
        problem += n_out == exact_transfers
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
    rules: dict | None = None,
    bench_aware: bool = False,
) -> TransferPlan:
    """Choose the transfers worth making, if any.

    With `bench_aware=False` this is the plain MILP: it maximises the change in the SUM of
    fifteen players' points, which treats a fourth-choice defender as worth exactly as much as
    the captain. That over-values depth, and it is how a -4 hit gets taken to upgrade a player
    who never starts.

    With `bench_aware=True` and `rules` supplied, each plan is instead judged on
    `bench.squad_value` — the XI in full plus each bench place weighted by the chance an
    autosub reaches it. That value cannot be written as a MILP objective, because it depends on
    which eleven the post-transfer squad would field, which is itself an optimisation over the
    solution. So the MILP proposes the best plan AT each transfer count and those few plans are
    rescored exactly. Solve-once, rescore-many — the same shape as the season simulator.

    **Off by default and not yet measured.** It corrects a real inconsistency (`select_squad`
    already discounts the bench, at a flat guessed 0.10) but this project has watched six
    principled corrections measure to nothing, and a change to the transfer policy needs the
    ensemble to resolve. See ground rule 3.
    """
    common = {
        "bank": bank, "free_transfers": free_transfers, "max_per_club": max_per_club,
        "squad_quota": squad_quota, "hit_cost": hit_cost, "points_col": points_col,
    }
    if bench_aware and rules is None:
        # Ground rule 7: a fallback that changes the model must say so. Silently reverting to
        # the sum-of-fifteen here would hand back a plan the caller believes was bench-aware.
        log.warning(
            "bench_aware=True but no `rules` given — cannot pick an XI, so falling back to the "
            "sum-of-fifteen objective. Pass rules=load_scoring_rules() to enable it."
        )
    if not bench_aware or rules is None:
        return _solve_transfers(squad, candidates, max_transfers=max_transfers, **common)

    from .bench import squad_value

    pool = candidates[~candidates["player_id"].isin(squad["player_id"])]
    before = squad_value(squad, rules, points_col=points_col)

    best, best_score, scored = None, None, []
    for n in range(max_transfers + 1):
        try:
            plan = _solve_transfers(
                squad, candidates, max_transfers=max_transfers, exact_transfers=n, **common
            )
        except ValueError:
            continue
        if plan.n_transfers != n:
            continue                      # infeasible at this size; CBC returned nothing
        kept = squad[~squad["player_id"].isin(plan.transfers_out["player_id"])]
        after_squad = pd.concat([kept, plan.transfers_in], ignore_index=True)
        after = squad_value(after_squad, rules, points_col=points_col)
        score = after - before - plan.hit_cost
        scored.append({"n": n, "delta": after - before, "hits": plan.hit_cost, "net": score})
        if best_score is None or score > best_score:
            best, best_score = plan, score

    if best is None:                      # nothing solved; fall back to the plain objective
        return _solve_transfers(squad, candidates, max_transfers=max_transfers, **common)

    # Report the value the DECISION was made on, not the sum-of-fifteen the MILP proposed it
    # with — otherwise the summary quotes a number nothing was chosen by.
    chosen = next(row for row in scored if row["n"] == best.n_transfers)
    best.gain = float(chosen["delta"])
    best.net_gain = float(chosen["net"])
    best.meta = {**best.meta, "bench_aware": True, "considered": scored,
                 "squad_value_before": before, "candidates": len(pool)}
    log.info("bench-aware rescoring: %s", scored)
    return best


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
