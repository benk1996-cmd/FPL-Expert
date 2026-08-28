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
    defer_aware: bool = False,
    per_gw: dict | None = None,
    decay: float = 0.84,
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

    **`bench_aware` is off by default and stays off** — the ensemble measured it at
    -126 / -65 / +98 across three seasons and rejected it. See DECISIONS (2026-08-26).

    `defer_aware` fixes a different and more basic error: **the optimiser has no concept of
    next week's free transfer.** It compares "N transfers now, paying hits" against "fewer
    transfers now, and never make the rest", when the real alternative is "fewer now, the rest
    next week for nothing". Because a hit is one-off and the horizon gain is a six-week sum,
    almost any upgrade clears a nominal 4 — so with the `max_transfers` cap lifted the policy
    takes SEVEN transfers and six hits to reach the ideal fifteen in one week, and scores that
    as a gain. Only the cap prevents it, and a cap is not a reason.

    The correction is structural, not a tuned bar. If a transfer can be made next week for
    free, taking it now buys exactly one gameweek of its edge — so a hit must be justified by
    the CURRENT gameweek's improvement, not by the whole horizon. Formally, comparing "take now"
    against "defer one week" cancels every later gameweek and leaves

        take now iff  (this gameweek's XI gain from the extra transfers)  >  the hit

    Off by default until the ensemble resolves it. Turning `bench_aware` on before measuring it
    was a mistake once already this week.
    """
    common = {
        "bank": bank, "free_transfers": free_transfers, "max_per_club": max_per_club,
        "squad_quota": squad_quota, "hit_cost": hit_cost, "points_col": points_col,
    }
    if (bench_aware or defer_aware) and rules is None:
        # Ground rule 7: a fallback that changes the model must say so. Silently reverting to
        # the plain objective would hand back a plan the caller believes was corrected.
        log.warning(
            "bench_aware/defer_aware set but no `rules` given — cannot pick an XI, so falling "
            "back to the sum-of-fifteen objective. Pass rules=load_scoring_rules()."
        )
    if rules is None or not (bench_aware or defer_aware):
        return _solve_transfers(squad, candidates, max_transfers=max_transfers, **common)

    if defer_aware and per_gw:
        # The consistent form: both halves at once. `defer_aware` alone still carried a
        # sum-of-fifteen horizon; `bench_aware` alone still compared a horizon against an
        # unmultiplied hit. Each fixed one inconsistency and kept the other.
        return _consistent_plan(
            squad, candidates, rules=rules, per_gw=per_gw, decay=decay,
            max_transfers=max_transfers, free_transfers=free_transfers, common=common,
        )

    if defer_aware:
        return _defer_aware_plan(
            squad, candidates, rules=rules, max_transfers=max_transfers,
            free_transfers=free_transfers, hit_cost=hit_cost, points_col=points_col,
            common=common,
        )

    from .bench import squad_value

    pool = candidates[~candidates["player_id"].isin(squad["player_id"])]
    before = squad_value(squad, rules, points_col=points_col)

    best, best_score, scored = None, None, []
    for n in range(max_transfers + 1):
        try:
            plan = _solve_transfers(
                squad, candidates, max_transfers=max_transfers, exact_transfers=n, **common
            )
        except (ValueError, RuntimeError):
            # Infeasible AT THIS SIZE — no legal way to make exactly n transfers within the
            # budget, quota and club limits. That is a gap in the ladder, not a failure.
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


def _consistent_plan(
    squad, candidates, *, rules, per_gw, decay, max_transfers, free_transfers, common
):
    """Judge every plan on one quantity, in one set of units.

    Squads are valued by `horizon_xi_value` — the decayed sum of the ELEVEN they would field
    each week — and a hit is charged against the weeks it actually buys, because the deferred
    branch converges to the same squad once its free transfers arrive.
    """
    from .valuation import deferral_gap, horizon_xi_value

    free_cap = max(0, min(free_transfers, max_transfers))
    baseline = _solve_transfers(squad, candidates, max_transfers=free_cap, **common)
    after_free = _apply(squad, baseline)

    best, best_advantage, considered = baseline, 0.0, [{
        "n": baseline.n_transfers, "hits": 0.0, "tempo": 0.0, "advantage": 0.0,
        "horizon_xi": horizon_xi_value(after_free, per_gw, rules, decay=decay),
    }]
    for n in range(free_cap + 1, max_transfers + 1):
        try:
            plan = _solve_transfers(
                squad, candidates, max_transfers=max_transfers, exact_transfers=n, **common
            )
        except (ValueError, RuntimeError):
            continue
        if plan.n_transfers != n:
            continue
        after = _apply(squad, plan)
        weeks = max(1, plan.hits)
        tempo = deferral_gap(after, after_free, per_gw, rules, weeks=weeks, decay=decay)
        advantage = tempo - plan.hit_cost
        considered.append({
            "n": n, "hits": plan.hit_cost, "tempo": tempo, "advantage": advantage,
            "horizon_xi": horizon_xi_value(after, per_gw, rules, decay=decay),
        })
        if advantage > best_advantage:
            best, best_advantage = plan, advantage

    best.meta = {**best.meta, "consistent": True, "considered": considered}
    log.info("consistent valuation: %s", considered)
    return best


def _defer_aware_plan(
    squad, candidates, *, rules, max_transfers, free_transfers, hit_cost, points_col, common
):
    """Take a hit only when it buys enough THIS gameweek to beat making the move next week.

    The free-transfer plan is the baseline, because it costs nothing and is always available.
    Any plan beyond it is judged on the single thing a hit actually buys — tempo — which is the
    improvement to this week's XI, since every later gameweek is identical either way.
    """
    from .bench import xi_value

    free_cap = max(0, min(free_transfers, max_transfers))
    baseline = _solve_transfers(squad, candidates, max_transfers=free_cap, **common)
    after_free = _apply(squad, baseline)
    free_value = xi_value(after_free, rules)

    best, best_advantage, considered = baseline, 0.0, [
        {"n": baseline.n_transfers, "hits": 0.0, "immediate_edge": 0.0, "advantage": 0.0}
    ]
    for n in range(free_cap + 1, max_transfers + 1):
        try:
            plan = _solve_transfers(
                squad, candidates, max_transfers=max_transfers, exact_transfers=n, **common
            )
        except (ValueError, RuntimeError):
            # Infeasible AT THIS SIZE — no legal way to make exactly n transfers within the
            # budget, quota and club limits. That is a gap in the ladder, not a failure.
            continue
        if plan.n_transfers != n:
            continue
        edge = xi_value(_apply(squad, plan), rules) - free_value
        advantage = edge - plan.hit_cost
        considered.append(
            {"n": n, "hits": plan.hit_cost, "immediate_edge": edge, "advantage": advantage}
        )
        if advantage > best_advantage:
            best, best_advantage = plan, advantage

    best.meta = {
        **best.meta, "defer_aware": True, "considered": considered,
        "free_plan_transfers": baseline.n_transfers,
    }
    log.info("defer-aware: %s", considered)
    return best


def _apply(squad: pd.DataFrame, plan) -> pd.DataFrame:
    """The fifteen you would hold after `plan`."""
    if plan is None or not plan.n_transfers:
        return squad
    kept = squad[~squad["player_id"].isin(plan.transfers_out["player_id"])]
    return pd.concat([kept, plan.transfers_in], ignore_index=True)


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
