"""Transfers and team sheets chosen together, in one MILP.

The bilevel problem
-------------------

A squad's value is not a property of its fifteen players — it is the result of a SECOND
optimisation, the eleven you would field:

    maximise over transfers T:
        sum_k decay^k * [ max over legal XI within squad(T) of the XI's points ] - 4 * hits

A MILP needs a fixed coefficient per variable ("buying player j is worth c_j"), and here the
coefficient depends on the solution: Guehi is worth 4.53 if he starts and near nothing if he
does not, and whether he starts depends on who else was bought. That is why
`recommend_transfers` sums all FIFTEEN players' points — it is the only honest constant
available — and why bench value is counted at full weight and never realised.

Why it collapses to one level
-----------------------------

The nesting is max-inside-max, and the inner constraints touch the outer variables only through
"you may field only who you own". When that holds, both levels can live in the same problem:
add `start[p][k]` beside the ownership variables and let the solver pick the eleven, because
picking the best eleven is what maximises the objective. `select_squad` has always done this
for a single gameweek; this extends it across the planning horizon.

The bench weight disappears as a concept. There is no flat 0.10 and no autosub weighting to
tune: a bench place is worth exactly the weeks in which the solver chooses to field that player.

Cost
----

Roughly `players * horizon * 2` binaries against the previous `players`. `pool_per_position`
prunes the candidate list before building the model — a player outside the top N of his
position by forecast points is never the right buy, and 600 candidates is generous when most
are not viable. Set it to None to keep every candidate.
"""

from __future__ import annotations

import logging

import pandas as pd
import pulp

from .transfers import HIT_COST, TransferPlan

log = logging.getLogger(__name__)


def prune_candidates(
    candidates: pd.DataFrame, per_position: int | None, points_col: str, keep: set
) -> pd.DataFrame:
    """Top `per_position` by forecast in each position, plus everyone already held."""
    if per_position is None:
        return candidates
    best = (
        candidates.sort_values(points_col, ascending=False)
        .groupby("position", sort=False)
        .head(per_position)
    )
    held = candidates[candidates["player_id"].isin(keep)]
    return pd.concat([best, held]).drop_duplicates(subset="player_id").reset_index(drop=True)


def _points_by_week(ids, per_gw, points_col, gws):
    """(player_id, gw) -> points. A player absent from a week has a blank and scores nothing."""
    table = {}
    for gw in gws:
        frame = per_gw[gw]
        table[gw] = dict(zip(frame["player_id"], frame[points_col], strict=True))
    return {(p, gw): float(table[gw].get(p, 0.0)) for p in ids for gw in gws}


def solve_joint(
    squad: pd.DataFrame,
    candidates: pd.DataFrame,
    per_gw: dict[int, pd.DataFrame],
    rules: dict,
    *,
    bank: float,
    free_transfers: int,
    max_transfers: int = 2,
    max_per_club: int = 3,
    hit_cost: float = HIT_COST,
    decay: float = 0.84,
    points_col: str = "expected_points",
    pool_per_position: int | None = None,
    time_limit: int = 60,
    exact_transfers: int | None = None,
) -> TransferPlan:
    """Choose transfers and every week's eleven at once."""
    squad = squad.reset_index(drop=True)
    held_ids = set(squad["player_id"])
    pool = candidates[~candidates["player_id"].isin(held_ids)]
    pool = prune_candidates(pool, pool_per_position, points_col, keep=set()).reset_index(drop=True)
    if pool.empty:
        raise ValueError("no candidates outside the current squad")

    gws = sorted(per_gw)[: max(1, len(per_gw))]
    everyone = pd.concat(
        [squad.assign(_held=True), pool.assign(_held=False)], ignore_index=True
    )
    ids = everyone["player_id"].tolist()
    position = dict(zip(everyone["player_id"], everyone["position"], strict=True))
    club = dict(zip(everyone["player_id"], everyone["team"], strict=True))
    points = _points_by_week(ids, per_gw, points_col, gws)

    problem = pulp.LpProblem("fpl_joint", pulp.LpMaximize)
    sell = pulp.LpVariable.dicts("sell", squad.index.tolist(), cat="Binary")
    buy = pulp.LpVariable.dicts("buy", pool.index.tolist(), cat="Binary")
    start = pulp.LpVariable.dicts("start", [(p, g) for p in ids for g in gws], cat="Binary")
    capt = pulp.LpVariable.dicts("capt", [(p, g) for p in ids for g in gws], cat="Binary")
    hits = pulp.LpVariable("hits", lowBound=0, cat="Integer")

    held_index = {squad.at[i, "player_id"]: i for i in squad.index}
    pool_index = {pool.at[j, "player_id"]: j for j in pool.index}
    own = {
        p: (1 - sell[held_index[p]]) if p in held_index else buy[pool_index[p]] for p in ids
    }

    n_out = pulp.lpSum(sell.values())
    problem += n_out == pulp.lpSum(buy.values())
    problem += n_out <= max_transfers
    if exact_transfers is not None:
        problem += n_out == exact_transfers
    problem += hits >= n_out - free_transfers
    problem += hits >= 0

    problem += (
        pulp.lpSum(pool.at[j, "price"] * buy[j] for j in pool.index)
        <= pulp.lpSum(squad.at[i, "selling_price"] * sell[i] for i in squad.index) + bank
    )

    for pos in rules["squad"]["positions"]:
        out_pos = pulp.lpSum(sell[i] for i in squad.index if squad.at[i, "position"] == pos)
        in_pos = pulp.lpSum(buy[j] for j in pool.index if pool.at[j, "position"] == pos)
        problem += out_pos == in_pos
    for team in set(club.values()):
        problem += pulp.lpSum(own[p] for p in ids if club[p] == team) <= max_per_club

    formation = rules["squad"]["formation"]
    for g in gws:
        for p in ids:
            problem += start[(p, g)] <= own[p]
            problem += capt[(p, g)] <= start[(p, g)]
        problem += pulp.lpSum(start[(p, g)] for p in ids) == rules["squad"]["starting_xi"]
        problem += pulp.lpSum(capt[(p, g)] for p in ids) == 1
        for pos, limits in formation.items():
            members = [p for p in ids if position[p] == pos]
            if "min" in limits:
                problem += pulp.lpSum(start[(p, g)] for p in members) >= limits["min"]
            if "max" in limits:
                problem += pulp.lpSum(start[(p, g)] for p in members) <= limits["max"]

    problem += (
        pulp.lpSum(
            decay**step * points[(p, g)] * (start[(p, g)] + capt[(p, g)])
            for step, g in enumerate(gws)
            for p in ids
        )
        - hit_cost * hits
    )

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[problem.status]
    if status not in {"Optimal", "Not Solved"}:
        raise RuntimeError(f"joint solve failed (status: {status})")

    out_idx = [i for i in squad.index if sell[i].value() and sell[i].value() > 0.5]
    in_idx = [j for j in pool.index if buy[j].value() and buy[j].value() > 0.5]
    taken = max(0, len(out_idx) - free_transfers)
    value = float(pulp.value(problem.objective) or 0.0)

    return TransferPlan(
        transfers_in=pool.loc[in_idx],
        transfers_out=squad.loc[out_idx],
        n_transfers=len(out_idx),
        hits=taken,
        hit_cost=taken * hit_cost,
        gain=value,
        net_gain=value,
        bank_after=float(
            bank + squad.loc[out_idx, "selling_price"].sum() - pool.loc[in_idx, "price"].sum()
        ),
        status=status,
        meta={
            "joint": True, "candidates": len(pool), "weeks": len(gws),
            "binaries": len(sell) + len(buy) + 2 * len(ids) * len(gws),
            # Each week's XI score, UNDECAYED, from the solver's own team sheets. The deferral
            # comparison must use these rather than a separate picker, or the two halves drift
            # apart on which eleven they think you would field.
            "weekly": [
                float(
                    sum(
                        points[(p, g)]
                        * ((start[(p, g)].value() or 0.0) + (capt[(p, g)].value() or 0.0))
                        for p in ids
                    )
                )
                for g in gws
            ],
        },
    )


def solve_joint_deferred(
    squad: pd.DataFrame,
    candidates: pd.DataFrame,
    per_gw: dict[int, pd.DataFrame],
    rules: dict,
    *,
    bank: float,
    free_transfers: int,
    max_transfers: int = 2,
    max_per_club: int = 3,
    hit_cost: float = HIT_COST,
    decay: float = 0.84,
    points_col: str = "expected_points",
    pool_per_position: int | None = 40,
    time_limit: int = 60,
) -> TransferPlan:
    """Both corrections at once: XI chosen inside the solve, hits charged only for tempo.

    `solve_joint` alone still compares a decayed six-week objective against a one-off hit, so it
    keeps taking them; the deferral rule alone still ranks a shortlist the sum-of-fifteen
    objective produced. Neither is sufficient and each was measured separately.

    The free-transfer plan is the baseline. Any larger plan is charged against the weeks it
    brings forward, using the solver's OWN weekly team sheets so both halves agree on which
    eleven is being valued.
    """
    common = {
        "bank": bank, "free_transfers": free_transfers, "max_per_club": max_per_club,
        "hit_cost": hit_cost, "decay": decay, "points_col": points_col,
        "pool_per_position": pool_per_position, "time_limit": time_limit,
    }
    free_cap = max(0, min(free_transfers, max_transfers))
    baseline = solve_joint(squad, candidates, per_gw, rules, max_transfers=free_cap, **common)
    free_weekly = baseline.meta["weekly"]

    best, best_advantage, considered = baseline, 0.0, [
        {"n": baseline.n_transfers, "hits": 0.0, "tempo": 0.0, "advantage": 0.0}
    ]
    for n in range(free_cap + 1, max_transfers + 1):
        try:
            plan = solve_joint(
                squad, candidates, per_gw, rules, max_transfers=max_transfers,
                exact_transfers=n, **common
            )
        except (ValueError, RuntimeError):
            continue
        if plan.n_transfers != n:
            continue
        weeks = max(1, plan.hits)
        tempo = sum(
            decay**k * (plan.meta["weekly"][k] - free_weekly[k])
            for k in range(min(weeks, len(free_weekly), len(plan.meta["weekly"])))
        )
        advantage = float(tempo) - plan.hit_cost
        considered.append(
            {"n": n, "hits": plan.hit_cost, "tempo": float(tempo), "advantage": advantage}
        )
        if advantage > best_advantage:
            best, best_advantage = plan, advantage

    best.meta = {**best.meta, "deferred": True, "considered": considered}
    log.info("joint+deferral: %s", considered)
    return best
