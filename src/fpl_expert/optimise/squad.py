"""Squad selection as a mixed-integer program.

Picking a squad is not "take the highest expected points" — it is a constrained knapsack.
£100m buys 15 players in fixed positional quotas with at most 3 from any club, and only 11
of them score, with a captain doubling. Greedy selection reliably fails here: it spends the
budget on premiums and then cannot fill the remaining slots legally.

Formulated exactly, this solves in well under a second, so there is no reason to approximate.

Three linked binary decisions per player:

    squad[i]    in the 15
    start[i]    in the starting XI            (start <= squad)
    captain[i]  wears the armband             (captain <= start)

The bench is not worthless — autosubs replace non-playing starters — so bench players carry
a small weight rather than zero. Weighting them fully would buy an expensive bench that never
plays; weighting them zero picks £4.0m players who never cover an absence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import pulp

log = logging.getLogger(__name__)

# What a bench place is worth relative to a starting place. Bench players score only when an
# autosub fires, which is uncommon but not rare.
DEFAULT_BENCH_WEIGHT = 0.10


@dataclass
class SquadSolution:
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    bench: pd.DataFrame
    captain: pd.Series
    vice_captain: pd.Series
    total_cost: float
    expected_points: float
    objective_value: float = 0.0
    status: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"status: {self.status}",
            (f"cost: £{self.total_cost:.1f}m   "
             f"expected points (this GW): {self.expected_points:.2f}"),
            (f"captain: {self.captain.get('web_name', '?')}"
             f"   vice: {self.vice_captain.get('web_name', '?')}"),
        ]
        return "\n".join(lines)


def select_squad(
    players: pd.DataFrame,
    *,
    budget: float,
    squad_quota: dict[str, int],
    formation: dict[str, dict[str, int]],
    max_per_club: int,
    squad_size: int = 15,
    starting_size: int = 11,
    bench_weight: float = DEFAULT_BENCH_WEIGHT,
    points_col: str = "expected_points",
    double_captain: bool = True,
    must_include: list[int] | None = None,
    exclude: list[int] | None = None,
    time_limit: float = 60.0,
    flexibility_weight: float = 0.0,
) -> SquadSolution:
    """Solve for the best legal squad, starting XI and captain.

    `players` needs `player_id`, `position`, `team`, `price` and a points column.
    """
    df = players.reset_index(drop=True).copy()
    if exclude:
        df = df[~df["player_id"].isin(exclude)].reset_index(drop=True)
    ids = df.index.tolist()
    points = df[points_col].fillna(0.0).to_dict()
    price = df["price"].to_dict()

    problem = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", ids, cat="Binary")
    start = pulp.LpVariable.dicts("start", ids, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", ids, cat="Binary")

    # A starter must be in the squad; a captain must be starting. Without these the solver
    # would happily captain a player it never bought.
    for i in ids:
        problem += start[i] <= squad[i]
        problem += captain[i] <= start[i]

    problem += pulp.lpSum(squad[i] for i in ids) == squad_size
    problem += pulp.lpSum(start[i] for i in ids) == starting_size
    problem += pulp.lpSum(captain[i] for i in ids) == 1
    problem += pulp.lpSum(price[i] * squad[i] for i in ids) <= budget

    for position, quota in squad_quota.items():
        members = [i for i in ids if df.at[i, "position"] == position]
        problem += pulp.lpSum(squad[i] for i in members) == quota
        limits = formation.get(position, {})
        if "min" in limits:
            problem += pulp.lpSum(start[i] for i in members) >= limits["min"]
        if "max" in limits:
            problem += pulp.lpSum(start[i] for i in members) <= limits["max"]

    for club in df["team"].dropna().unique():
        members = [i for i in ids if df.at[i, "team"] == club]
        problem += pulp.lpSum(squad[i] for i in members) <= max_per_club

    for player_id in must_include or []:
        members = [i for i in ids if df.at[i, "player_id"] == player_id]
        if members:
            problem += pulp.lpSum(squad[i] for i in members) == 1

    # The armband grants a SECOND copy of the captain's score — but only when `points_col`
    # is a single gameweek. A horizon column already amortises captaincy across the window
    # through `captaincy_uplift`, so doubling it there counts the premium twice and biases
    # the squad toward one premium captain. `double_captain=False` is the horizon case.
    objective = pulp.lpSum(
        points[i] * (start[i] + (captain[i] if double_captain else 0)
                     + bench_weight * (squad[i] - start[i]))
        for i in ids
    )

    if flexibility_weight:
        # Option value of being able to pivot: reward holding the most expensive player in
        # each position, because a downgrade is always affordable while an upgrade may not
        # be. `max` is not linear, so it is expressed with a selector variable per position —
        # since the term is being MAXIMISED the solver sets it to the priciest held player,
        # which is exactly the reach we want to measure.
        for position in squad_quota:
            members = [i for i in ids if df.at[i, "position"] == position]
            if not members:
                continue
            anchor = pulp.LpVariable.dicts(f"anchor_{position}", members, cat="Binary")
            for i in members:
                problem += anchor[i] <= squad[i]
            problem += pulp.lpSum(anchor[i] for i in members) == 1
            objective += flexibility_weight * pulp.lpSum(
                price[i] * anchor[i] for i in members
            )

    problem += objective

    # A time limit is not optional. Pools with many identically-priced, identically-scored
    # players are highly symmetric, and branch-and-bound can explore that symmetry almost
    # indefinitely — a weekly job must never hang waiting on it.
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    status = pulp.LpStatus[problem.status]
    if status not in {"Optimal", "Not Solved"} or squad[ids[0]].value() is None:
        raise RuntimeError(f"no optimal squad found (status: {status})")
    if status != "Optimal":
        log.warning("solver hit the %.0fs limit; using best incumbent solution", time_limit)

    chosen = [i for i in ids if squad[i].value() > 0.5]
    starters = [i for i in ids if start[i].value() > 0.5]
    bench_idx = [i for i in chosen if i not in starters]

    if double_captain:
        captain_idx = next(i for i in ids if captain[i].value() > 0.5)
    else:
        # With the doubling removed, `captain[i]` no longer appears in the objective, so the
        # solver is free to put the armband anywhere — it picked a £8m forward over the
        # highest-scoring midfielder purely because both were feasible. The armband is a
        # WEEKLY decision, so it is made here on this gameweek's expected points rather than
        # on the multi-week column the squad was chosen with.
        armband_col = "expected_points" if "expected_points" in df.columns else points_col
        captain_idx = df.loc[starters, armband_col].idxmax()

    xi = df.loc[starters].sort_values(["position", points_col], ascending=[True, False])
    # Vice-captain: best starter who is not the captain, so the armband survives a late absence.
    vice_idx = xi[xi.index != captain_idx][points_col].idxmax()

    return SquadSolution(
        squad=df.loc[chosen],
        starting_xi=xi,
        bench=df.loc[bench_idx].sort_values(points_col, ascending=False),
        captain=df.loc[captain_idx],
        vice_captain=df.loc[vice_idx],
        total_cost=float(df.loc[chosen, "price"].sum()),
        # The value the OPTIMISER maximised, in whatever units `points_col` is.
        objective_value=float(
            df.loc[starters, points_col].sum()
            + (df.at[captain_idx, points_col] if double_captain else 0.0)
        ),
        # This gameweek's expected points, which is what the name promises. When the squad was
        # chosen on a horizon column these differ by a factor of six, and reporting the
        # objective under this label made `fpl squad` claim 221 points for one gameweek.
        expected_points=float(
            df.loc[starters, "expected_points"].sum() + df.at[captain_idx, "expected_points"]
        ) if "expected_points" in df.columns else float(
            df.loc[starters, points_col].sum() + df.at[captain_idx, points_col]
        ),
        status=status,
        meta={"bench_weight": bench_weight, "candidates": len(df)},
    )
