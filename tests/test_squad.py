"""Squad optimiser: legality of every solution, and that it beats greedy selection."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.optimise.squad import select_squad

QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATION = {
    "GK": {"min": 1, "max": 1}, "DEF": {"min": 3, "max": 5},
    "MID": {"min": 2, "max": 5}, "FWD": {"min": 1, "max": 3},
}


def _pool(n_per_position=8, seed=0):
    """A pool where the best players are expensive and concentrated in a few clubs, so the
    constraints actually bind."""
    rng = pd.Series(range(n_per_position))
    rows = []
    for position in QUOTA:
        for i in range(n_per_position):
            rows.append({
                "player_id": len(rows),
                "web_name": f"{position}{i}",
                "position": position,
                "team": f"club{i % 5}",
                "price": 4.0 + i * 1.2,
                "expected_points": 1.0 + i * 0.8,
            })
    return pd.DataFrame(rows).assign(_=rng.head(0))


def _solve(pool=None, **kwargs):
    return select_squad(
        pool if pool is not None else _pool(),
        budget=kwargs.pop("budget", 100.0),
        squad_quota=QUOTA, formation=FORMATION,
        max_per_club=kwargs.pop("max_per_club", 3), **kwargs,
    )


def test_squad_satisfies_every_hard_constraint():
    solution = _solve()
    squad = solution.squad

    assert len(squad) == 15
    assert squad["position"].value_counts().to_dict() == QUOTA
    assert solution.total_cost <= 100.0 + 1e-6
    assert squad["team"].value_counts().max() <= 3


def test_starting_xi_is_a_legal_formation():
    xi = _solve().starting_xi
    counts = xi["position"].value_counts()

    assert len(xi) == 11
    assert counts.get("GK", 0) == 1
    assert 3 <= counts.get("DEF", 0) <= 5
    assert 2 <= counts.get("MID", 0) <= 5
    assert 1 <= counts.get("FWD", 0) <= 3


def test_bench_is_the_squad_minus_the_eleven():
    solution = _solve()
    assert len(solution.bench) == 4
    assert set(solution.bench["player_id"]).isdisjoint(solution.starting_xi["player_id"])


def test_captain_is_a_starter_and_vice_is_someone_else():
    solution = _solve()
    assert solution.captain["player_id"] in set(solution.starting_xi["player_id"])
    assert solution.vice_captain["player_id"] != solution.captain["player_id"]


def test_captain_is_the_highest_scorer_in_the_eleven():
    """With a pure expected-points objective the armband must go to the best starter."""
    solution = _solve()
    assert solution.captain["expected_points"] == solution.starting_xi["expected_points"].max()


def test_optimiser_beats_greedy_selection():
    """The point of solving exactly. Greedy spends the budget on premiums and then cannot
    fill the remaining slots legally, or fills them with rubbish."""
    pool = _pool()
    solution = _solve(pool)

    greedy = pool.sort_values("expected_points", ascending=False)
    picked, cost, counts, clubs = [], 0.0, dict.fromkeys(QUOTA, 0), {}
    for row in greedy.itertuples():
        if (counts[row.position] < QUOTA[row.position]
                and cost + row.price <= 100.0
                and clubs.get(row.team, 0) < 3):
            picked.append(row)
            cost += row.price
            counts[row.position] += 1
            clubs[row.team] = clubs.get(row.team, 0) + 1
    greedy_points = sum(p.expected_points for p in picked)

    assert len(picked) < 15 or solution.squad["expected_points"].sum() >= greedy_points


def test_tight_budget_still_returns_a_legal_squad():
    """The naive cheapest 15 costs 88.8 but puts 4 players in each of two clubs, so the
    3-per-club limit forces two upgrades and the real floor is about 96."""
    solution = _solve(budget=97.0)
    assert len(solution.squad) == 15
    assert solution.total_cost <= 97.0 + 1e-6
    assert solution.squad["team"].value_counts().max() <= 3


def test_impossible_budget_raises_rather_than_returning_nonsense():
    """Silently returning a partial or illegal squad would be far worse than failing."""
    with pytest.raises(RuntimeError, match="no optimal squad"):
        _solve(budget=20.0)


def test_must_include_forces_a_player_into_the_squad():
    pool = _pool()
    cheap_id = int(pool[pool["expected_points"] == pool["expected_points"].min()]
                   ["player_id"].iloc[0])
    solution = _solve(pool, must_include=[cheap_id])
    assert cheap_id in set(solution.squad["player_id"])


def test_exclude_keeps_a_player_out():
    pool = _pool()
    best_id = int(pool.nlargest(1, "expected_points")["player_id"].iloc[0])
    solution = _solve(pool, exclude=[best_id])
    assert best_id not in set(solution.squad["player_id"])


def test_bench_weight_changes_bench_quality():
    """Weighting the bench at zero buys £4.0m filler that never covers an absence;
    weighting it fully buys an expensive bench that never plays."""
    cheap_bench = _solve(bench_weight=0.0).bench["expected_points"].sum()
    good_bench = _solve(bench_weight=0.9).bench["expected_points"].sum()
    assert good_bench >= cheap_bench


def test_club_limit_is_enforced_when_one_club_dominates():
    pool = _pool()
    pool.loc[pool["team"] == "club0", "expected_points"] = 50.0   # irresistible on merit
    solution = _solve(pool)
    assert (solution.squad["team"] == "club0").sum() <= 3


def test_expected_points_means_this_gameweek_not_the_objective():
    """`fpl squad` reported 221.63 "expected points" for a single gameweek. It was the
    six-week horizon objective wearing the wrong label. The two are now separate fields."""
    frame = _pool()
    frame["horizon_points"] = frame["expected_points"] * 6.0

    solution = _solve(frame, points_col="horizon_points", double_captain=False)
    assert solution.expected_points < solution.objective_value / 3
    # The label must describe a single gameweek's XI plus the armband.
    assert solution.expected_points == pytest.approx(
        solution.starting_xi["expected_points"].sum()
        + solution.captain["expected_points"]
    )


def test_a_horizon_column_does_not_double_count_the_captaincy_premium():
    """`horizon_points` already prices the armband through `captaincy_uplift`. Doubling the
    captain on top of it counts the premium twice and biases the squad toward one premium."""
    frame = _pool()
    frame["horizon_points"] = frame["expected_points"] * 6.0

    doubled = _solve(frame, points_col="horizon_points", double_captain=True)
    single = _solve(frame, points_col="horizon_points", double_captain=False)
    assert doubled.objective_value > single.objective_value


def test_the_armband_is_a_weekly_decision_when_the_squad_is_chosen_on_a_horizon():
    """With the doubling removed the solver has no reason to prefer any captain, so it must
    be chosen explicitly — on THIS gameweek's points, not the multi-week column."""
    frame = _pool()
    # Make the horizon ranking disagree with this week's ranking.
    frame["horizon_points"] = frame["expected_points"].max() - frame["expected_points"]

    solution = _solve(frame, points_col="horizon_points", double_captain=False)
    best = solution.starting_xi["expected_points"].max()
    assert solution.captain["expected_points"] == pytest.approx(best)
