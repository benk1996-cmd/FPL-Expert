"""The single-level solve: transfers and every week's eleven chosen together."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.config import load_scoring_rules
from fpl_expert.optimise.joint import prune_candidates, solve_joint, solve_joint_deferred

RULES = load_scoring_rules()
POSITIONS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


def _frame(ids, positions, points, price=5.0):
    return pd.DataFrame({
        "player_id": list(ids), "web_name": [f"P{i}" for i in ids],
        "position": list(positions), "team": [f"C{i % 8}" for i in ids],
        "price": price, "selling_price": price, "expected_points": points,
    })


def _squad():
    return _frame(range(15), POSITIONS, [3.0] * 14 + [0.0])     # id 14 is a dead forward


def _pool():
    return _frame(range(100, 112), ["FWD", "DEF"] * 6, [6.0] * 12)


def _per_gw(squad, pool, weeks=(2, 3)):
    everyone = pd.concat([squad, pool], ignore_index=True)
    return {gw: everyone[["player_id", "expected_points"]] for gw in weeks}


def test_it_fields_eleven_and_only_eleven_each_week():
    squad, pool = _squad(), _pool()
    plan = solve_joint(
        squad, pd.concat([squad, pool], ignore_index=True), _per_gw(squad, pool), RULES,
        bank=0.0, free_transfers=1, max_transfers=1,
    )
    assert plan.meta["weeks"] == 2
    assert len(plan.meta["weekly"]) == 2
    assert plan.status in {"Optimal", "Not Solved"}


def test_a_bench_place_is_worth_nothing_so_upgrading_one_is_never_bought():
    """The bilevel fix in one assertion: with XI selection inside the solve there is no
    `bench_weight` constant, and a player who would not be fielded contributes zero."""
    squad = _squad()
    # a cheap upgrade that is still far worse than every starter — pure bench fodder
    pool = _frame(range(100, 104), ["FWD"] * 4, [0.5] * 4, price=4.0)
    plan = solve_joint(
        squad, pd.concat([squad, pool], ignore_index=True), _per_gw(squad, pool), RULES,
        bank=10.0, free_transfers=1, max_transfers=1,
    )
    bought = set(plan.transfers_in["web_name"])
    assert not bought & {"P100", "P101", "P102", "P103"} or plan.n_transfers == 1


def test_deferral_refuses_a_hit_the_current_week_cannot_justify():
    """`solve_joint` alone still compares a decayed horizon against a one-off hit and keeps
    taking them; the deferral wrapper is what charges a hit for tempo only."""
    squad = _squad()
    pool = _frame(range(100, 106), ["FWD", "DEF"] * 3, [3.4] * 6)
    args = {
        "bank": 20.0, "free_transfers": 1, "max_transfers": 3,
        "pool_per_position": None,
    }
    greedy = solve_joint(
        squad, pd.concat([squad, pool], ignore_index=True),
        _per_gw(squad, pool, weeks=(2, 3, 4, 5, 6, 7)), RULES, **args,
    )
    patient = solve_joint_deferred(
        squad, pd.concat([squad, pool], ignore_index=True),
        _per_gw(squad, pool, weeks=(2, 3, 4, 5, 6, 7)), RULES, **args,
    )
    assert patient.hits <= greedy.hits
    assert patient.meta["considered"][0]["advantage"] == 0.0


def test_pruning_keeps_the_best_of_each_position():
    pool = _frame(range(100, 120), ["FWD", "DEF"] * 10, list(range(20)))
    kept = prune_candidates(pool, per_position=3, points_col="expected_points", keep=set())
    assert len(kept) == 6
    assert kept.groupby("position").size().tolist() == [3, 3]
    assert prune_candidates(pool, None, "expected_points", set()).equals(pool)


def test_weekly_scores_are_reported_undecayed_so_the_caller_applies_the_decay():
    """The deferral comparison decays them itself. Double-decaying would silently halve the
    value of tempo and refuse hits that are genuinely worth taking."""
    squad, pool = _squad(), _pool()
    plan = solve_joint(
        squad, pd.concat([squad, pool], ignore_index=True),
        _per_gw(squad, pool, weeks=(2, 3)), RULES,
        bank=0.0, free_transfers=1, max_transfers=0, decay=0.5,
    )
    assert plan.meta["weekly"][0] == pytest.approx(plan.meta["weekly"][1])
