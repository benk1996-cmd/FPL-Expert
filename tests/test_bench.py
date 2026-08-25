"""Autosub-weighted bench value: the arithmetic, and what it changes about a transfer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from fpl_expert.config import load_scoring_rules
from fpl_expert.optimise.bench import (
    autosub_slot_weights,
    blank_distribution,
    squad_value,
)

RULES = load_scoring_rules()


POSITIONS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
# A 1-3-4-3 leaves exactly these four out. Named explicitly because an earlier version of this
# fixture gave every forward the low score, which quietly put a "bench" player in the XI and
# made the test assert the opposite of what it claimed.
BENCH_IDX = [1, 5, 6, 11]          # spare GK, two spare DEF, one spare MID


def _squad(points, p_zero=0.1):
    """Two GK, five DEF, five MID, three FWD."""
    return pd.DataFrame({
        "player_id": range(15),
        "web_name": [f"P{i}" for i in range(15)],
        "position": POSITIONS,
        "team": [f"C{i % 6}" for i in range(15)],
        "price": 5.0,
        "selling_price": 5.0,
        "horizon_points": points,
        "p_zero": [p_zero] * 15,
    })


def _points(xi: float, bench: float) -> list[float]:
    return [bench if i in BENCH_IDX else xi for i in range(15)]


def test_blank_distribution_is_an_exact_poisson_binomial():
    """Two independent coin flips: 0 blanks .25, exactly one .5, both .25."""
    assert blank_distribution([0.5, 0.5]) == pytest.approx([0.25, 0.5, 0.25])
    assert blank_distribution([]) == pytest.approx([1.0])
    assert blank_distribution([0.0] * 5) == pytest.approx([1.0, 0, 0, 0, 0, 0])
    assert sum(blank_distribution([0.13, 0.4, 0.22, 0.05])) == pytest.approx(1.0)


def test_slot_weights_are_the_survival_function_and_decrease():
    """P(N>=1) >= P(N>=2) >= P(N>=3) — later bench slots are reached less often."""
    weights = autosub_slot_weights([0.15] * 10, n_slots=3)
    assert weights == sorted(weights, reverse=True)
    assert 0.0 <= weights[-1] <= weights[0] <= 1.0

    certain = autosub_slot_weights([1.0, 1.0], n_slots=2)
    assert certain == pytest.approx([1.0, 1.0])          # both starters always blank
    assert autosub_slot_weights([0.0] * 10, n_slots=3) == pytest.approx([0.0, 0.0, 0.0])


def test_the_first_bench_slot_is_worth_far_more_than_the_flat_guess():
    """The number this replaces was a hardcoded 0.10 for every bench place.

    With a realistic XI the first slot is reached most weeks, so 0.10 under-valued it by
    several times over — which is the substance of the change, not a rounding difference.
    """
    from fpl_expert.optimise.squad import DEFAULT_BENCH_WEIGHT

    first, _, third = autosub_slot_weights([0.15] * 10, n_slots=3)
    assert first > 4 * DEFAULT_BENCH_WEIGHT
    assert third < first


def test_squad_value_counts_the_xi_fully_and_the_bench_at_a_discount():
    points = [10.0] * 15
    value = squad_value(_squad(points), RULES)

    assert value > 11 * 10.0          # the bench is not worthless
    assert value < 15 * 10.0          # nor is it worth as much as a starting place


def test_a_blank_proof_xi_makes_the_bench_worthless():
    """If no starter can fail to appear, no autosub ever fires."""
    squad = _squad([10.0] * 15, p_zero=0.0)
    assert squad_value(squad, RULES) == pytest.approx(11 * 10.0)


def test_bench_points_move_squad_value_far_less_than_starter_points():
    """The property the transfer optimiser was missing: upgrading a bench player is worth a
    fraction of upgrading a starter, so it should not justify the same 4-point hit."""
    # Distinct bench scores so the bench ORDER is stable: a bump must not silently promote a
    # player up the queue, which is what makes a naive version of this test measure slot 1.
    base = _points(20.0, 5.0)
    for slot, idx in enumerate(BENCH_IDX):
        base[idx] = 9.0 - slot                     # 9, 8, 7, 6 down the bench
    baseline = squad_value(_squad(base), RULES)

    def gain(index, amount=0.5):
        bumped = list(base)
        bumped[index] += amount
        return squad_value(_squad(bumped), RULES) - baseline

    starter = gain(0)                              # a starting keeper
    deepest = gain(BENCH_IDX[-1])                  # still last in the queue after +0.5

    assert starter == pytest.approx(0.5)           # a starter's points count in full
    assert 0.0 < deepest < 0.2 * starter           # the deepest bench place, barely at all
    assert gain(BENCH_IDX[1]) > deepest            # and earlier slots are worth more


def test_an_empty_squad_is_worth_nothing_rather_than_raising():
    assert squad_value(pd.DataFrame(), RULES) == 0.0


def test_missing_p_zero_degrades_to_no_autosubs_rather_than_guessing():
    squad = _squad([10.0] * 15).drop(columns=["p_zero"])
    assert squad_value(squad, RULES) == pytest.approx(11 * 10.0)


def test_weights_never_exceed_one_however_bad_the_squad():
    assert max(autosub_slot_weights(np.linspace(0.5, 1.0, 10), n_slots=3)) <= 1.0


def test_the_live_path_is_bench_aware_by_default_and_the_backtest_is_not():
    """A deliberate, recorded divergence — not an oversight.

    `analyse_entry` defaults ON so `fpl myteam` stops taking hits for bench upgrades.
    `simulate_season` defaults OFF because every backtested number in DECISIONS was measured
    under the sum-of-fifteen policy, and flipping it would silently invalidate them. This test
    pins both so the gap cannot close by accident in either direction — it must be closed by
    running the `bench_aware` ensemble variant and deciding.
    """
    import inspect

    from fpl_expert.advice import analyse_entry
    from fpl_expert.backtest.season_sim import simulate_season
    from fpl_expert.cli import ENSEMBLE_VARIANTS

    assert inspect.signature(analyse_entry).parameters["bench_aware"].default is True
    assert inspect.signature(simulate_season).parameters["bench_aware"].default is False
    assert ENSEMBLE_VARIANTS["bench_aware"] == {"bench_aware": True}


def test_bench_aware_falls_back_loudly_rather_than_silently_when_rules_are_missing(caplog):
    """`bench_aware=True` without `rules` cannot compute an XI. It must not quietly behave as
    if the flag were off — that is the fallback-that-degrades-silently trap."""
    import inspect

    from fpl_expert.optimise.transfers import recommend_transfers

    params = inspect.signature(recommend_transfers).parameters
    assert params["bench_aware"].default is False      # the FUNCTION stays off by default
    assert params["rules"].default is None

    squad, candidates = _squad(_points(20.0, 5.0)), _squad(_points(30.0, 6.0))
    candidates["player_id"] = range(100, 115)
    with caplog.at_level(logging.WARNING):
        recommend_transfers(squad, candidates, bank=50.0, free_transfers=1,
                            bench_aware=True, rules=None)
    assert "falling back" in caplog.text
