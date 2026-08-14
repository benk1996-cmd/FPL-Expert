"""Rank-aware objective, bonus allocation, validation metrics and monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.monitor import check, retraining_due
from fpl_expert.backtest.validate import calibration_summary, component_accuracy
from fpl_expert.models.bonus import allocate_match_bonus
from fpl_expert.optimise.risk import captain_choice, differential_score, effective_points

# --- bonus allocation -----------------------------------------------------


def test_match_bonus_pool_is_fixed_at_six():
    """Exactly six bonus points exist per fixture (3+2+1) regardless of how many strong
    players are on the pitch. Two goalscorers cannot both collect three."""
    allocated = allocate_match_bonus(pd.Series([1.0, 0.8, 0.6, 0.4, 0.2]))
    assert allocated.sum() == pytest.approx(6.0)


def test_bonus_allocation_follows_propensity_order():
    allocated = allocate_match_bonus(pd.Series([0.9, 0.3, 0.1]))
    assert allocated.iloc[0] > allocated.iloc[1] > allocated.iloc[2]


def test_bonus_allocation_handles_a_match_with_no_propensity():
    allocated = allocate_match_bonus(pd.Series([0.0, 0.0]))
    assert allocated.sum() == 0.0


def test_negative_propensity_cannot_earn_bonus():
    allocated = allocate_match_bonus(pd.Series([-1.0, 2.0]))
    assert allocated.iloc[0] == 0.0
    assert allocated.iloc[1] == pytest.approx(6.0)


# --- rank-aware objective -------------------------------------------------


def test_lambda_zero_is_pure_expected_points():
    points = pd.Series([5.0, 4.0])
    assert effective_points(points, pd.Series([80.0, 2.0]), 0.0).tolist() == [5.0, 4.0]


def test_rank_objective_discounts_template_players():
    """Rank depends on your score minus the field's, and the template largely cancels.
    A player owned by most of the field cannot move your rank much when he hauls."""
    points = pd.Series([5.0, 5.0])
    ownership = pd.Series([90.0, 5.0])          # template vs differential, equal forecast
    adjusted = effective_points(points, ownership, lambda_rank=0.5)
    assert adjusted.iloc[1] > adjusted.iloc[0]


def test_rank_objective_still_prefers_a_much_better_template_player():
    """Being contrarian without edge is pure noise — the discount must not override a real
    forecast gap."""
    adjusted = effective_points(
        pd.Series([9.0, 3.0]), pd.Series([90.0, 1.0]), lambda_rank=0.4
    )
    assert adjusted.iloc[0] > adjusted.iloc[1]


def test_differential_score_rewards_low_ownership():
    scores = differential_score(pd.Series([5.0, 5.0]), pd.Series([100.0, 3.0]))
    assert scores.iloc[1] > scores.iloc[0]


def test_captain_choice_defaults_to_the_highest_scorer():
    candidates = pd.DataFrame({
        "web_name": ["a", "b"], "expected_points": [6.0, 5.0],
        "points_variance": [4.0, 4.0], "eo": [80.0, 5.0],
    })
    assert captain_choice(candidates)["web_name"] == "a"


def test_captain_choice_can_prefer_a_differential_when_chasing_rank():
    """Captaining a 70%-owned premium is nearly rank-neutral, so a manager chasing a high
    rank may rationally take a slightly lower mean at much lower ownership."""
    candidates = pd.DataFrame({
        "web_name": ["template", "differential"],
        "expected_points": [6.0, 5.6], "points_variance": [4.0, 9.0],
        "eo": [160.0, 4.0],
    })
    assert captain_choice(candidates, lambda_rank=0.0)["web_name"] == "template"
    assert captain_choice(candidates, lambda_rank=0.6)["web_name"] == "differential"


# --- validation -----------------------------------------------------------


def _scored(n=200, seed=0):
    rng = np.random.default_rng(seed)
    actual = rng.poisson(3, n).astype(float)
    return pd.DataFrame({
        "gw": np.repeat([1, 2], n // 2),
        "expected_points": actual * 0.8 + rng.normal(0, 0.5, n),
        "actual_points": actual,
        "expected_minutes": rng.uniform(0, 90, n),
        "actual_minutes": rng.uniform(0, 90, n),
        "p_long": rng.uniform(0, 1, n),
    })


def test_component_accuracy_reports_rank_correlation():
    """Rank correlation matters more than error size: FPL decisions are comparisons."""
    accuracy = component_accuracy(_scored()).set_index("component")
    assert accuracy.loc["expected_points", "spearman"] > 0.7
    assert accuracy.loc["expected_points", "n"] == 200


def test_calibration_summary_compares_predicted_to_observed():
    summary = calibration_summary(_scored())
    assert "P(60+ minutes)" in set(summary["probability"])
    assert (summary["ece"] >= 0).all()


# --- monitoring -----------------------------------------------------------


def test_monitor_flags_a_breach():
    """The failure this guards against is a model that keeps producing plausible numbers
    while drifting — which happened three times while building this system."""
    frame = _scored()
    frame["expected_points"] = frame["actual_points"].sample(frac=1, random_state=1).to_numpy()
    results = check(frame)
    assert (results["status"] == "BREACH").any()


def test_monitor_passes_a_healthy_model():
    results = check(_scored())
    spearman = results[results["metric"] == "points_spearman"]
    assert spearman["status"].iloc[0] == "ok"


def test_retraining_cadence_is_fixed_not_judged():
    """Refitting weekly chases noise and makes runs irreproducible; never refitting throws
    away a season of data."""
    snapshots = pd.DataFrame({"target_gw": [1, 2, 6]})
    due, message = retraining_due(snapshots, every_gameweeks=6)
    assert due and "DUE" in message

    not_due, _ = retraining_due(pd.DataFrame({"target_gw": [1, 2, 3]}), every_gameweeks=6)
    assert not not_due


def test_retraining_with_no_snapshots():
    due, message = retraining_due(pd.DataFrame())
    assert not due and "no snapshots" in message
