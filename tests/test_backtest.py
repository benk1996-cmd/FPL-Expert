"""Scoring rules and rolling-origin splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.metrics import (
    accuracy,
    brier_score,
    calibration_table,
    compare,
    expected_calibration_error,
    log_loss,
)
from fpl_expert.backtest.walkforward import holdout_tail, prior_seasons, season_splits


def test_log_loss_rewards_confident_correctness():
    y = np.array([2, 2])
    confident = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    hedged = np.array([[1 / 3, 1 / 3, 1 / 3]] * 2)
    assert log_loss(y, confident) < log_loss(y, hedged)


def test_log_loss_punishes_confident_errors_hardest():
    y = np.array([0])
    assert log_loss(y, np.array([[0.001, 0.0, 0.999]])) > log_loss(y, np.array([[0.3, 0.3, 0.4]]))


def test_brier_is_zero_for_a_perfect_forecast():
    assert brier_score(np.array([1]), np.array([[0.0, 1.0, 0.0]])) == pytest.approx(0.0)


def test_accuracy_alone_can_look_good_while_the_model_is_useless():
    """58% of player-gameweeks are zero minutes, so always predicting 'never plays' scores
    well on accuracy and is worthless. This is why log loss is the headline metric."""
    y = np.array([0] * 58 + [2] * 42)
    always_zero = np.tile([1.0, 0.0, 0.0], (100, 1))
    honest = np.tile([0.58, 0.0, 0.42], (100, 1))

    assert accuracy(y, always_zero) == pytest.approx(0.58)
    assert accuracy(y, honest) == pytest.approx(0.58)          # identical on accuracy
    assert log_loss(y, honest) < log_loss(y, always_zero)      # but not on log loss


def test_calibration_table_detects_overconfidence():
    probs = np.full(100, 0.9)
    outcomes = np.array([1.0] * 50 + [0.0] * 50)   # claimed 90%, delivered 50%
    table = calibration_table(outcomes, probs, bins=10)
    row = table[table["n"] > 0].iloc[0]
    assert row["gap"] == pytest.approx(0.4)
    assert expected_calibration_error(outcomes, probs) == pytest.approx(0.4)


def test_compare_ranks_by_log_loss():
    y = np.array([2, 2, 0, 0])
    good = np.array([[0.1, 0.1, 0.8]] * 2 + [[0.8, 0.1, 0.1]] * 2)
    bad = np.array([[0.4, 0.3, 0.3]] * 4
                   )
    out = compare(y, {"bad": bad, "good": good})
    assert out["model"].iloc[0] == "good"


def test_season_splits_never_train_on_the_future():
    df = pd.DataFrame({"season": np.repeat(
        ["2021-22", "2022-23", "2023-24", "2024-25"], 3)})
    splits = list(season_splits(df, min_train_seasons=2))

    assert [s for s, _, _ in splits] == ["2023-24", "2024-25"]
    for test_season, train, test in splits:
        assert train["season"].max() < test_season
        assert set(test["season"]) == {test_season}


def test_season_splits_needs_enough_history():
    df = pd.DataFrame({"season": ["2024-25", "2024-25"]})
    with pytest.raises(ValueError, match="need more than"):
        list(season_splits(df, min_train_seasons=2))


def test_prior_seasons_excludes_the_tested_season():
    """The saved minutes model is fitted on every season, including the one being replayed.
    Reusing it in a backtest lets the biggest driver of points see the future of the very
    gameweeks it is scored on — measured at 185 season points, a 9% overstatement."""
    features = pd.DataFrame({
        "season": ["2023-24", "2024-25", "2025-26"], "x": [1, 2, 3],
    })
    train = prior_seasons(features, "2025-26")

    assert set(train["season"]) == {"2023-24", "2024-25"}
    assert "2025-26" not in set(train["season"])


def test_prior_seasons_is_empty_for_the_earliest_season():
    """Nothing to train on means the season cannot be backtested, and the caller must be
    able to detect that rather than silently fitting on nothing."""
    features = pd.DataFrame({"season": ["2023-24"], "x": [1]})
    assert prior_seasons(features, "2023-24").empty


def test_holdout_tail_takes_the_most_recent_slice():
    """Validation drawn at random from the same weeks as training would let early stopping
    tune against contemporaneous information."""
    train = pd.DataFrame({"_idx": range(100), "x": range(100)})
    fit, valid = holdout_tail(train, frac=0.2)
    assert len(valid) == 20
    assert valid["_idx"].min() > fit["_idx"].max()
