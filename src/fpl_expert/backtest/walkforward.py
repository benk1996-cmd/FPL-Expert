"""Rolling-origin evaluation: train on the past, test on the future, never the reverse.

Random cross-validation is invalid for this problem. Shuffling player-gameweeks puts a
player's GW20 in the training set and his GW19 in the test set, so the model learns from
the future of the very row it is scoring. It produces flattering, meaningless numbers.

Splitting by season is the honest version and mirrors how the model will actually be used:
everything before the tested season is available, nothing from it or after.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pandas as pd

log = logging.getLogger(__name__)


def season_splits(
    df: pd.DataFrame, min_train_seasons: int = 2, season_col: str = "season"
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield (test_season, train, test) with train strictly preceding test."""
    seasons = sorted(df[season_col].dropna().unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(f"need more than {min_train_seasons} seasons, have {len(seasons)}")

    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train = df[df[season_col].isin(seasons[:i])]
        test = df[df[season_col] == test_season]
        if train.empty or test.empty:
            continue
        yield test_season, train, test


def prior_seasons(
    features: pd.DataFrame, season: str, season_col: str = "season"
) -> pd.DataFrame:
    """Rows strictly before a season — the only legitimate training data for testing it.

    Exists as a named function because getting this wrong is silent and expensive. The saved
    minutes model is fitted on every season including the one being replayed; using it in a
    backtest lets the largest single driver of points see the future of the gameweeks it is
    scored on. Measured cost of that leak on 2025-26: 185 season points, a 9% overstatement.
    """
    return features[features[season_col] < season]


def holdout_tail(train: pd.DataFrame, frac: float = 0.15, order_col: str = "_idx") -> tuple:
    """Split the most recent slice off a training set for early stopping.

    Taken from the END of the training period, not at random: a validation set drawn from
    the same weeks as training would let early stopping tune against contemporaneous
    information and overstate how long to train.
    """
    ordered = train.sort_values(order_col)
    cut = int(len(ordered) * (1 - frac))
    return ordered.iloc[:cut], ordered.iloc[cut:]
