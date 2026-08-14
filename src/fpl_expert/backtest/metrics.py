"""Scoring rules and calibration checks for probabilistic forecasts.

Accuracy is close to worthless here. A model that always predicts "0 minutes" scores 58%
accuracy on this dataset and is useless, because every downstream decision needs the
*probability*, not the modal class. So the headline metrics are proper scoring rules —
log loss and Brier — which are minimised only by honest probabilities, plus an explicit
calibration table, because a model can score well overall while being systematically
overconfident in the range that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15


def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass log loss. Punishes confident mistakes hardest."""
    proba = np.clip(np.asarray(proba, float), EPS, 1 - EPS)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return float(-np.log(proba[np.arange(len(y_true)), np.asarray(y_true, int)]).mean())


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot outcome."""
    proba = np.asarray(proba, float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), np.asarray(y_true, int)] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float((np.asarray(proba).argmax(axis=1) == np.asarray(y_true, int)).mean())


def calibration_table(
    y_true: np.ndarray, probs: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed frequency for one class, in probability bins.

    The column that matters is `gap`. A model can post a good log loss overall and still
    be badly miscalibrated in a narrow band — and in FPL the band that matters most is the
    high end, where captaincy and transfer decisions get made.
    """
    probs = np.asarray(probs, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": probs, "y": np.asarray(y_true, float)})
    out = df.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    out["gap"] = out["predicted"] - out["observed"]
    return out.reset_index()


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    """Sample-weighted mean absolute calibration gap."""
    table = calibration_table(y_true, probs, bins)
    return float((table["gap"].abs() * table["n"]).sum() / table["n"].sum())


def compare(y_true: np.ndarray, candidates: dict[str, np.ndarray]) -> pd.DataFrame:
    """Score several forecasters on the same outcomes, best log loss first."""
    rows = [
        {
            "model": name,
            "log_loss": log_loss(y_true, proba),
            "brier": brier_score(y_true, proba),
            "accuracy": accuracy(y_true, proba),
            # Calibration of the decision-relevant class: playing 60+ minutes.
            "ece_60plus": expected_calibration_error(
                (np.asarray(y_true, int) == 2).astype(float), np.asarray(proba)[:, 2]
            ),
        }
        for name, proba in candidates.items()
    ]
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
