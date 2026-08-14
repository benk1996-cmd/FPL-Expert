"""Forecast validation: per-component accuracy, and the closing line as a benchmark.

Two complementary views, because they answer different questions.

*Component accuracy* asks whether each sub-model predicts what it claims to. Rank correlation
matters more than error magnitude: FPL decisions are comparisons between players, so a model
that is biased but orders players correctly is more useful than one with lower error and
scrambled ordering.

*Closing-line value* asks whether the match model adds anything over the market — and does so
far faster than outcomes can. Detecting a 7pp clean-sheet bias takes roughly 16 gameweeks of
results but under one gameweek of closing-line comparison. Closing odds postdate the deadline
so they can never be a feature; used as a yardstick they are the most efficient signal
available.

**Discipline note.** Watching closing-line divergence weekly and then tuning the model is
researcher-mediated leakage — the information reaches the model through our hands rather than
through a column, and it leaves no trace in the code. Treat these as monitoring with
pre-committed thresholds, and judge every tuning decision against the season simulator
running on admissible data only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import expected_calibration_error, log_loss

log = logging.getLogger(__name__)

COMPONENT_TARGETS = {
    "expected_points": "actual_points",
    "expected_minutes": "actual_minutes",
}


def component_accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Correlation and error for each forecast column against its realised outcome."""
    rows = []
    for predicted, actual in COMPONENT_TARGETS.items():
        if predicted not in forecasts or actual not in forecasts:
            continue
        frame = forecasts[[predicted, actual]].dropna()
        if frame.empty:
            continue
        rho = spearmanr(frame[predicted], frame[actual]).statistic
        rows.append({
            "component": predicted,
            "n": len(frame),
            "spearman": round(float(rho), 4),
            "mae": round(float((frame[predicted] - frame[actual]).abs().mean()), 4),
            "bias": round(float((frame[predicted] - frame[actual]).mean()), 4),
        })
    return pd.DataFrame(rows)


def top_n_precision(forecasts: pd.DataFrame, n: int = 20, by: str = "gw") -> pd.DataFrame:
    """How many of each gameweek's genuinely best players the model ranked in its top N.

    Closer to how the system is actually used than a global error metric: the optimiser only
    ever looks at the top of the ranking.
    """
    rows = []
    for gw, frame in forecasts.groupby(by):
        frame = frame.dropna(subset=["expected_points", "actual_points"])
        if len(frame) < n:
            continue
        predicted = set(frame.nlargest(n, "expected_points").index)
        realised = set(frame.nlargest(n, "actual_points").index)
        rows.append({by: gw, "hits": len(predicted & realised), "precision": len(predicted & realised) / n})
    return pd.DataFrame(rows)


def closing_line_comparison(
    predictions: pd.DataFrame, closing: pd.DataFrame
) -> pd.DataFrame:
    """Score our match probabilities against the closing line on the same fixtures.

    `predictions` needs `season`, `date`, `home_team`, `away_team`, `p_home`, `p_draw`,
    `p_away` plus realised goals; `closing` supplies `p_*_close`.
    """
    keys = ["season", "date", "home_team", "away_team"]
    merged = predictions.merge(closing, on=keys, how="inner", suffixes=("", "_c"))
    merged = merged.dropna(subset=["p_home", "p_home_close"])
    if merged.empty:
        return pd.DataFrame()

    outcome = np.where(
        merged["home_goals"] > merged["away_goals"], 0,
        np.where(merged["home_goals"] == merged["away_goals"], 1, 2),
    )
    ours = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    theirs = merged[["p_home_close", "p_draw_close", "p_away_close"]].to_numpy()

    divergence = merged["p_home"] - merged["p_home_close"]
    return pd.DataFrame([
        {
            "n_matches": len(merged),
            "our_log_loss": round(log_loss(outcome, ours), 4),
            "closing_log_loss": round(log_loss(outcome, theirs), 4),
            "edge": round(log_loss(outcome, theirs) - log_loss(outcome, ours), 4),
            "mean_signed_divergence": round(float(divergence.mean()), 4),
            "mean_abs_divergence": round(float(divergence.abs().mean()), 4),
        }
    ])


def calibration_summary(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Calibration of the decision-relevant probabilities the pipeline emits."""
    rows = []
    for column, outcome, label in (
        ("p_long", None, "P(60+ minutes)"),
        ("p_appear", None, "P(any minutes)"),
    ):
        if column not in forecasts or "actual_minutes" not in forecasts:
            continue
        frame = forecasts[[column, "actual_minutes"]].dropna()
        realised = (
            (frame["actual_minutes"] >= 60) if column == "p_long"
            else (frame["actual_minutes"] > 0)
        ).astype(float)
        rows.append({
            "probability": label,
            "n": len(frame),
            "predicted": round(float(frame[column].mean()), 4),
            "observed": round(float(realised.mean()), 4),
            "ece": round(expected_calibration_error(realised.to_numpy(), frame[column].to_numpy()), 4),
        })
    return pd.DataFrame(rows)
