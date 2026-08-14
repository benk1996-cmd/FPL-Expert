"""Monitoring: catch the model degrading before a season is lost to it.

Deliberately threshold-based rather than eyeballed. The failure mode this guards against is
not a crash — it is a model that keeps producing plausible numbers while drifting away from
reality, which is exactly what happened three separate times while building this system (the
collapsed Kish prior, the xG noise scale, the zero-filled DefCon column). Every one of those
produced output that looked entirely reasonable.

Thresholds are pre-committed so that reacting to a breach is a decision rather than a
rationalisation. See `validate.py` for why weekly eyeballing plus tuning is itself a leak.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

# Breaching any of these should trigger investigation, not an immediate refit.
THRESHOLDS = {
    "minutes_ece": 0.05,          # calibration of P(60+); walk-forward measured 0.012
    "points_spearman_min": 0.25,  # rank correlation of forecast vs realised points
    "points_bias_abs": 1.0,       # systematic over/under-forecasting, points per player
    "clean_sheet_divergence": 0.08,   # mean signed gap vs the closing line
}


def check(
    forecasts: pd.DataFrame,
    *,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Evaluate the standing health checks and report pass/fail per metric."""
    from .validate import calibration_summary, component_accuracy

    thresholds = {**THRESHOLDS, **(thresholds or {})}
    rows = []

    calibration = calibration_summary(forecasts)
    if not calibration.empty:
        row = calibration[calibration["probability"] == "P(60+ minutes)"]
        if not row.empty:
            value = float(row["ece"].iloc[0])
            rows.append(_row("minutes_ece", value, thresholds["minutes_ece"], "below"))

    accuracy = component_accuracy(forecasts)
    points = accuracy[accuracy["component"] == "expected_points"]
    if not points.empty:
        rows.append(_row(
            "points_spearman", float(points["spearman"].iloc[0]),
            thresholds["points_spearman_min"], "above",
        ))
        rows.append(_row(
            "points_bias", abs(float(points["bias"].iloc[0])),
            thresholds["points_bias_abs"], "below",
        ))

    return pd.DataFrame(rows)


def _row(name: str, value: float, threshold: float, direction: str) -> dict:
    ok = value <= threshold if direction == "below" else value >= threshold
    return {
        "metric": name,
        "value": round(value, 4),
        "threshold": threshold,
        "direction": direction,
        "status": "ok" if ok else "BREACH",
    }


def retraining_due(snapshots: pd.DataFrame, every_gameweeks: int = 6) -> tuple[bool, str]:
    """Whether the models are stale enough to warrant a refit.

    Refitting every week would chase noise and make results irreproducible between runs;
    never refitting ignores a whole season of new data. A fixed cadence is the honest
    compromise, and it is recorded rather than left to judgement.
    """
    if snapshots.empty:
        return False, "no snapshots recorded yet"
    completed = snapshots["target_gw"].dropna()
    if completed.empty:
        return False, "no completed gameweeks"
    latest = int(completed.max())
    due = latest > 0 and latest % every_gameweeks == 0
    return due, f"GW{latest}: refit every {every_gameweeks} gameweeks -> {'DUE' if due else 'not due'}"
