"""Post-hoc calibration of expected points, using ownership as a signal we lack.

Measured across 2023-24 to 2025-26, the forecast under-predicts everyone and under-predicts
the players the field owns most by nearly three times as much:

    ownership   predicted   actual    bias
    <5%              2.00     2.27   -0.28
    15-30%           3.32     3.84   -0.53
    >50%             4.80     6.05   -1.25

That is not simple compression — a straight regression of actual on predicted gives a slope
of 0.923 with a positive intercept, and highly-owned players still beat that adjusted line.
Adding `log(1 + ownership)` carries a positive coefficient in every season tested
(0.383 / 0.301 / 0.238) and lifts R^2 by roughly 7.5% relative.

**The crowd knows something the model does not.** Millions of managers aggregate scout
reports, press conferences, eye-test and rotation intelligence that none of our features
capture. Ownership is published before every deadline, so using it is legitimate rather
than hindsight.

Two things this is NOT:

*It is not blindly following the crowd.* The ownership term sits alongside our own forecast,
so a player the model rates highly and the field ignores still scores well. It corrects a
measured bias; it does not replace the model.

*It is not a rank strategy.* Raising `lambda_rank` — deliberately favouring differentials —
made both points and rank worse, because being contrarian without edge is pure variance.
This pulls the opposite way, toward the template, and does so because the template is
genuinely underpriced by our forecasts rather than as a rank tactic.

Fitting must be walk-forward: the coefficients for a season come only from seasons before it.

## MEASURED RESULT: this makes decisions WORSE. Off by default.

Applied walk-forward through the season simulator, it cost **83 season points** and moved the
mean simulated rank from 9,656 to 51,822:

    variant       2023-24  2024-25  2025-26    mean    rank
    raw              2701     2722     2485    2636    9,656
    calibrated       2657     2633     2368    2553   51,822

The bias it corrects is real, consistent and statistically clear. It is still not exploitable,
for three reasons worth remembering before attempting this again:

1. **A regression coefficient is not a per-pound coefficient.** Highly-owned players are
   expensive. Raising their scores makes the optimiser spend budget on them, and the fitted
   bias says nothing about whether that spend beats the alternative.
2. **The signal is small and the distortion is not.** Ownership lifted R^2 by 0.007 on a base
   of 0.09. That is a real but slight improvement in *prediction*, applied to a *ranking*
   problem where it reshuffles the whole selection.
3. **Walk-forward coefficients are larger than in-sample ones** (0.44-0.65 here against
   0.24-0.38 fitted within-season), so the correction applied in practice is stronger than
   the diagnostic that motivated it.

This is the second time a principled fix for a measured bias made things worse — the first
was allocating exactly six bonus points per fixture. The generalisable lesson: **a bias that
is real in a regression is not necessarily exploitable inside a constrained optimiser.**

The module is kept for `bias_by_ownership`, which remains a useful standing diagnostic, and
so the negative result is reproducible rather than folklore.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Players expected to barely feature add noise without informing the fit — their actual
# points are dominated by whether they got on at all.
MIN_EXPECTED_MINUTES = 20.0


@dataclass
class OwnershipCalibrator:
    """Linear correction: actual ~ intercept + slope * predicted + ownership_coef * log1p(eo)."""

    intercept: float = 0.0
    slope: float = 1.0
    ownership_coef: float = 0.0
    fitted_on: tuple[str, ...] = ()

    @property
    def uses_ownership(self) -> bool:
        return self.ownership_coef != 0.0

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        points_col: str = "expected_points",
        actual_col: str = "actual_points",
        ownership_col: str = "eo",
    ) -> OwnershipCalibrator:
        """Fit on OUTCOMES, so this may only ever see seasons before the one being predicted."""
        data = frame.dropna(subset=[points_col, actual_col, ownership_col])
        if "expected_minutes" in data.columns:
            data = data[data["expected_minutes"] > MIN_EXPECTED_MINUTES]
        if len(data) < 100:
            log.warning("too little data to calibrate (%d rows); leaving as identity", len(data))
            return self

        design = np.column_stack([
            np.ones(len(data)),
            data[points_col].to_numpy(float),
            np.log1p(data[ownership_col].clip(lower=0).to_numpy(float)),
        ])
        coefficients, *_ = np.linalg.lstsq(design, data[actual_col].to_numpy(float), rcond=None)
        self.intercept, self.slope, self.ownership_coef = (float(c) for c in coefficients)
        if "season" in data.columns:
            self.fitted_on = tuple(sorted(data["season"].unique()))
        log.info(
            "calibrated on %s: intercept %.3f slope %.3f ownership %.3f",
            list(self.fitted_on) or "unknown", self.intercept, self.slope, self.ownership_coef,
        )
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        points_col: str = "expected_points",
        ownership_col: str = "eo",
        out_col: str = "calibrated_points",
    ) -> pd.DataFrame:
        """Apply the correction. Falls back to the raw forecast where ownership is missing."""
        out = frame.copy()
        predicted = out[points_col].fillna(0.0).to_numpy(float)
        if ownership_col in out.columns:
            ownership = np.log1p(out[ownership_col].fillna(0.0).clip(lower=0).to_numpy(float))
        else:
            ownership = np.zeros(len(out))

        calibrated = self.intercept + self.slope * predicted + self.ownership_coef * ownership
        # A forecast can be small but never negative — a player cannot be expected to lose
        # points merely for being unfashionable.
        out[out_col] = np.clip(calibrated, 0.0, None)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "intercept": self.intercept, "slope": self.slope,
                "ownership_coef": self.ownership_coef, "fitted_on": list(self.fitted_on),
            }),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> OwnershipCalibrator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            intercept=data["intercept"], slope=data["slope"],
            ownership_coef=data["ownership_coef"], fitted_on=tuple(data.get("fitted_on", [])),
        )


def fit_walk_forward(
    frames_by_season: dict[str, pd.DataFrame], season: str, **kwargs
) -> OwnershipCalibrator:
    """Calibrator for `season`, fitted only on seasons that precede it.

    Fitting on the tested season would be straightforward leakage: the correction is
    estimated from the very outcomes it is then scored against.
    """
    prior = [f for s, f in sorted(frames_by_season.items()) if s < season]
    if not prior:
        log.info("no prior seasons for %s; calibration is the identity", season)
        return OwnershipCalibrator()
    return OwnershipCalibrator().fit(pd.concat(prior, ignore_index=True), **kwargs)


def bias_by_ownership(
    frame: pd.DataFrame,
    *,
    points_col: str = "expected_points",
    actual_col: str = "actual_points",
    ownership_col: str = "eo",
    bands: tuple[float, ...] = (0, 5, 15, 30, 50, 200),
) -> pd.DataFrame:
    """Diagnostic table: predicted vs actual by ownership band.

    This is what exposed the problem, and it is worth re-running whenever the forecast
    changes — a model that is well calibrated on average can still be badly wrong on exactly
    the players the field owns.
    """
    data = frame.dropna(subset=[points_col, actual_col, ownership_col])
    if "expected_minutes" in data.columns:
        data = data[data["expected_minutes"] > MIN_EXPECTED_MINUTES]
    labels = [f"{int(a)}-{int(b)}%" for a, b in itertools.pairwise(bands)]
    data = data.assign(band=pd.cut(data[ownership_col], list(bands), labels=labels))

    table = data.groupby("band", observed=True).agg(
        n=(ownership_col, "size"),
        predicted=(points_col, "mean"),
        actual=(actual_col, "mean"),
    )
    table["bias"] = table["predicted"] - table["actual"]
    return table.reset_index()
