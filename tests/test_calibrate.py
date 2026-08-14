"""Ownership calibration: correcting a measured bias without following the crowd blindly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.calibrate import (
    OwnershipCalibrator,
    bias_by_ownership,
    fit_walk_forward,
)


def _frame(n=800, seed=0, ownership_effect=0.4, season="2024-25"):
    """Synthetic data where highly-owned players genuinely outscore the forecast."""
    rng = np.random.default_rng(seed)
    predicted = rng.uniform(1, 8, n)
    ownership = rng.uniform(0, 90, n)
    actual = (
        0.5 + 0.9 * predicted + ownership_effect * np.log1p(ownership) + rng.normal(0, 1, n)
    )
    return pd.DataFrame({
        "season": season, "expected_points": predicted, "eo": ownership,
        "actual_points": actual, "expected_minutes": 80.0,
    })


def test_calibrator_recovers_the_ownership_effect():
    fitted = OwnershipCalibrator().fit(_frame(2000))
    assert fitted.ownership_coef == pytest.approx(0.4, abs=0.08)
    assert fitted.slope == pytest.approx(0.9, abs=0.06)


def test_calibration_raises_highly_owned_players_more():
    """The whole point: the correction is larger where the bias was larger."""
    fitted = OwnershipCalibrator().fit(_frame(2000))
    frame = pd.DataFrame({
        "expected_points": [5.0, 5.0], "eo": [1.0, 80.0], "expected_minutes": [80.0, 80.0],
    })
    out = fitted.transform(frame)
    assert out["calibrated_points"].iloc[1] > out["calibrated_points"].iloc[0]


def test_a_strong_forecast_still_beats_a_popular_weak_one():
    """Not blind crowd-following: our own forecast keeps the larger coefficient, so a player
    the model rates and the field ignores still ranks above a popular mediocrity."""
    fitted = OwnershipCalibrator().fit(_frame(2000))
    frame = pd.DataFrame({
        "expected_points": [8.0, 3.0], "eo": [1.0, 90.0], "expected_minutes": [80.0, 80.0],
    })
    out = fitted.transform(frame)
    assert out["calibrated_points"].iloc[0] > out["calibrated_points"].iloc[1]


def test_calibrated_points_are_never_negative():
    fitted = OwnershipCalibrator(intercept=-5.0, slope=1.0, ownership_coef=0.0)
    out = fitted.transform(pd.DataFrame({"expected_points": [0.1], "eo": [0.0]}))
    assert out["calibrated_points"].iloc[0] >= 0.0


def test_missing_ownership_falls_back_to_the_raw_forecast():
    """Before the season starts there is no ownership history; the layer must degrade."""
    fitted = OwnershipCalibrator(intercept=0.0, slope=1.0, ownership_coef=0.3)
    out = fitted.transform(pd.DataFrame({"expected_points": [4.0]}))
    assert out["calibrated_points"].iloc[0] == pytest.approx(4.0)


def test_too_little_data_leaves_the_calibration_as_identity():
    fitted = OwnershipCalibrator().fit(_frame(20))
    assert fitted.slope == 1.0 and fitted.ownership_coef == 0.0


def test_low_minute_players_are_excluded_from_the_fit():
    """Their actual score is dominated by whether they got on at all, which is noise here."""
    frame = _frame(1000)
    frame.loc[frame.index[:500], "expected_minutes"] = 5.0
    frame.loc[frame.index[:500], "actual_points"] = 99.0     # would wreck an unfiltered fit
    fitted = OwnershipCalibrator().fit(frame)
    assert fitted.slope < 3.0


# --- walk-forward ---------------------------------------------------------


def test_walk_forward_never_fits_on_the_tested_season():
    """Fitting on the season being scored is straightforward leakage — the correction would
    be estimated from the very outcomes it is then judged against."""
    frames = {
        "2023-24": _frame(600, seed=1, season="2023-24"),
        "2024-25": _frame(600, seed=2, season="2024-25"),
        "2025-26": _frame(600, seed=3, ownership_effect=5.0, season="2025-26"),
    }
    fitted = fit_walk_forward(frames, "2025-26")
    assert "2025-26" not in fitted.fitted_on
    assert set(fitted.fitted_on) == {"2023-24", "2024-25"}
    assert fitted.ownership_coef < 2.0        # untouched by the tested season's huge effect


def test_walk_forward_on_the_earliest_season_is_the_identity():
    frames = {"2023-24": _frame(600, season="2023-24")}
    fitted = fit_walk_forward(frames, "2023-24")
    assert fitted.slope == 1.0 and fitted.ownership_coef == 0.0


# --- diagnostics and persistence -----------------------------------------


def test_bias_table_exposes_the_pattern_that_started_this():
    table = bias_by_ownership(_frame(3000)).set_index("band")
    low, high = table.iloc[0], table.iloc[-1]
    assert low["bias"] > high["bias"]        # under-prediction worsens with ownership
    assert (table["n"] > 0).all()


def test_calibrator_round_trips(tmp_path):
    fitted = OwnershipCalibrator(intercept=0.4, slope=0.93, ownership_coef=0.31,
                                 fitted_on=("2023-24",))
    path = tmp_path / "calibration.json"
    fitted.save(path)
    loaded = OwnershipCalibrator.load(path)

    assert loaded.ownership_coef == pytest.approx(0.31)
    assert loaded.fitted_on == ("2023-24",)
    assert loaded.uses_ownership
