"""Secondary components, bonus, and the points assembler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.bonus import expected_bonus
from fpl_expert.models.secondary import (
    defensive_contribution_points,
    defensive_points,
    expected_card_points,
    expected_save_points,
    expected_saves,
)

THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}


# --- saves ----------------------------------------------------------------


def test_save_points_use_the_distribution_not_the_mean():
    """One point per THREE saves. A keeper expected to make 2.8 saves scores zero if you
    apply the rule to the mean, but clears three saves over half the time."""
    naive = 2.8 // 3
    modelled = expected_save_points(pd.Series([2.8])).iloc[0]
    assert naive == 0
    assert modelled > 0.5


def test_save_points_increase_with_workload():
    points = expected_save_points(pd.Series([1.0, 3.0, 6.0]))
    assert points.iloc[0] < points.iloc[1] < points.iloc[2]
    assert points.iloc[2] == pytest.approx(2.0, abs=0.4)   # ~6 saves -> ~2 points


def test_saves_scale_with_opponent_threat():
    """A keeper facing the best attack must not be forecast the same workload as one facing
    the worst — save points and clean-sheet points pull in opposite directions."""
    saves = expected_saves(
        pd.Series([3.0, 3.0]), pd.Series([90.0, 90.0]), pd.Series([2.5, 0.7])
    )
    assert saves.iloc[0] > saves.iloc[1]


# --- defensive contributions ---------------------------------------------


def test_defcon_is_a_threshold_probability_not_a_rate():
    """Defenders need 10 CBIT for 2 points, capped once per match. A player averaging 9 is
    not 'nearly there' in a linear sense — the answer is P(reaching 10)."""
    points = defensive_contribution_points(
        pd.Series([9.0]), pd.Series([90.0]), pd.Series(["DEF"]), THRESHOLDS
    )
    assert 0 < points.iloc[0] < 2.0


def test_defcon_never_exceeds_the_cap():
    points = defensive_contribution_points(
        pd.Series([40.0]), pd.Series([90.0]), pd.Series(["DEF"]), THRESHOLDS
    )
    assert points.iloc[0] <= 2.0


def test_defcon_thresholds_differ_by_position():
    """Defenders need 10, midfielders 12 — the same rate is worth more to a defender."""
    rate, minutes = pd.Series([11.0, 11.0]), pd.Series([90.0, 90.0])
    points = defensive_contribution_points(rate, minutes, pd.Series(["DEF", "MID"]), THRESHOLDS)
    assert points.iloc[0] > points.iloc[1]


def test_goalkeepers_score_no_defensive_contribution():
    points = defensive_contribution_points(
        pd.Series([15.0]), pd.Series([90.0]), pd.Series(["GK"]), THRESHOLDS
    )
    assert points.iloc[0] == 0.0


def test_defcon_scales_with_minutes():
    rate = pd.Series([12.0, 12.0])
    points = defensive_contribution_points(
        rate, pd.Series([90.0, 20.0]), pd.Series(["DEF", "DEF"]), THRESHOLDS
    )
    assert points.iloc[0] > points.iloc[1]


# --- clean sheets ---------------------------------------------------------


def test_clean_sheet_requires_sixty_minutes():
    """The join between the match model and the minutes model. A likely substitute must not
    be credited with full clean-sheet value."""
    points = defensive_points(
        p_clean_sheet=pd.Series([0.5, 0.5]),
        p_long=pd.Series([0.95, 0.10]),
        expected_conceded_penalty=pd.Series([0.0, 0.0]),
        position=pd.Series(["DEF", "DEF"]),
        clean_sheet_points={"DEF": 4},
    )
    assert points.iloc[0] == pytest.approx(0.5 * 0.95 * 4)
    assert points.iloc[1] == pytest.approx(0.5 * 0.10 * 4)


def test_conceded_penalty_applies_only_to_keepers_and_defenders():
    points = defensive_points(
        p_clean_sheet=pd.Series([0.0, 0.0]),
        p_long=pd.Series([1.0, 1.0]),
        expected_conceded_penalty=pd.Series([-0.6, -0.6]),
        position=pd.Series(["DEF", "MID"]),
        clean_sheet_points={"DEF": 4, "MID": 1},
    )
    assert points.iloc[0] == pytest.approx(-0.6)
    assert points.iloc[1] == pytest.approx(0.0)


def test_midfielders_get_a_smaller_clean_sheet_reward():
    points = defensive_points(
        pd.Series([0.4, 0.4]), pd.Series([1.0, 1.0]), pd.Series([0.0, 0.0]),
        pd.Series(["DEF", "MID"]), {"DEF": 4, "MID": 1},
    )
    assert points.iloc[0] == pytest.approx(4 * points.iloc[1])


# --- cards ----------------------------------------------------------------


def test_card_points_are_negative_and_scale_with_minutes():
    points = expected_card_points(pd.Series([0.3, 0.3]), pd.Series([90.0, 30.0]))
    assert points.iloc[0] < points.iloc[1] < 0


# --- bonus ----------------------------------------------------------------


def test_bonus_scales_with_team_attacking_strength():
    """Bonus follows goal involvement and concentrates in winning teams."""
    points = expected_bonus(
        pd.Series([0.4, 0.4]), pd.Series([90.0, 90.0]), pd.Series([2.4, 0.8]),
        pd.Series([0.3, 0.3]),
    )
    assert points.iloc[0] > points.iloc[1]


def test_bonus_is_capped_for_thin_samples():
    """Only three players per match earn bonus, so an extreme per-90 rate from a tiny
    sample must not project absurdly."""
    points = expected_bonus(
        pd.Series([9.0]), pd.Series([90.0]), pd.Series([1.43]), pd.Series([0.25])
    )
    assert points.iloc[0] <= 1.6


def test_bonus_is_never_negative():
    points = expected_bonus(
        pd.Series([0.0]), pd.Series([0.0]), pd.Series([0.5]), pd.Series([0.0])
    )
    assert points.iloc[0] >= 0.0


def test_no_minutes_means_no_bonus():
    points = expected_bonus(
        pd.Series([0.5]), pd.Series([0.0]), pd.Series([2.0]), pd.Series([0.3])
    )
    assert points.iloc[0] == pytest.approx(0.0)


# --- availability windows -------------------------------------------------


def test_stat_seasons_excludes_seasons_that_never_recorded_it():
    """Regression: `defensive_contribution` exists only from 2025-26. Including earlier
    seasons zero-fills the counts while still counting their minutes, diluting the rate by
    a factor of several — the exact trap `coverage_report` exists to expose."""
    from fpl_expert.pipeline import _stat_seasons

    history = pd.DataFrame({
        "season": ["2023-24", "2023-24", "2025-26", "2025-26"],
        "defensive_contribution": [np.nan, np.nan, 8.0, 11.0],
        "goals_scored": [1, 0, 2, 1],
    })
    assert _stat_seasons(history, "defensive_contribution") == ["2025-26"]
    assert _stat_seasons(history, "goals_scored") == ["2023-24", "2025-26"]
    assert _stat_seasons(history, "not_a_column") == []
