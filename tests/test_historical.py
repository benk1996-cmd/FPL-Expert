"""Schema-drift handling across seasons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_expert.data.historical import coverage_report


def test_coverage_distinguishes_missing_from_zero():
    """A column absent for a season must report 0.0 coverage, not be read as 'all zeros'.

    This is the guard against training the DefCon model on seasons that predate the stat
    and concluding that nobody makes tackles.
    """
    df = pd.DataFrame({
        "season": ["2021-22"] * 2 + ["2025-26"] * 2,
        "defensive_contribution": [np.nan, np.nan, 0, 2],
        "expected_goals": [np.nan, np.nan, 0.3, 0.1],
    })
    report = coverage_report(df).set_index("season")

    assert report.loc["2021-22", "defensive_contribution"] == 0.0
    # 2025-26 has a genuine zero in it — that is still recorded data, so coverage is full.
    assert report.loc["2025-26", "defensive_contribution"] == 1.0
    assert report.loc["2025-26", "rows"] == 2


def test_coverage_reports_absent_column_as_zero():
    """Older seasons lack the column entirely rather than carrying NaNs."""
    df = pd.DataFrame({"season": ["2019-20"], "total_points": [5]})
    report = coverage_report(df).set_index("season")
    assert report.loc["2019-20", "defensive_contribution"] == 0.0
    assert report.loc["2019-20", "expected_goals"] == 0.0
