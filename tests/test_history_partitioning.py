"""Refreshing one season must not delete the others.

Regression test: `ingest_history` originally wrote a single combined table, so
`fpl history -s 2025-26` silently replaced all seven seasons with one.
"""

from __future__ import annotations

import pandas as pd

from fpl_expert.data.historical import load_history
from fpl_expert.data.storage import write_table


def test_partial_refresh_preserves_other_seasons(tmp_config):
    write_table(pd.DataFrame({"season": ["2024-25"], "GW": [1], "total_points": [5]}),
                "interim", "history", season="2024-25")
    write_table(pd.DataFrame({"season": ["2025-26"], "GW": [1], "total_points": [7]}),
                "interim", "history", season="2025-26")

    # Re-write just one season, as a targeted refresh would.
    write_table(pd.DataFrame({"season": ["2025-26"], "GW": [1], "total_points": [9]}),
                "interim", "history", season="2025-26")

    both = load_history()
    assert sorted(both["season"].unique()) == ["2024-25", "2025-26"]
    assert both.loc[both["season"] == "2025-26", "total_points"].iloc[0] == 9  # refreshed
    assert both.loc[both["season"] == "2024-25", "total_points"].iloc[0] == 5  # untouched


def test_load_history_unions_columns_across_seasons(tmp_config):
    """A season predating a stat must read back as NaN, never 0."""
    write_table(pd.DataFrame({"season": ["2021-22"], "total_points": [5]}),
                "interim", "history", season="2021-22")
    write_table(pd.DataFrame({"season": ["2025-26"], "total_points": [7],
                              "defensive_contribution": [2]}),
                "interim", "history", season="2025-26")

    df = load_history()
    old = df.loc[df["season"] == "2021-22", "defensive_contribution"].iloc[0]
    assert pd.isna(old)
