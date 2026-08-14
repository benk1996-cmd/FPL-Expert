"""Effective ownership and the skilled-cohort filter. No network access."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.data.ownership import (
    effective_ownership,
    picks_to_frame,
    sample_entries,
    skilled_cohort,
)


def _picks(entry, elements, multipliers, captain):
    return pd.DataFrame({
        "entry": entry, "element": elements, "position": range(1, len(elements) + 1),
        "multiplier": multipliers, "is_captain": [e == captain for e in elements],
    })


# --- the skilled cohort ---------------------------------------------------


def test_cohort_selects_on_past_rank_not_current_form():
    """Early-season rank is noise; a prior top-100k finish is real evidence of skill."""
    past = pd.DataFrame({
        "entry": [1, 1, 2, 3],
        "season": ["2024/25", "2025/26", "2025/26", "2025/26"],
        "rank": [50_000, 8_000, 2_500_000, 90_000],
        "total_points": [2400, 2500, 2000, 2410],
    })
    assert skilled_cohort(past, max_rank=100_000) == [1, 3]


def test_cohort_can_demand_sustained_performance():
    """One good season may be luck; requiring two filters harder."""
    past = pd.DataFrame({
        "entry": [1, 1, 3],
        "season": ["2024/25", "2025/26", "2025/26"],
        "rank": [50_000, 8_000, 90_000],
        "total_points": [2400, 2500, 2410],
    })
    assert skilled_cohort(past, max_rank=100_000, min_seasons=2) == [1]


def test_cohort_ignores_entries_with_no_past_seasons():
    """A brand-new account has no track record and cannot be judged skilled."""
    past = pd.DataFrame({"entry": [1], "season": ["2025/26"], "rank": [None],
                         "total_points": [2400]})
    assert skilled_cohort(past) == []


def test_cohort_on_empty_history():
    assert skilled_cohort(pd.DataFrame()) == []


def test_sample_is_deterministic_and_bounded():
    ids = list(range(1, 1001))
    a, b = sample_entries(ids, 100, seed=7), sample_entries(ids, 100, seed=7)
    assert a == b and len(a) == 100 and len(set(a)) == 100


def test_sample_returns_everything_when_pool_is_small():
    assert sample_entries([1, 2, 3], 100) == [1, 2, 3]


# --- effective ownership --------------------------------------------------


def test_effective_ownership_counts_captaincy_twice():
    """The point of EO: a captained player scores double for that manager, so he weighs
    double against the field. Raw ownership understates exactly the premiums that matter."""
    picks = pd.concat([
        _picks(1, [10, 20], [2, 1], captain=10),     # captains 10
        _picks(2, [10, 20], [2, 1], captain=10),     # captains 10
    ], ignore_index=True)
    eo = effective_ownership(picks).set_index("element")

    assert eo.loc[10, "owned_pct"] == 100.0
    assert eo.loc[10, "captain_pct"] == 100.0
    assert eo.loc[10, "eo"] == 200.0               # owned by all AND captained by all
    assert eo.loc[20, "eo"] == 100.0


def test_benched_players_count_for_ownership_but_not_eo():
    """A benched player scores nothing, so he cannot move you relative to the field."""
    picks = _picks(1, [10, 30], [1, 0], captain=10)
    eo = effective_ownership(picks).set_index("element")

    assert eo.loc[30, "owned_pct"] == 100.0        # in the squad
    assert eo.loc[30, "start_pct"] == 0.0
    assert eo.loc[30, "eo"] == 0.0                 # but contributes nothing


def test_triple_captain_counts_treble():
    picks = _picks(1, [10], [3], captain=10)
    assert effective_ownership(picks).set_index("element").loc[10, "eo"] == 300.0


def test_effective_ownership_is_sorted_by_eo():
    picks = pd.concat([
        _picks(1, [10, 20], [1, 2], captain=20),
        _picks(2, [10, 20], [1, 2], captain=20),
    ], ignore_index=True)
    assert effective_ownership(picks)["element"].tolist() == [20, 10]


def test_effective_ownership_on_empty_picks():
    out = effective_ownership(pd.DataFrame(columns=["entry", "element", "multiplier",
                                                    "is_captain"]))
    assert out.empty and "eo" in out.columns


# --- parsing --------------------------------------------------------------


def test_picks_to_frame_extracts_multipliers_and_chip():
    payload = {
        "active_chip": "3xc",
        "picks": [
            {"element": 10, "position": 1, "multiplier": 3, "is_captain": True},
            {"element": 20, "position": 12, "multiplier": 0, "is_captain": False},
        ],
    }
    df = picks_to_frame(99, payload)
    assert df["entry"].unique().tolist() == [99]
    assert df.loc[df["element"] == 10, "multiplier"].iloc[0] == 3
    assert df["active_chip"].iloc[0] == "3xc"


def test_picks_to_frame_handles_missing_captain_flag():
    df = picks_to_frame(1, {"picks": [{"element": 5, "position": 1, "multiplier": 1}]})
    assert not df["is_captain"].iloc[0]


@pytest.mark.parametrize("payload", [{}, {"picks": []}])
def test_picks_to_frame_on_empty_payload(payload):
    assert picks_to_frame(1, payload).empty
