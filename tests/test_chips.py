"""Chip valuation and the optimal-stopping policy."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.optimise.chips import (
    active_window,
    bench_boost_value,
    chip_windows,
    expected_best_remaining,
    free_hit_value,
    should_play,
    triple_captain_value,
    wildcard_value,
)

RULES = {
    "chips": {
        "wildcard": {"count": 2, "windows": [[2, 19], [20, 38]]},
        "free_hit": {"count": 2, "windows": [[2, 19], [20, 38]]},
        "bench_boost": {"count": 2, "windows": [[1, 19], [20, 38]]},
        "triple_captain": {"count": 2, "windows": [[1, 19], [20, 38]]},
    }
}


# --- values ---------------------------------------------------------------


def test_bench_boost_is_worth_the_bench_only():
    squad = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "expected_points": [6.0, 5.0, 2.0, 1.5],
    })
    assert bench_boost_value(squad, {3, 4}, "expected_points") == pytest.approx(3.5)


def test_triple_captain_adds_one_copy_not_three():
    """The armband already doubles. Valuing the chip at 3x the score overstates it threefold
    and burns it on a good-but-not-exceptional week."""
    xi = pd.DataFrame({"player_id": [1, 2], "expected_points": [9.0, 5.0]})
    assert triple_captain_value(xi, 1, "expected_points") == pytest.approx(9.0)


def test_triple_captain_falls_back_to_the_best_starter():
    xi = pd.DataFrame({"player_id": [1, 2], "expected_points": [9.0, 5.0]})
    assert triple_captain_value(xi, 999, "expected_points") == pytest.approx(9.0)


def test_free_hit_value_is_the_gain_and_never_negative():
    assert free_hit_value(40.0, 62.0) == pytest.approx(22.0)
    assert free_hit_value(60.0, 45.0) == 0.0


def test_wildcard_counts_the_hits_it_saves():
    """Unlike a free hit the new squad persists, so it is valued over the horizon plus the
    hits you would otherwise have paid one at a time."""
    assert wildcard_value(100.0, 118.0, hits_avoided=2) == pytest.approx(26.0)


# --- windows --------------------------------------------------------------


def test_two_windows_per_chip_none_carrying_over():
    windows = chip_windows(RULES)
    assert len(windows["wildcard"]) == 2
    assert windows["wildcard"][0] == (2, 19)


def test_wildcard_cannot_be_played_in_gw1_but_bench_boost_can():
    """A verified rules quirk: Wildcard and Free Hit open at GW2, the other two at GW1."""
    windows = chip_windows(RULES)
    assert active_window(windows["wildcard"], 1) is None
    assert active_window(windows["bench_boost"], 1) == (1, 19)


def test_first_half_chips_expire_at_gw19():
    windows = chip_windows(RULES)
    assert active_window(windows["bench_boost"], 19) == (1, 19)
    assert active_window(windows["bench_boost"], 20) == (20, 38)


# --- stopping policy ------------------------------------------------------


def test_holds_when_a_better_week_is_visible():
    play, reason = should_play("bench_boost", 12.0, [25.0, 14.0], weeks_to_expiry=10)
    assert not play and "holding" in reason


def test_plays_when_this_is_the_best_remaining_week():
    play, _ = should_play("bench_boost", 30.0, [12.0, 14.0], weeks_to_expiry=10)
    assert play


def test_refuses_a_weak_week_even_if_it_is_the_best_visible():
    """A chip is not worth burning just because nothing better is in the forecast window."""
    play, reason = should_play("wildcard", 3.0, [2.0], weeks_to_expiry=15)
    assert not play and "floor" in reason


def test_plays_at_expiry_rather_than_wasting_the_chip():
    """A chip worth 15 now beats a chip worth 25 you never get to use."""
    play, reason = should_play("bench_boost", 5.0, [], weeks_to_expiry=0)
    assert play and "final chance" in reason


def test_unseen_future_weeks_are_not_treated_as_nonexistent():
    """Forecasts reach a few gameweeks; beyond that the policy must still believe better
    weeks exist, or it fires almost immediately and burns the chip early."""
    ignoring = expected_best_remaining([10.0], unknown_weeks=0, unknown_estimate=12.0)
    accounting = expected_best_remaining(
        [10.0], unknown_weeks=15, unknown_estimate=12.0, chip="bench_boost"
    )
    assert accounting > ignoring


def test_double_gameweek_expectation_makes_the_policy_patient():
    """Bench Boost roughly doubles in a double gameweek. Without that expectation the policy
    played it in GW2 of 2025-26 for +21 when GW33 was worth +144."""
    typical = 15.0
    play, _ = should_play(
        "bench_boost", 20.0, [], unknown_weeks=20, unknown_estimate=typical,
        weeks_to_expiry=17,
    )
    assert not play          # 20 does not beat an expected ~2x typical week later


def test_urgency_overrides_patience_near_expiry():
    play, reason = should_play(
        "bench_boost", 20.0, [], unknown_weeks=0, unknown_estimate=15.0,
        weeks_to_expiry=1,
    )
    assert play and "expiring" in reason
