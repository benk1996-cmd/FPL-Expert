"""Rank-aware decisions: beating the field rather than maximising points.

The premise these tests are built around, and the reason the module is narrow: for a TOTAL
POINTS objective the armband is already solved. Doubling is linear, so the highest mean wins
whatever the distribution looks like. Everything here is about the rank objective, where that
stops being true.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.optimise.rank_objective import (
    RankCaptainSelector,
    evaluate_captains,
    expected_beat_rate,
)


def test_beat_rate_is_the_share_of_the_field_below_us():
    field = np.arange(100, dtype=float)
    assert expected_beat_rate(np.array([50.0]), field) == pytest.approx(0.50, abs=0.01)
    assert expected_beat_rate(np.array([99.0]), field) == pytest.approx(0.99, abs=0.01)


def test_beat_rate_averages_over_our_own_uncertainty():
    """Our score is a distribution too, so a single point estimate would overstate how
    confidently we sit anywhere."""
    field = np.arange(100, dtype=float)
    spread = np.array([10.0, 90.0])
    assert expected_beat_rate(spread, field) == pytest.approx(0.50, abs=0.02)


# --- what a mean cannot say ------------------------------------------------


def _two_captains(n_draws=20_000, base=55.0, field_mean=60.0, seed=0):
    """Two candidates with the SAME mean, one steady and one all-or-nothing."""
    rng = np.random.default_rng(seed)
    steady = rng.normal(6.0, 1.0, n_draws)
    volatile = np.where(rng.random(n_draws) < 0.25, 24.0, 0.0)
    week = np.column_stack([np.full(n_draws, base), steady, volatile])
    field = rng.normal(field_mean, 18.0, (n_draws, 500))
    return week, field


def test_equal_means_are_not_equal_for_rank():
    """The whole justification for the module. Two captains a points objective cannot tell
    apart are materially different once the objective is beating other managers."""
    week, field = _two_captains()
    table = evaluate_captains({1: 1, 2: 2}, {0: 0}, week, field)

    means = table.set_index("player_id")["expected_points"]
    assert means[1] == pytest.approx(means[2], abs=0.1)      # identical to a points objective
    rates = table.set_index("player_id")["beat_rate"]
    assert abs(rates[1] - rates[2]) > 0.005                  # not identical for rank


def test_a_leader_prefers_the_steady_captain_and_a_chaser_the_volatile_one():
    """The direction is conditional, which is exactly what a fixed ownership discount could
    never express — it would give the same advice in GW3 and GW38."""
    ahead_week, ahead_field = _two_captains(base=55.0, field_mean=52.0)
    ahead = evaluate_captains({1: 1, 2: 2}, {0: 0}, ahead_week, ahead_field)
    assert ahead.iloc[0]["player_id"] == 1                   # steady protects a lead

    behind_week, behind_field = _two_captains(base=55.0, field_mean=75.0)
    behind = evaluate_captains({1: 1, 2: 2}, {0: 0}, behind_week, behind_field)
    assert behind.iloc[0]["player_id"] == 2                  # only a gamble catches up


def test_sharing_the_draws_cancels_a_player_the_field_also_owns():
    """Why simulation beats an ownership heuristic: when the field holds the player being
    evaluated, his good weeks lift them as much as us and drop out of the difference. Nothing
    has to be assumed about how much a template captain is worth."""
    rng = np.random.default_rng(1)
    n = 20_000
    template = rng.normal(8.0, 6.0, n)
    differential = rng.normal(8.0, 6.0, n)
    week = np.column_stack([np.full(n, 50.0), template, differential])

    # A field that starts (and so is moved by) the template player, but not the differential.
    field = rng.normal(58.0, 10.0, (n, 400)) + template[:, None]
    table = evaluate_captains({1: 1, 2: 2}, {0: 0}, week, field).set_index("player_id")

    assert table.loc[1, "expected_points"] == pytest.approx(
        table.loc[2, "expected_points"], abs=0.2
    )
    # Captaining what the field already has moves you with them; the differential does not.
    assert table.loc[2, "beat_rate"] > table.loc[1, "beat_rate"]


def test_the_points_pick_is_the_reference_row():
    week, field = _two_captains()
    week[:, 1] += 3.0                                        # candidate 1 now clearly better
    table = evaluate_captains({1: 1, 2: 2}, {0: 0}, week, field).set_index("player_id")
    assert table.loc[1, "points_delta"] == pytest.approx(0.0)
    assert table.loc[1, "beat_rate_delta"] == pytest.approx(0.0)
    assert table.loc[2, "points_delta"] < 0


def test_no_candidates_gives_an_empty_table():
    assert evaluate_captains({}, {}, np.zeros((2, 2)), np.zeros((2, 3))).empty


# --- the selector ----------------------------------------------------------


class _Trace:
    """Minimal stand-in for a FieldTrace."""

    def __init__(self, n_managers=200, n_players=6, gameweeks=(1, 2)):
        self.player_ids = np.arange(n_players)
        self.gameweeks = list(gameweeks)
        self.n_managers = n_managers
        rng = np.random.default_rng(0)
        self.squad = rng.integers(0, n_players, (len(gameweeks), n_managers, 3))
        self.starting = np.ones_like(self.squad, dtype=bool)
        self.captain = self.squad[:, :, 0]

    def score_gameweek(self, week_points, step):
        held = week_points[self.squad[step]]
        return (held * self.starting[step]).sum(axis=1) + week_points[self.captain[step]]


class _Distribution:
    def __init__(self, means, n_players):
        self.means = np.asarray(means, dtype=float)
        self.n = n_players

    def sample(self, rng, draws=1):
        return rng.poisson(self.means, size=(draws, len(self.means))).astype(float)


def _selector(min_points_delta=1.0, n_draws=60):
    trace = _Trace()
    keys = pd.DataFrame({"player_id": np.arange(6)})
    distributions = {
        gw: (keys, _Distribution([2, 3, 4, 5, 6, 7], 6)) for gw in trace.gameweeks
    }
    realised = np.tile(np.arange(6, dtype=float), (2, 1))
    return RankCaptainSelector(
        trace, distributions, realised, n_draws=n_draws, seed=0,
        min_points_delta=min_points_delta,
    )


def _held():
    return pd.DataFrame({
        "player_id": [3, 4, 5], "expected_points": [5.0, 6.0, 7.0], "position": "MID",
    })


def test_selector_returns_a_player_from_the_squad():
    chosen = _selector()(1, _held(), "expected_points", 0.0)
    assert chosen in set(_held()["player_id"])


def test_selector_falls_back_to_expected_points_for_an_unknown_gameweek():
    """A gameweek the field was not built for must not silently produce a broken armband."""
    chosen = _selector()(99, _held(), "expected_points", 0.0)
    assert chosen == 5                                       # the highest expected points


def test_selector_handles_an_empty_squad():
    assert _selector()(1, _held().head(0), "expected_points", 0.0) is None


def test_the_guard_rail_stops_it_giving_away_points_for_noise():
    """A pure beat-rate objective will hand the armband to a clearly worse player for a
    third-decimal gain that a few hundred draws cannot even resolve."""
    strict = _selector(min_points_delta=0.0)(1, _held(), "expected_points", 0.0)
    assert strict == 5              # only the points-optimal pick clears a zero allowance


def test_the_field_baseline_advances_each_gameweek():
    """The selector has to know whether it is protecting a lead, so the field's running total
    must move with the season rather than sitting at zero."""
    selector = _selector()
    assert not selector.field_baseline.any()
    selector(1, _held(), "expected_points", 0.0)
    assert selector.field_baseline.any()
