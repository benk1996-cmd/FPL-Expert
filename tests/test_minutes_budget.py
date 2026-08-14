"""The team minutes budget: eleven players, ninety minutes, a constraint the model cannot see.

The per-player minutes model scores everyone independently, so nothing stops a squad's expected
minutes summing to anything. Live GW1 frames run from 399 (a thin promoted squad) to 1236 (one
carrying 36 registered players) against a ceiling of 990. The error tracks SQUAD SIZE, not
injuries — 36 small probabilities accumulate where only eleven can play.

Off by default. It improves the forecast (player MAE 15.02 -> 14.84, bias -0.43 -> +0.14) but
its effect on season points flips sign between seasons (-90 / +77 / +51), so it is published as
an alternative view rather than adopted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.minutes import TEAM_MINUTES_BUDGET, balance_team_minutes


def _squad(club: str, n: int, p_long: float, p_short: float = 0.15, gated: int = 0):
    rows = []
    for i in range(n):
        out = i < gated
        rows.append({
            "team": club,
            "p_short": 0.0 if out else p_short,
            "p_long": 0.0 if out else p_long,
        })
    frame = pd.DataFrame(rows)
    frame["p_zero"] = 1 - frame["p_short"] - frame["p_long"]
    frame["expected_minutes"] = frame["p_short"] * 22 + frame["p_long"] * 85
    return frame


def _totals(balanced, team):
    return balanced.groupby(team)["expected_minutes"].sum()


def test_an_oversized_squad_is_scaled_down():
    """36 registered players cannot play 13.7 players' worth of football."""
    frame = _squad("Big", 36, 0.45)
    assert frame["expected_minutes"].sum() > 1400

    balanced = balance_team_minutes(frame, frame["team"])
    assert balanced["expected_minutes"].sum() == pytest.approx(TEAM_MINUTES_BUDGET, abs=2)


def test_a_thin_squad_is_scaled_up():
    frame = _squad("Thin", 18, 0.35)
    assert frame["expected_minutes"].sum() < 700

    balanced = balance_team_minutes(frame, frame["team"])
    assert balanced["expected_minutes"].sum() == pytest.approx(TEAM_MINUTES_BUDGET, abs=2)


def test_teams_are_balanced_independently():
    frame = pd.concat([_squad("Big", 36, 0.45), _squad("Thin", 18, 0.35)], ignore_index=True)
    totals = _totals(balance_team_minutes(frame, frame["team"]), frame["team"])
    assert totals.loc["Big"] == pytest.approx(TEAM_MINUTES_BUDGET, abs=2)
    assert totals.loc["Thin"] == pytest.approx(TEAM_MINUTES_BUDGET, abs=2)


def test_unavailable_players_are_never_given_minutes():
    """The guard that matters. An injured player is not a candidate for the minutes his fit
    team-mates give up — otherwise the availability gate would be silently undone."""
    frame = _squad("Gated", 25, 0.5, gated=8)
    balanced = balance_team_minutes(frame, frame["team"])

    assert (balanced.head(8)["expected_minutes"] == 0).all()
    assert (balanced.head(8)["p_appear"] == 0).all()
    assert balanced["expected_minutes"].sum() == pytest.approx(TEAM_MINUTES_BUDGET, abs=2)


def test_probabilities_stay_valid():
    frame = _squad("Big", 36, 0.45)
    balanced = balance_team_minutes(frame, frame["team"])

    assert (balanced["p_short"] + balanced["p_long"] <= 1.0 + 1e-9).all()
    assert (balanced["p_zero"] >= -1e-9).all()
    assert np.allclose(
        balanced["p_zero"] + balanced["p_short"] + balanced["p_long"], 1.0, atol=1e-9
    )


def test_the_shape_of_a_players_distribution_is_preserved():
    """Only his likelihood of featuring moves, not the split between a cameo and a full match —
    which is what keeps appearance, clean-sheet and defcon points internally consistent."""
    frame = _squad("Big", 36, 0.45)
    balanced = balance_team_minutes(frame, frame["team"])

    before = frame["p_short"] / frame["p_long"]
    after = balanced["p_short"] / balanced["p_long"]
    assert np.allclose(before, after)


def test_a_saturated_squad_cannot_be_pushed_past_certainty():
    """Eleven nailed starters already fill the budget. Eleven players who are certain to play
    90 minutes exceed it, and the rescale must not produce probabilities above one."""
    frame = _squad("AllNailed", 11, p_long=1.0, p_short=0.0)
    balanced = balance_team_minutes(frame, frame["team"])

    assert (balanced["p_long"] <= 1.0 + 1e-9).all()
    assert (balanced["p_zero"] >= -1e-9).all()


def test_derived_columns_are_recomputed_not_left_stale():
    """`expected_minutes`, `p_appear` and `expected_appearance_points` all depend on the bucket
    probabilities. Rescaling one without the others would leave the frame self-contradictory."""
    frame = _squad("Big", 36, 0.45)
    balanced = balance_team_minutes(frame, frame["team"])

    assert np.allclose(
        balanced["expected_minutes"], balanced["p_short"] * 22 + balanced["p_long"] * 85
    )
    assert np.allclose(balanced["p_appear"], balanced["p_short"] + balanced["p_long"])
    assert np.allclose(
        balanced["expected_appearance_points"],
        balanced["p_short"] * 1 + balanced["p_long"] * 2,
    )


def test_an_already_correct_squad_is_left_alone():
    frame = _squad("Fine", 20, 0.0)
    frame.loc[:10, "p_long"] = 1.0
    frame.loc[:10, "p_short"] = 0.0
    frame["expected_minutes"] = frame["p_short"] * 22 + frame["p_long"] * 85
    total = frame["expected_minutes"].sum()

    balanced = balance_team_minutes(frame, frame["team"])
    assert balanced["expected_minutes"].sum() == pytest.approx(total, rel=0.15)
