"""Attacking returns: allocation, finishing skill, and penalty duty."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.attack import (
    ASSIST_RATE,
    PENALTY_CONVERSION,
    allocate_team_goals,
    attacking_points,
    attacking_rates,
    goal_involvement_pmf,
    penalty_shares,
)


def _totals(rows):
    return pd.DataFrame(rows)


# --- shrinkage ------------------------------------------------------------


def test_finishing_multiplier_is_shrunk_toward_the_league_rate():
    """A player who scored 3 from 1.0 xG has a raw ratio of 3.0. That is noise, and an
    unshrunk multiplier would treble his forecast."""
    totals = _totals([
        {"exposure_90s": 2.0, "xg": 1.0, "xa": 0.2, "goals": 3.0, "position": "FWD"},
        *[{"exposure_90s": 30.0, "xg": 12.0, "xa": 3.0, "goals": 12.0, "position": "FWD"}
          for _ in range(40)],
    ])
    out = attacking_rates(totals)
    assert out["finishing_multiplier"].iloc[0] < 1.6      # nowhere near 3.0
    assert out["finishing_confidence"].iloc[0] < 0.4      # and flagged as thin evidence


def test_player_with_no_history_lands_on_the_group_prior():
    totals = _totals([
        {"exposure_90s": 0.0, "xg": 0.0, "xa": 0.0, "goals": 0.0, "position": "FWD"},
        *[{"exposure_90s": 20.0, "xg": 8.0, "xa": 2.0, "goals": 8.0, "position": "FWD"}
          for _ in range(30)],
    ])
    out = attacking_rates(totals)
    assert out["xg_confidence"].iloc[0] == 0.0
    assert out["xg_per90"].iloc[0] > 0                    # a prior, not a zero


def test_undetectable_skill_collapses_to_the_group_rate_and_says_so():
    """When a population's spread is fully explained by sampling noise, every player should
    land on the group mean — and the prior must report that it detected nothing.

    This is what real finishing data does: measured on 2022-23 onward, the spread in
    goals-per-xG among forwards is SMALLER than Poisson noise alone would produce, so there
    is no demonstrable finishing skill to reward. A silent floor and a genuine signal look
    identical downstream, hence `skill_detected`.
    """
    from fpl_expert.features.rates import fit_gamma_prior

    rng = np.random.default_rng(3)
    xg = np.full(200, 8.0)
    goals = rng.poisson(1.0 * xg)          # every player converts at exactly the same rate
    prior = fit_gamma_prior(goals, xg, noise_scale=1.0)

    assert not prior.skill_detected
    assert prior.mean == pytest.approx(1.0, abs=0.05)


def test_skill_detected_when_it_is_genuinely_there():
    from fpl_expert.features.rates import fit_gamma_prior

    xg = np.full(200, 8.0)
    goals = np.repeat([0.6, 1.4], 100) * xg   # two genuinely different populations
    assert fit_gamma_prior(goals, xg, noise_scale=1.0).skill_detected


def test_prolific_and_blank_players_separate_given_enough_exposure():
    totals = _totals(
        [{"exposure_90s": 30.0, "xg": 18.0, "xa": 3.0, "goals": 18.0, "position": "FWD"}] * 20
        + [{"exposure_90s": 30.0, "xg": 1.5, "xa": 3.0, "goals": 1.0, "position": "FWD"}] * 20
    )
    out = attacking_rates(totals)
    assert out["xg_per90"].iloc[0] > out["xg_per90"].iloc[-1] + 0.2


# --- penalty duty ---------------------------------------------------------


def test_penalty_share_reads_the_live_order():
    players = pd.DataFrame({"penalties_order": [1, 2, np.nan, 3]})
    shares = penalty_shares(players)
    assert shares.iloc[0] > shares.iloc[1] > shares.iloc[3] > 0
    assert shares.iloc[2] == 0.0          # not on penalties


def test_penalty_share_is_zero_when_the_column_is_absent():
    """Archive rows have no set-piece order; the model must degrade, not crash."""
    assert penalty_shares(pd.DataFrame({"x": [1, 2]})).tolist() == [0.0, 0.0]


# --- allocation -----------------------------------------------------------


def _squad():
    return pd.DataFrame({
        "player": ["striker", "winger", "defender", "benched"],
        "xg_per90": [0.60, 0.25, 0.05, 0.50],
        "xa_per90": [0.20, 0.40, 0.05, 0.30],
        "finishing_multiplier": [1.0, 1.0, 1.0, 1.0],
        "expected_minutes": [85.0, 80.0, 90.0, 0.0],
        "penalty_share": [0.85, 0.0, 0.0, 0.0],
    })


def test_allocation_reproduces_the_team_total():
    """The whole point of allocating top-down: player goals must sum to the match model's
    team expectation, so attacking returns and clean sheets cannot contradict each other."""
    out = allocate_team_goals(_squad(), team_expected_goals=2.0)
    assert out["expected_goals"].sum() == pytest.approx(2.0, abs=0.02)


def test_allocation_conserves_the_team_total_when_no_penalty_taker_is_known():
    """The condition every backtest row is in, and where a 6.1% leak hid for months.

    The archive has no set-piece order, so `penalty_share` is 0 throughout. Open play used to
    subtract the team's theoretical penalty total regardless, deleting 0.11/1.43 x 0.79 of
    every team's expected goals and allocating it to nobody. The test above missed it because
    its squad HAS a taker, so almost all the mass was handed back.
    """
    squad = _squad()
    squad["penalty_share"] = 0.0
    out = allocate_team_goals(squad, team_expected_goals=2.0)

    assert out["expected_goals"].sum() == pytest.approx(2.0, rel=1e-6)
    assert out["expected_penalty_goals"].sum() == pytest.approx(0.0)


def test_partial_penalty_duty_still_conserves_the_total():
    """Conservation must hold for any share, not just the two extremes."""
    for share in (0.0, 0.35, 0.85, 1.0):
        squad = _squad()
        squad.loc[squad["player"] == "striker", "penalty_share"] = share
        out = allocate_team_goals(squad, team_expected_goals=1.8)
        assert out["expected_goals"].sum() == pytest.approx(1.8, rel=1e-6)


def test_assist_rate_matches_the_archive():
    """0.72 was an unmeasured estimate and was the whole of the 27% assist under-prediction.
    The archive puts FPL assists per FPL goal at 0.89-0.93 in every one of seven seasons."""
    assert 0.88 <= ASSIST_RATE <= 0.94


def test_allocation_respects_rate_and_minutes():
    out = allocate_team_goals(_squad(), 2.0).set_index("player")
    assert out.loc["striker", "expected_goals"] > out.loc["winger", "expected_goals"]
    assert out.loc["defender", "expected_goals"] < out.loc["winger", "expected_goals"]


def test_a_player_with_no_minutes_scores_nothing():
    out = allocate_team_goals(_squad(), 2.0).set_index("player")
    assert out.loc["benched", "expected_goals"] == pytest.approx(0.0)
    assert out.loc["benched", "expected_assists"] == pytest.approx(0.0)


def test_penalty_taker_gets_a_premium_over_an_identical_non_taker():
    """Designated takers are frequently the best value in the game; averaging penalties
    across the squad would erase exactly that edge."""
    squad = _squad()
    squad.loc[squad["player"] == "winger", ["xg_per90", "expected_minutes"]] = [0.60, 85.0]
    out = allocate_team_goals(squad, 2.0).set_index("player")
    assert out.loc["striker", "expected_goals"] > out.loc["winger", "expected_goals"]
    assert out.loc["striker", "expected_penalty_goals"] > 0
    assert out.loc["winger", "expected_penalty_goals"] == pytest.approx(0.0)


def test_penalty_goals_scale_with_conversion_rate():
    out = allocate_team_goals(_squad(), 1.43).set_index("player")
    # One taker on 85% duty, league-average attack, converting at PENALTY_CONVERSION.
    assert out.loc["striker", "expected_penalty_goals"] == pytest.approx(
        0.11 * 0.85 * PENALTY_CONVERSION, rel=0.05
    )


def test_assists_are_capped_below_team_goals():
    """Not every goal is assisted — solo goals, rebounds and penalties carry none."""
    out = allocate_team_goals(_squad(), 2.0)
    assert out["expected_assists"].sum() == pytest.approx(2.0 * ASSIST_RATE, abs=0.02)


def test_stronger_fixtures_produce_more_returns():
    weak = allocate_team_goals(_squad(), 0.8)["expected_goals"].sum()
    strong = allocate_team_goals(_squad(), 3.0)["expected_goals"].sum()
    assert strong > weak * 3


# --- distributions and points --------------------------------------------


def test_goal_pmf_is_a_distribution_with_a_tail():
    pmf = goal_involvement_pmf(0.6)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf[0] > pmf[1] > pmf[2]
    assert pmf[2] > 0            # the haul tail is what captaincy is decided on


def test_attacking_points_use_the_positional_scoring_table():
    """Goals are worth different amounts by position — 6 for a defender, 4 for a forward."""
    points = attacking_points(
        pd.Series([1.0, 1.0]), pd.Series([0.0, 0.0]), pd.Series(["DEF", "FWD"]),
        goal_points={"GK": 10, "DEF": 6, "MID": 5, "FWD": 4},
    )
    assert points.iloc[0] == pytest.approx(6.0)
    assert points.iloc[1] == pytest.approx(4.0)


def test_assists_are_worth_three_regardless_of_position():
    points = attacking_points(
        pd.Series([0.0]), pd.Series([2.0]), pd.Series(["DEF"]),
        goal_points={"DEF": 6}, assist_points=3,
    )
    assert points.iloc[0] == pytest.approx(6.0)
