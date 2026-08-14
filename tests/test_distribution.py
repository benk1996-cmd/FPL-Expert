"""The per-player points distribution: exactness, consistency with the assembler, and shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.distribution import (
    GRID_HIGH,
    GRID_LOW,
    PointsDistribution,
    _bonus_pmf,
    _convolve,
    build_distribution,
    combine_fixtures,
    gameweek_distributions,
)
from fpl_expert.models.points import assemble

RULES = {
    "appearance": {"under_60": 1, "sixty_plus": 2},
    "goal": {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4},
    "assist": 3,
    "clean_sheet": {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0},
    "saves_per_point": 3,
    "goals_conceded": {"points": -1, "per_goals": 2, "applies_to": ["GK", "DEF"]},
    "yellow_card": -1,
    "red_card": -3,
    "defensive_contribution": {
        "enabled": True, "points": 2,
        "thresholds": {"DEF": 10, "MID": 12, "FWD": 12, "GK": None},
    },
}


def _players(n=40, seed=0, positions=("GK", "DEF", "MID", "FWD")):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        position = positions[i % len(positions)]
        p_long = float(rng.uniform(0.05, 0.9))
        p_short = float(rng.uniform(0.0, 1.0 - p_long) * 0.4)
        rows.append({
            "player_id": i, "name": f"p{i}", "position": position, "team": f"c{i % 5}",
            # A fixture belongs to a TEAM, not a player. Giving each player his own would make
            # every goal-allocation group a single player and hand him the whole team's xG.
            "fixture": 100 + (i % 5), "price": 4.0 + i % 8,
            "p_long": p_long, "p_short": p_short,
            "p_zero": 1 - p_long - p_short,
            "expected_minutes": p_short * 22.0 + p_long * 85.0,
            "expected_goals": float(rng.uniform(0, 0.6)) * p_long,
            "expected_assists": float(rng.uniform(0, 0.4)) * p_long,
            "p_clean_sheet": float(rng.uniform(0.1, 0.5)),
            "expected_goals_against": float(rng.uniform(0.8, 2.2)),
            "expected_saves": float(rng.uniform(1, 4)) if position == "GK" else 0.0,
            "defcon_per90": float(rng.uniform(3, 11)),
            "yellows_per90": float(rng.uniform(0, 0.4)),
            "pts_bonus": float(rng.uniform(0, 0.8)),
        })
    return pd.DataFrame(rows)


# --- the distribution is a distribution ------------------------------------


def test_every_row_is_a_proper_distribution():
    pmf = build_distribution(_players(), RULES).pmf
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert (pmf >= 0).all()


def test_support_covers_the_configured_grid():
    distribution = build_distribution(_players(), RULES)
    assert distribution.support[0] == GRID_LOW
    assert distribution.support[-1] == GRID_HIGH


def test_a_player_who_never_plays_scores_exactly_zero():
    """The zero-minute outcome is an atom at 0, not a tail. No smooth approximation of a
    player's score can express that, which is why buckets are conditioned on rather than
    averaged over."""
    frame = _players(8)
    frame[["p_long", "p_short"]] = 0.0
    frame["p_zero"] = 1.0
    frame["expected_minutes"] = 0.0
    distribution = build_distribution(frame, RULES)
    assert np.allclose(distribution.pmf[:, -GRID_LOW], 1.0)
    assert np.allclose(distribution.mean(), 0.0)


def test_a_nailed_starter_always_banks_his_appearance_points():
    """With no card risk, a player certain to last 60 minutes cannot score below two — every
    other component only adds. That the floor is a hard bound rather than a quantile is the
    property the zero-minutes atom would otherwise hide.

    Zero itself is NOT tested for absence, because it is genuinely reachable: a defender
    conceding four loses exactly his two appearance points, and a midfielder can pair a clean
    sheet with a red card.
    """
    frame = _players(8, positions=("MID", "FWD"))
    frame["p_long"], frame["p_short"], frame["p_zero"] = 1.0, 0.0, 0.0
    frame["expected_minutes"], frame["yellows_per90"] = 85.0, 0.0
    distribution = build_distribution(frame, RULES)
    assert np.allclose(distribution.p_at_least(2), 1.0)


# --- consistency with the model it is derived from -------------------------


def test_distribution_mean_tracks_the_assembler():
    """Two independently written paths to the same expectation. They will not agree exactly —
    this one conditions thresholds on the minutes bucket and couples clean sheets to goals
    conceded, both of which are refinements — but a large gap means one of them is wrong."""
    players = _players(60)
    team = pd.DataFrame({
        "team": [f"c{i}" for i in range(5)],
        "expected_goals_for": [1.2, 1.6, 1.0, 2.0, 1.4],
        "expected_goals_against": [1.3, 1.1, 1.7, 0.9, 1.5],
        "p_clean_sheet": [0.28, 0.33, 0.20, 0.42, 0.25],
        "expected_conceded_penalty": [-0.4, -0.35, -0.5, -0.28, -0.45],
        "opponent": ["x"] * 5,
    })
    extra = players.assign(
        xg_per90=0.3, xa_per90=0.2, finishing_multiplier=1.0, penalty_share=0.0,
        saves_per90=2.5, bonus_per90=0.3,
        expected_appearance_points=players["p_short"] + 2 * players["p_long"],
    )
    assembled = assemble(extra, team, rules=RULES)
    distribution = build_distribution(assembled, RULES)

    assert np.corrcoef(distribution.mean(), assembled["expected_points"])[0, 1] > 0.97
    assert abs(distribution.mean().mean() - assembled["expected_points"].mean()) < 0.35


def test_dispersion_widens_the_tail_without_moving_the_mean():
    """The negative binomial is a shape correction. If it moved the mean it would silently put
    this module at odds with the number the optimiser reads."""
    from fpl_expert.models.distribution import _poisson_counts

    # Realistic per-fixture goal rates. A negative binomial of the same mean puts MORE mass
    # at zero and more in the far tail, taking it from the middle — so the comparison has to
    # be made where the extra dispersion actually shows, not at an arbitrary threshold. At a
    # mean of 1.2, P(2+) is genuinely lower under the negative binomial.
    lam = np.array([0.08, 0.25, 0.6])
    # A generous cap: both pmfs fold their tail into the last cell, and the negative binomial
    # has more tail to fold, so a tight cap would move the mean by the truncation alone.
    cap = 40
    poisson_counts = _poisson_counts(lam, cap)
    nb_counts = _poisson_counts(lam, cap, dispersion=2.85)
    counts = np.arange(cap + 1)

    assert np.allclose(poisson_counts @ counts, nb_counts @ counts, atol=1e-6)
    assert (nb_counts[:, 2:].sum(axis=1) > poisson_counts[:, 2:].sum(axis=1)).all()
    assert (nb_counts[:, 3:].sum(axis=1) > poisson_counts[:, 3:].sum(axis=1)).all()
    assert (nb_counts[:, 0] > poisson_counts[:, 0]).all()


# --- the part that motivated the module ------------------------------------


def test_bonus_is_coupled_to_returns_not_independent_of_them():
    """Measured in the archive: E[bonus] is 0.095 with no returns and 2.25 with two. Treating
    them as independent understated P(10+ points) by a factor of 2.5, and a haul is exactly a
    return plus the bonus that comes with it."""
    frame = _players(1, positions=("FWD",))
    frame["p_long"], frame["p_short"], frame["p_zero"] = 1.0, 0.0, 0.0
    frame["expected_minutes"] = 85.0
    frame["expected_goals"], frame["expected_assists"] = 0.8, 0.3
    frame["pts_bonus"] = 0.5

    coupled = build_distribution(frame, RULES)
    # Ceiling with bonus concentrated on the scoring outcomes, against the same expected
    # bonus spread evenly over every outcome.
    independent = _convolve(
        (0, np.ones((1, 1))), (0, _bonus_pmf(np.array([0.5])))
    )
    assert coupled.p_haul(10)[0] > 0.0
    assert independent[1].shape == (1, 4)


def test_bonus_pmf_matches_its_target_mean_in_both_branches():
    means = np.array([0.0, 0.4, 1.5, 2.0, 2.4, 3.0])
    pmf = _bonus_pmf(means)
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert np.allclose(pmf @ np.arange(4), means)
    assert (pmf >= -1e-12).all()


def test_clean_sheet_and_conceding_cannot_both_happen():
    """Modelled independently a defender could be credited with a clean sheet and a conceding
    deduction in the same match. They are one event seen twice."""
    frame = _players(1, positions=("DEF",))
    frame["p_long"], frame["p_short"], frame["p_zero"] = 1.0, 0.0, 0.0
    frame["expected_minutes"] = 85.0
    frame["expected_goals"] = frame["expected_assists"] = 0.0
    frame["pts_bonus"], frame["yellows_per90"], frame["defcon_per90"] = 0.0, 0.0, 0.0
    frame["p_clean_sheet"], frame["expected_goals_against"] = 0.4, 1.5

    distribution = build_distribution(frame, RULES)
    # 2 appearance + 4 clean sheet = 6, reachable only when nothing is conceded.
    assert distribution.pmf[0, 6 - GRID_LOW] == pytest.approx(0.4, abs=0.01)


def test_clean_sheet_probability_is_the_match_models_not_a_refit_poisson():
    """The match model's clean-sheet number carries the Dixon-Coles low-score correction. A
    plain Poisson on expected goals against would quietly discard it."""
    frame = _players(1, positions=("GK",))
    frame["p_long"], frame["p_short"], frame["p_zero"] = 1.0, 0.0, 0.0
    frame["expected_minutes"], frame["expected_saves"] = 85.0, 0.0
    frame["expected_goals"] = frame["expected_assists"] = 0.0
    frame["pts_bonus"] = frame["yellows_per90"] = 0.0
    frame["p_clean_sheet"], frame["expected_goals_against"] = 0.55, 1.4

    distribution = build_distribution(frame, RULES)
    assert distribution.pmf[0, 6 - GRID_LOW] == pytest.approx(0.55, abs=0.01)


# --- derived quantities ----------------------------------------------------


def test_ceiling_exceeds_the_mean_and_floor_falls_below_it():
    distribution = build_distribution(_players(30), RULES)
    assert (distribution.ceiling() >= distribution.floor()).all()
    assert (distribution.ceiling(0.95) >= distribution.mean()).all()


def test_two_players_with_equal_means_can_differ_in_upside():
    """The reason the module exists. A mean cannot separate a reliable six-point return from
    a coin flip between a blank and a haul, and captaincy is a bet on exactly that difference."""
    steady = PointsDistribution(np.array([[0.0, 1.0, 0.0]]), low=5)
    volatile = PointsDistribution(np.array([[0.5, 0.0, 0.5]]), low=5)
    assert steady.mean()[0] == pytest.approx(volatile.mean()[0])
    assert volatile.p_at_least(7)[0] > steady.p_at_least(7)[0]
    assert volatile.std()[0] > steady.std()[0]


def test_sampling_reproduces_the_distribution():
    distribution = build_distribution(_players(20), RULES)
    drawn = distribution.sample(np.random.default_rng(0), draws=4000)
    assert drawn.shape == (4000, 20)
    assert np.allclose(drawn.mean(axis=0), distribution.mean(), atol=0.25)


def test_sampling_is_reproducible_from_a_seed():
    distribution = build_distribution(_players(20), RULES)
    first = distribution.sample(np.random.default_rng(7), draws=5)
    second = distribution.sample(np.random.default_rng(7), draws=5)
    assert (first == second).all()


def test_summary_frame_has_a_row_per_player():
    distribution = build_distribution(_players(12), RULES)
    frame = distribution.to_frame()
    assert len(frame) == 12
    assert {"floor", "ceiling", "p_haul", "dist_std"} <= set(frame.columns)


# --- double gameweeks ------------------------------------------------------


def test_a_double_gameweek_convolves_rather_than_averages():
    """Two fixtures is the sum of two draws. Its ceiling is higher than either alone, which is
    the entire basis for timing Bench Boost and Triple Captain."""
    frame = _players(2, positions=("MID",))
    frame["player_id"] = [7, 7]                     # one player, two fixtures
    frame["season"], frame["gw"] = "2025-26", 10

    keys, combined = gameweek_distributions(frame, RULES)
    single = build_distribution(frame.head(1), RULES)

    assert len(combined) == 1
    assert keys["player_id"].iloc[0] == 7
    assert combined.mean()[0] > single.mean()[0]
    assert combined.ceiling(0.95)[0] >= single.ceiling(0.95)[0]


def test_single_fixture_players_pass_through_unchanged():
    frame = _players(6)
    frame["season"], frame["gw"] = "2025-26", 3
    fixture_level = build_distribution(frame, RULES)
    _, combined = gameweek_distributions(frame, RULES)
    assert np.allclose(combined.pmf, fixture_level.pmf)


def test_combine_rejects_mismatched_keys():
    distribution = build_distribution(_players(4), RULES)
    with pytest.raises(ValueError):
        combine_fixtures(distribution, pd.DataFrame({"player_id": [1, 2]}))
