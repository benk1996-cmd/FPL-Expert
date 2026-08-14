"""Repeated simulation: pairing, reproducibility, and not fooling ourselves about noise."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.repeat_sim import (
    draws_needed,
    is_resolved,
    paired_comparison,
    repeat_season,
    resample_actuals,
)
from fpl_expert.models.distribution import PointsDistribution, gameweek_distributions
from tests.test_distribution import RULES as SCORING

RULES = {
    **SCORING,
    "squad": {
        "budget": 100.0, "size": 15, "starting_xi": 11, "max_per_club": 3,
        "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "formation": {
            "GK": {"min": 1, "max": 1}, "DEF": {"min": 3, "max": 5},
            "MID": {"min": 2, "max": 5}, "FWD": {"min": 1, "max": 3},
        },
    },
}


def _fixture_level(n_gws=3, seed=0):
    """A small season at fixture grain — what the distribution is built from.

    Club and price are keyed on different moduli deliberately. Tying both to `i % 6` makes
    every cheap player share a club, and the 3-per-club limit then forces an over-budget
    squad — an infeasible fixture rather than an interesting one.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gws + 1):
        pid = 0
        for position, count in (("GK", 8), ("DEF", 16), ("MID", 16), ("FWD", 10)):
            for i in range(count):
                p_long = float(rng.uniform(0.2, 0.95))
                rows.append({
                    "season": "2025-26", "gw": gw, "player_id": pid, "name": f"{position}{i}",
                    "web_name": f"{position}{i}", "position": position, "team": f"c{i % 8}",
                    "fixture": gw * 100 + i, "price": 4.0 + (i % 6) * 1.5 + rng.normal(0, 0.05),
                    "p_long": p_long, "p_short": 0.1, "p_zero": 0.9 - p_long,
                    "expected_minutes": 0.1 * 22.0 + p_long * 85.0,
                    "expected_goals": float(rng.uniform(0, 0.5)),
                    "expected_assists": float(rng.uniform(0, 0.3)),
                    "p_clean_sheet": float(rng.uniform(0.15, 0.45)),
                    "expected_goals_against": float(rng.uniform(0.8, 2.0)),
                    "expected_saves": float(rng.uniform(1, 4)) if position == "GK" else 0.0,
                    "defcon_per90": float(rng.uniform(3, 11)),
                    "yellows_per90": float(rng.uniform(0, 0.3)),
                    "pts_bonus": float(rng.uniform(0, 0.6)),
                    "pts_defcon": 0.2, "pts_saves": 0.1,
                    "expected_points": 1.0 + (i % 6) * 0.7 + rng.normal(0, 0.05),
                    "actual_points": float(rng.poisson(2 + (i % 6))),
                    "selected": float(1000 * (1 + (i % 6))),
                })
                pid += 1
    return pd.DataFrame(rows)


# --- resampling ------------------------------------------------------------


def test_resampling_replaces_outcomes_but_not_forecasts():
    """The strategy must decide in ignorance of the draw. If a resample touched the forecast
    columns the exercise would be scoring decisions made with knowledge of the answer."""
    frame = _fixture_level()
    keys, distribution = gameweek_distributions(frame, RULES)
    sampled = resample_actuals(frame, distribution, keys, np.random.default_rng(0))

    assert (sampled["expected_points"].to_numpy() == frame["expected_points"].to_numpy()).all()
    assert not (sampled["actual_points"].to_numpy() == frame["actual_points"].to_numpy()).all()


def test_resampled_outcomes_are_integers_in_the_supported_range():
    frame = _fixture_level()
    keys, distribution = gameweek_distributions(frame, RULES)
    sampled = resample_actuals(frame, distribution, keys, np.random.default_rng(1))
    drawn = sampled["actual_points"].to_numpy()

    assert np.allclose(drawn, np.round(drawn))
    assert drawn.min() >= distribution.support[0]
    assert drawn.max() <= distribution.support[-1]


def test_resampling_is_reproducible_from_a_seed():
    frame = _fixture_level()
    keys, distribution = gameweek_distributions(frame, RULES)
    first = resample_actuals(frame, distribution, keys, np.random.default_rng([0, 3]))
    second = resample_actuals(frame, distribution, keys, np.random.default_rng([0, 3]))
    assert (first["actual_points"].to_numpy() == second["actual_points"].to_numpy()).all()


def test_resampled_seasons_average_out_to_the_forecast():
    """Over many draws the mean outcome must return the distribution's own mean — otherwise
    the resampler is introducing a bias that would be read as a strategy effect."""
    frame = _fixture_level(n_gws=1)
    keys, distribution = gameweek_distributions(frame, RULES)
    totals = [
        resample_actuals(frame, distribution, keys, np.random.default_rng([9, d]))[
            "actual_points"
        ].sum()
        for d in range(60)
    ]
    assert np.mean(totals) == pytest.approx(distribution.mean().sum(), rel=0.05)


# --- the solve-once optimisation rests on these two facts ------------------


def test_rescoring_the_real_outcomes_reproduces_the_simulator_exactly():
    """`rescore` replaces a full re-solve, so it has to agree to the point, not approximately.
    Any drift here would silently make every resampled comparison measure something other
    than the simulator it claims to be measuring."""
    from fpl_expert.backtest.season_sim import rescore, simulate_season

    frame = _fixture_level(n_gws=6, seed=2)
    for settings in ({"use_chips": False}, {"use_chips": True}):
        result = simulate_season(frame, rules=RULES, **settings)
        outcomes = dict(
            zip(zip(frame["gw"], frame["player_id"]), frame["actual_points"])
        )
        assert rescore(result.trace, outcomes) == pytest.approx(result.total_points)


def test_decisions_do_not_depend_on_realised_outcomes():
    """The other half of the optimisation, and a correctness property in its own right: a
    simulator whose decisions moved with the outcomes would be reading the answer."""
    from fpl_expert.backtest.season_sim import simulate_season

    frame = _fixture_level(n_gws=6, seed=3)
    scrambled = frame.copy()
    rng = np.random.default_rng(0)
    scrambled["actual_points"] = rng.permutation(scrambled["actual_points"].to_numpy())

    original = simulate_season(frame, rules=RULES, use_chips=True)
    shuffled = simulate_season(scrambled, rules=RULES, use_chips=True)

    assert [w["scoring_ids"] for w in original.trace] == [
        w["scoring_ids"] for w in shuffled.trace
    ]
    assert [w["captain_id"] for w in original.trace] == [
        w["captain_id"] for w in shuffled.trace
    ]
    assert [w["chip"] for w in original.trace] == [w["chip"] for w in shuffled.trace]
    # ...and the totals must differ, or the outcomes were not really scrambled.
    assert original.total_points != shuffled.total_points


def test_bench_boost_scores_the_whole_squad_in_the_trace():
    """The trace has to carry which players scored, not just how many — a Bench Boost week
    scores fifteen and every other week eleven."""
    from fpl_expert.backtest.season_sim import score_gameweek

    frame = _fixture_level(n_gws=1)
    squad = set(frame.nlargest(15, "expected_points")["player_id"])
    normal = score_gameweek(squad, None, frame, RULES)
    boosted = score_gameweek(squad, None, frame, RULES, chip="bench_boost")

    assert len(normal["scoring_ids"]) == 11
    assert len(boosted["scoring_ids"]) == 15


# --- pairing ---------------------------------------------------------------


def test_variants_face_identical_outcomes_within_a_draw():
    """Pairing is what makes the comparison affordable. Season totals swing far more between
    draws than between strategies, and that swing only cancels if every variant meets the
    same drawn season."""
    frame = _fixture_level(n_gws=3)
    result = repeat_season(
        frame.copy(), frame, n_draws=3, seed=0, field_size=500, rules=RULES,
        variants={"a": {"use_chips": False}, "b": {"use_chips": False}},
    )
    wide = result.draws.pivot(index="draw", columns="variant", values="points")
    # Identical settings under identical outcomes must give identical scores.
    assert np.allclose(wide["a"], wide["b"])
    # ...and different draws must not, or nothing is being resampled at all.
    assert wide["a"].nunique() > 1


def test_paired_comparison_reports_a_zero_difference_against_itself():
    draws = pd.DataFrame({
        "draw": [0, 0, 1, 1], "variant": ["a", "b", "a", "b"],
        "points": [2000, 2100, 1900, 2050], "rank": [1, 1, 1, 1],
    })
    comparison = paired_comparison(draws, "a")
    own = comparison[comparison["variant"] == "a"].iloc[0]
    assert own["mean_diff"] == 0.0
    other = comparison[comparison["variant"] == "b"].iloc[0]
    assert other["mean_diff"] == pytest.approx(125.0)
    assert other["win_rate"] == 1.0


def test_a_consistent_winner_is_resolved_and_a_coin_flip_is_not():
    """The distinction the module exists to draw. A variant that wins by 3 points with a
    spread of 200 has not been shown to be better, however many seasons it won."""
    rng = np.random.default_rng(0)
    n = 40
    shared = rng.normal(2000, 200, n)
    draws = pd.concat([
        pd.DataFrame({"draw": range(n), "variant": "base", "points": shared}),
        pd.DataFrame({"draw": range(n), "variant": "clear", "points": shared + 60}),
        pd.DataFrame({"draw": range(n), "variant": "noise",
                      "points": shared + rng.normal(3, 200, n)}),
    ])
    comparison = paired_comparison(draws, "base")
    assert is_resolved(comparison, "clear")
    assert not is_resolved(comparison, "noise")


def test_draws_needed_grows_as_the_gap_shrinks():
    rng = np.random.default_rng(1)
    n = 30
    shared = rng.normal(2000, 150, n)
    draws = pd.concat([
        pd.DataFrame({"draw": range(n), "variant": "base", "points": shared}),
        pd.DataFrame({"draw": range(n), "variant": "other",
                      "points": shared + rng.normal(5, 40, n)}),
    ])
    comparison = paired_comparison(draws, "base")
    assert draws_needed(comparison, "other", 20) < draws_needed(comparison, "other", 5)


# --- end to end ------------------------------------------------------------


def test_repeat_season_returns_a_row_per_draw_and_variant():
    frame = _fixture_level(n_gws=3)
    result = repeat_season(
        frame.copy(), frame, n_draws=2, seed=0, field_size=400, rules=RULES,
        variants={"chips": {"use_chips": True}, "none": {"use_chips": False}},
    )
    assert len(result.draws) == 4
    assert set(result.draws["variant"]) == {"chips", "none"}
    assert (result.draws["rank"] > 0).all()

    summary = result.summary()
    assert {"points", "points_sd", "points_se", "rank"} <= set(summary.columns)


def test_summary_standard_error_shrinks_with_more_draws():
    """The reported error is the error of the MEAN, so it must fall as draws accumulate —
    that is what tells you whether more compute would move the answer."""
    from fpl_expert.backtest.repeat_sim import RepeatResult

    rng = np.random.default_rng(0)
    def build(n):
        return RepeatResult(
            pd.DataFrame({
                "draw": range(n), "variant": "a",
                "points": rng.normal(2000, 100, n), "rank": 5000,
            }),
            n,
        )

    assert build(200).summary()["points_se"].iloc[0] < build(10).summary()["points_se"].iloc[0]


def test_rank_is_suppressed_when_the_field_is_saturated():
    """Reporting a rank from only the draws that did not saturate would be a median over the
    unluckiest seasons — worse than reporting nothing, because it looks like a measurement."""
    from fpl_expert.backtest.repeat_sim import RepeatResult

    n = 20
    mostly_saturated = RepeatResult(
        pd.DataFrame({
            "draw": range(n), "variant": "a", "points": np.linspace(2000, 2100, n),
            "rank": [None] * (n - 2) + [5000, 6000],
            "field_saturated": [True] * (n - 2) + [False, False],
        }),
        n,
    )
    assert "rank" not in mostly_saturated.summary().columns

    contained = RepeatResult(
        pd.DataFrame({
            "draw": range(n), "variant": "a", "points": np.linspace(2000, 2100, n),
            "rank": np.linspace(4000, 9000, n), "field_saturated": [False] * n,
        }),
        n,
    )
    assert "rank" in contained.summary().columns


def test_paired_comparison_rejects_an_unknown_baseline():
    draws = pd.DataFrame({"draw": [0], "variant": ["a"], "points": [1.0]})
    with pytest.raises(KeyError):
        paired_comparison(draws, "missing")


def test_distribution_sampling_shape_matches_the_key_frame():
    frame = _fixture_level(n_gws=2)
    keys, distribution = gameweek_distributions(frame, RULES)
    assert len(keys) == len(distribution)
    assert isinstance(distribution, PointsDistribution)
    assert set(keys.columns) == {"season", "gw", "player_id"}
