"""The ensemble simulator: perturbed decision paths, and paired comparison across them.

A season replay is one draw from a chaotic process. A 0.1% forecast perturbation moves a real
season total with a standard deviation of 38 points, because one flipped transfer changes the
squad and the squad changes every later decision. These tests pin the three properties that
make the instrument trustworthy: the perturbation reaches decisions but never outcomes, the
same path index gives the same perturbation to every variant, and the comparison is paired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.ensemble import (
    compare_paths,
    ensemble_season,
    per_season_paired,
    perturb,
    resolvable,
)

RULES = {
    "squad": {
        "budget": 100.0, "size": 15, "starting_xi": 11, "max_per_club": 3,
        "positions": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "formation": {
            "GK": {"min": 1, "max": 1}, "DEF": {"min": 3, "max": 5},
            "MID": {"min": 2, "max": 5}, "FWD": {"min": 1, "max": 3},
        },
    },
}


def _season(n_gws=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for gw in range(1, n_gws + 1):
        pid = 0
        for position, count in (("GK", 8), ("DEF", 16), ("MID", 16), ("FWD", 10)):
            for i in range(count):
                rows.append({
                    "gw": gw, "player_id": pid, "web_name": f"{position}{i}",
                    "position": position, "team": f"club{i % 8}",
                    "price": 4.0 + (i % 6) * 1.5 + rng.normal(0, 0.05),
                    "expected_points": 1.0 + (i % 6) * 0.7 + rng.normal(0, 0.05),
                    "actual_points": float(rng.poisson(2 + (i % 6))),
                })
                pid += 1
    return pd.DataFrame(rows)


# --- the perturbation ------------------------------------------------------------------


def test_outcomes_are_never_perturbed():
    """The one thing that would invalidate everything. Jitter explores decision paths; if it
    reached `actual_points` it would be resampling the season instead."""
    frame = _season()
    shifted, _ = perturb(frame, None, jitter=0.05, seed=3)
    pd.testing.assert_series_equal(shifted["actual_points"], frame["actual_points"])
    assert not shifted["expected_points"].equals(frame["expected_points"])


def test_the_same_seed_gives_the_same_perturbation():
    """What makes pairing possible: two variants at path 3 must meet identical forecasts."""
    frame = _season()
    first, _ = perturb(frame, None, seed=3)
    second, _ = perturb(frame, None, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_paths():
    frame = _season()
    first, _ = perturb(frame, None, seed=0)
    second, _ = perturb(frame, None, seed=1)
    assert not first["expected_points"].equals(second["expected_points"])


def test_a_player_gameweek_is_perturbed_identically_everywhere_it_appears():
    """A decision week and the horizon views that also contain that gameweek have to agree, or
    the perturbation is not an alternative forecast — just noise layered on itself."""
    frame = _season()
    lookahead = {1: {1: frame[frame["gw"] == 1], 2: frame[frame["gw"] == 2]}}
    shifted, shifted_lookahead = perturb(frame, lookahead, jitter=0.02, seed=5)

    for gw in (1, 2):
        direct = shifted[shifted["gw"] == gw].set_index("player_id")["expected_points"]
        via = shifted_lookahead[1][gw].set_index("player_id")["expected_points"]
        pd.testing.assert_series_equal(direct, via, check_names=False)


def test_zero_jitter_changes_nothing():
    frame = _season()
    shifted, _ = perturb(frame, None, jitter=0.0, seed=1)
    pd.testing.assert_frame_equal(shifted, frame)


# --- the ensemble ----------------------------------------------------------------------


def test_one_row_per_path():
    runs = ensemble_season(_season(), label="v", paths=3, rules=RULES)
    assert len(runs) == 3
    assert sorted(runs["path"]) == [0, 1, 2]
    assert (runs["points"] > 0).all()


def test_paths_actually_differ_when_jitter_is_large_enough():
    """If every path returned the same total the instrument would be measuring nothing."""
    runs = ensemble_season(_season(n_gws=4), label="v", paths=6, jitter=0.05, rules=RULES)
    assert runs["points"].nunique() > 1


# --- the comparison --------------------------------------------------------------------


def test_the_baseline_differences_against_itself_at_zero():
    runs = pd.concat([
        ensemble_season(_season(), label="a", paths=3, rules=RULES),
        ensemble_season(_season(), label="b", paths=3, rules=RULES, horizon=0),
    ], ignore_index=True)
    table = compare_paths(runs, "a").set_index("variant")
    assert table.loc["a", "mean_diff"] == 0.0
    assert table.loc["a", "se"] == 0.0


def test_the_comparison_is_paired_not_pooled():
    """Pairing is the whole reason this resolves anything. Two variants that differ by a
    constant on every path must show that difference with ZERO standard error, however wide
    the spread of the totals themselves."""
    runs = pd.DataFrame([
        {"label": "base", "path": p, "points": 2000 + 100 * p} for p in range(5)
    ] + [
        {"label": "better", "path": p, "points": 2000 + 100 * p + 7} for p in range(5)
    ])
    table = compare_paths(runs, "base").set_index("variant")

    assert table.loc["better", "mean_diff"] == pytest.approx(7.0)
    assert table.loc["better", "se"] == pytest.approx(0.0)
    assert table.loc["better", "wins"] == 1.0
    assert table.loc["better", "sd"] > 100      # the totals themselves are all over the place


def test_an_unknown_baseline_is_an_error_not_an_empty_table():
    runs = ensemble_season(_season(), label="v", paths=2, rules=RULES)
    with pytest.raises(KeyError, match="nope"):
        compare_paths(runs, "nope")


# --- the adoption rule -----------------------------------------------------------------


def _runs(diffs_by_season):
    """A variant beating the baseline by `diff` on every path of each season."""
    rows = []
    for season, diff in diffs_by_season.items():
        for path in range(4):
            rows.append({"season": season, "path": path, "label": "base", "points": 2000.0})
            rows.append({
                "season": season, "path": path, "label": "v", "points": 2000.0 + diff
            })
    return pd.DataFrame(rows)


def test_a_consistent_effect_is_adoptable():
    table = resolvable(_runs({"a": 10, "b": 12, "c": 8}), "base").set_index("variant")
    assert bool(table.loc["v", "excludes_zero"])
    assert bool(table.loc["v", "seasons_agree"])
    assert bool(table.loc["v", "adoptable"])


def test_pooling_hides_a_large_effect_whose_sign_depends_on_the_season():
    """The reason `per_season_paired` exists. A policy worth +150 in one season and -150 in
    another pools to roughly zero with a wide interval, which reads as 'no effect' — when in
    fact the effect is enormous and its direction is the open question."""
    runs = _runs({"a": 150, "b": -150, "c": 0})

    pooled = compare_paths(runs, "base").set_index("variant")
    assert pooled.loc["v", "mean_diff"] == pytest.approx(0.0)

    split = per_season_paired(runs, "base")
    split = split[split["variant"] == "v"].set_index("season")["mean_diff"]
    assert split["a"] == pytest.approx(150.0)
    assert split["b"] == pytest.approx(-150.0)


def test_per_season_pairing_needs_a_season_column():
    runs = ensemble_season(_season(), label="v", paths=2, rules=RULES)
    with pytest.raises(KeyError, match="season"):
        per_season_paired(runs, "v")


def test_a_flipping_sign_is_not_adoptable_however_tight_the_interval():
    """Ground rule 2, enforced. Path averaging says how sure we are about THESE seasons; it
    says nothing about the next one, and a confident-looking mean built from +100 / +100 / -10
    is exactly the shape that has misled this project before."""
    table = resolvable(_runs({"a": 100, "b": 100, "c": -10}), "base").set_index("variant")
    assert bool(table.loc["v", "excludes_zero"])
    assert not bool(table.loc["v", "seasons_agree"])
    assert not bool(table.loc["v", "adoptable"])
