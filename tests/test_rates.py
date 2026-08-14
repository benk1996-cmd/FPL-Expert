"""The shrinkage estimator — the piece most able to fool us if it is subtly wrong."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.features.rates import (
    GammaPrior,
    decay_weights,
    fit_gamma_prior,
    price_tier,
    rate_features,
    season_gw_index,
    shrink,
)


def _history(rows):
    return pd.DataFrame(rows)


# --- the time axis --------------------------------------------------------


def test_season_gw_index_is_monotonic_across_seasons():
    season = pd.Series(["2024-25", "2024-25", "2025-26"])
    idx = season_gw_index(season, pd.Series([37, 38, 1]))
    assert idx.is_monotonic_increasing
    assert idx.iloc[2] - idx.iloc[1] == 1   # GW38 -> next season's GW1 is one step


def test_decay_weights_are_zero_at_and_after_as_of():
    """The hard zero is what stops a gameweek's feature being built from its own result."""
    idx = pd.Series([8, 9, 10, 11])
    w = decay_weights(idx, as_of=10, half_life_gws=4)
    assert w.iloc[2] == 0.0     # the gameweek being predicted
    assert w.iloc[3] == 0.0     # the future
    assert w.iloc[1] > w.iloc[0] > 0


def test_decay_half_life_halves_the_weight():
    w = decay_weights(pd.Series([0, 10]), as_of=20, half_life_gws=10)
    assert w.iloc[1] / w.iloc[0] == pytest.approx(2.0)   # 10 GWs newer = twice the weight


# --- the prior ------------------------------------------------------------


def test_prior_subtracts_poisson_noise_from_observed_spread():
    """Players with identical true rates still differ by chance; the prior must not read
    that sampling noise as real spread, or nothing gets shrunk."""
    rng = np.random.default_rng(0)
    exposure = np.full(400, 5.0)                  # 5 full matches each
    counts = rng.poisson(0.3 * exposure)          # ALL have the same true rate
    prior = fit_gamma_prior(counts, exposure)

    assert prior.mean == pytest.approx(0.3, abs=0.05)
    # Truth is zero between-player variance, so the prior should be very tight, i.e. a
    # large pseudo-exposure. A naive fit on raw rate variance gives beta of order 10.
    assert prior.pseudo_90s > 100


def test_prior_detects_genuine_heterogeneity():
    """When players really do differ, the prior must be wide enough to let them separate."""
    exposure = np.full(400, 20.0)
    rates = np.repeat([0.1, 0.9], 200)
    counts = rates * exposure                     # noiseless, so all spread is real
    prior = fit_gamma_prior(counts, exposure)

    assert prior.mean == pytest.approx(0.5, abs=0.05)
    assert prior.pseudo_90s < 5                   # weak prior — let the data speak


def test_prior_on_empty_population_is_degenerate_not_a_crash():
    prior = fit_gamma_prior(np.array([]), np.array([]))
    assert prior.beta > 0 and prior.mean >= 0


# --- shrinkage behaviour --------------------------------------------------


def test_hot_streak_is_shrunk_hard():
    """Two goals in 40 minutes is an observed 4.5/90. It must not survive as a feature."""
    prior = GammaPrior(alpha=0.3 * 8, beta=8.0)          # league mean 0.3, 8 pseudo-90s
    out = shrink(pd.Series([2.0]), pd.Series([40 / 90]), prior)

    assert out["rate"].iloc[0] < 0.6                      # nowhere near 4.5
    assert out["confidence"].iloc[0] < 0.1                # and flagged as barely evidenced


def test_zero_exposure_falls_back_to_the_prior_mean():
    """A promoted-club player or new signing: no history, so the price/position prior is
    the whole estimate, and confidence is exactly zero."""
    prior = GammaPrior(alpha=0.3 * 8, beta=8.0)
    out = shrink(pd.Series([0.0]), pd.Series([0.0]), prior)

    assert out["rate"].iloc[0] == pytest.approx(prior.mean)
    assert out["confidence"].iloc[0] == 0.0


def test_heavy_exposure_converges_on_the_observed_rate():
    prior = GammaPrior(alpha=0.3 * 8, beta=8.0)
    out = shrink(pd.Series([90.0]), pd.Series([100.0]), prior)   # 0.9/90 over 100 matches

    assert out["rate"].iloc[0] == pytest.approx(0.9, abs=0.05)
    assert out["confidence"].iloc[0] > 0.9


def test_confidence_is_the_share_of_evidence_from_the_player():
    prior = GammaPrior(alpha=1.0, beta=10.0)
    out = shrink(pd.Series([5.0]), pd.Series([10.0]), prior)
    assert out["confidence"].iloc[0] == pytest.approx(0.5)   # exposure == pseudo_90s


# --- end to end -----------------------------------------------------------


def test_rate_features_handles_established_and_cold_start_players_together():
    rows = []
    for gw in range(1, 11):                       # an established, prolific player
        rows.append({"element": 1, "season": "2025-26", "GW": gw, "minutes": 90,
                     "goals_scored": 1, "position": "FWD", "tier": 3})
    for gw in range(1, 11):                       # an established, blank player
        rows.append({"element": 2, "season": "2025-26", "GW": gw, "minutes": 90,
                     "goals_scored": 0, "position": "FWD", "tier": 3})
    rows.append({"element": 3, "season": "2025-26", "GW": 1, "minutes": 0,
                 "goals_scored": 0, "position": "FWD", "tier": 3})   # never played

    out = rate_features(_history(rows), as_of=season_gw_index(
        pd.Series(["2025-26"]), pd.Series([11])).iloc[0],
        stats=["goals_scored"], half_life_gws=20)

    by_id = out.set_index("element")
    assert by_id.loc[1, "goals_scored_per90"] > by_id.loc[2, "goals_scored_per90"]
    assert by_id.loc[3, "goals_scored_confidence"] == 0.0
    assert by_id.loc[1, "goals_scored_confidence"] > 0.4
    # The never-played player sits between the two, at the group prior.
    assert (by_id.loc[2, "goals_scored_per90"]
            < by_id.loc[3, "goals_scored_per90"]
            < by_id.loc[1, "goals_scored_per90"])


def test_rate_features_excludes_the_target_gameweek():
    """Point-in-time: a huge haul in the gameweek being predicted must not leak in."""
    rows = [{"element": 1, "season": "2025-26", "GW": gw, "minutes": 90,
             "goals_scored": 0, "position": "FWD", "tier": 1} for gw in range(1, 5)]
    rows.append({"element": 1, "season": "2025-26", "GW": 5, "minutes": 90,
                 "goals_scored": 5, "position": "FWD", "tier": 1})

    as_of = season_gw_index(pd.Series(["2025-26"]), pd.Series([5])).iloc[0]
    out = rate_features(_history(rows), as_of=as_of, stats=["goals_scored"])
    assert out["goals_scored_per90"].iloc[0] < 0.2      # the 5-goal haul is invisible


def test_decay_does_not_collapse_the_prior():
    """Regression: Poisson variance scales with w^2, not w.

    Treating decayed exposure as raw exposure over-states the noise, drives the fitted
    between-player variance negative, and clamps every prior to its floor — so every player
    gets an identical rate with zero confidence. On real data this silently produced 0.417
    goals/90 for the entire league. Kish rescaling (e^2/e2) is what prevents it.
    """
    rows = []
    for element, goals in ((1, 1), (2, 0)):        # genuinely different players
        for season in ("2024-25", "2025-26"):
            for gw in range(1, 39):
                rows.append({"element": element, "season": season, "GW": gw, "minutes": 90,
                             "goals_scored": goals, "position": "FWD", "tier": 1})

    as_of = season_gw_index(pd.Series(["2024-25", "2025-26"]), pd.Series([1, 39])).iloc[1]
    out = rate_features(_history(rows), as_of=as_of, stats=["goals_scored"],
                        half_life_gws=20).set_index("element")

    assert out["goals_scored_confidence"].max() > 0.5      # was 0.0 with the bug
    assert out.loc[1, "goals_scored_per90"] > out.loc[2, "goals_scored_per90"] + 0.3


def test_uniform_weights_recover_raw_exposure():
    """Downweighting every observation equally must not destroy information.

    Kish effective exposure e^2/e2 reduces to the raw exposure under uniform weights, so a
    very long half-life and a short one give the same confidence for an ever-present.
    """
    rows = [{"element": 1, "season": "2025-26", "GW": gw, "minutes": 90,
             "goals_scored": 1, "position": "FWD", "tier": 1} for gw in range(1, 21)]
    rows += [{"element": 2, "season": "2025-26", "GW": gw, "minutes": 90,
              "goals_scored": 0, "position": "FWD", "tier": 1} for gw in range(1, 21)]
    as_of = season_gw_index(pd.Series(["2025-26"]), pd.Series([21])).iloc[0]

    slow = rate_features(_history(rows), as_of=as_of, stats=["goals_scored"],
                         half_life_gws=10_000).set_index("element")
    assert slow.loc[1, "exposure_90s"] == pytest.approx(20.0, rel=0.02)


def test_price_tier_buckets_within_range():
    tiers = price_tier(pd.Series([4.0, 5.5, 7.0, 9.0, 12.0, 15.0]), tiers=3)
    assert tiers.min() == 0 and tiers.max() == 2
    assert tiers.is_monotonic_increasing
