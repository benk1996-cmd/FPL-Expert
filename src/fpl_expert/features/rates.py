"""Per-90 rate estimation with recency weighting and empirical-Bayes shrinkage.

The problem this solves: a striker who scored twice in 40 minutes has an observed rate of
4.5 goals per 90. Taken at face value that is the best rate in the league; in truth it is
almost entirely noise. Meanwhile a player with no Premier League history at all has no
observed rate, and naive code either drops him or treats him as a zero — both wrong.

Both cases are the same statistical problem, and one estimator handles them.

Model: for player i, counts of an event arrive as

    y_i ~ Poisson(lambda_i * e_i)

where `e_i` is exposure in 90-minute units and `lambda_i` is the player's true per-90 rate.
Put a Gamma(alpha, beta) prior on lambda, fitted from the population of comparable players
(same position, similar price). Gamma is conjugate to Poisson, so the posterior mean is

    lambda_hat = (alpha + y_i) / (beta + e_i)

which is exactly the shrunk rate we want, and reads naturally: `beta` is a number of
*pseudo-90s* of prior evidence. A player must accumulate roughly `beta` full matches before
his own record outweighs the prior.

The two hard cases fall out for free rather than needing special-casing:

    e_i = 0   (a promoted-club player, a new signing)  ->  lambda_hat = alpha/beta,
                                                            i.e. the price/position prior
    e_i small (the two-goal cameo)                     ->  heavily shrunk toward that prior

`confidence = e_i / (e_i + beta)` is the share of the estimate coming from the player's own
record: 0 for a total unknown, approaching 1 for an ever-present. That is the low-confidence
flag the optimiser uses to avoid over-trusting newcomers.

Recency is folded into the same machinery by exponentially decaying counts and exposure
together, so a rate is always counts-over-exposure on a consistent weighting.

One correction matters and is easy to get wrong. A decayed count `c = sum(w_t * y_t)` has
variance `lambda * sum(w_t^2 * x_t)`, not `lambda * sum(w_t * x_t)` — Poisson variance
scales with the square of the weight. Treating decayed exposure as if it were raw exposure
over-states the Poisson noise, which drives the fitted between-player variance negative and
collapses every prior onto its floor: every player then receives an identical rate with
zero confidence. So counts and exposure are rescaled to their Kish effective equivalents

    effective exposure = e^2 / e2      effective count = c * e / e2

where `e = sum(w*x)` and `e2 = sum(w^2*x)`. After this rescaling the weighted data behaves
like ordinary Poisson data and everything downstream is standard. Note the reduction cases:
with no decay `e2 == e` and the transform is the identity, and with *uniform* weights the
effective exposure recovers the raw exposure — downweighting everything equally should not
destroy information, and it does not. Only unequal weights cost effective sample size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MINUTES_PER_90 = 90.0

# Floor on prior variance. If a population looks perfectly homogeneous (or sampling noise
# swamps the real spread) the moment estimate of variance can come out <= 0, which would
# make the Gamma degenerate. Falling back to a tiny variance means very strong shrinkage,
# which is the right default when we cannot detect real differences between players.
MIN_PRIOR_VARIANCE = 1e-6


@dataclass(frozen=True)
class GammaPrior:
    """Gamma(alpha, beta) prior on a per-90 rate. `beta` is in 90-minute units."""

    alpha: float
    beta: float
    between_player_variance: float = 0.0
    # Whether real between-player variation survived the noise correction by a margin
    # larger than the correction's own sampling error. False means the population's spread
    # is fully explained by chance, so every player is estimated at the group mean. That is
    # a legitimate answer, not a failure — but it must be visible, because a collapsed prior
    # and a genuine signal look identical downstream.
    skill_detected: bool = False

    @property
    def mean(self) -> float:
        return self.alpha / self.beta

    @property
    def pseudo_90s(self) -> float:
        """Exposure at which a player's own record carries equal weight to the prior."""
        return self.beta


def season_gw_index(season: pd.Series, gw: pd.Series) -> pd.Series:
    """A single monotonic gameweek axis across seasons, so decay spans season boundaries.

    Seasons sort lexicographically ('2019-20' < '2020-21'), so ranking their codes gives a
    stable ordering without hardcoding a calendar.
    """
    order = {s: i for i, s in enumerate(sorted(season.dropna().unique()))}
    return season.map(order) * 38 + gw


def decay_weights(index: pd.Series, as_of: float, half_life_gws: float) -> pd.Series:
    """Exponential recency weights. Rows at or after `as_of` get zero weight.

    The hard zero matters: it is what stops a feature for gameweek t from being built out
    of gameweek t's own result.
    """
    age = as_of - index
    weights = np.where(age > 0, 0.5 ** (age / half_life_gws), 0.0)
    return pd.Series(weights, index=index.index)


# How much sampling noise a statistic actually carries, relative to the Poisson assumption.
# MEASURED (2026-08-09) by splitting each player's matches into two halves and comparing the
# variance between the two independent estimates against the Poisson expectation:
#
#     goals_scored 0.92   assists 0.83   expected_goals 0.29   expected_assists 0.17
#
# Counts behave as Poisson, as they should. Expected-goals measures do NOT: xG is a sum of
# probabilities, not a count of events, so it is far more stable match to match. Applying a
# full Poisson correction to xG over-subtracts, drives the fitted between-player variance
# negative, and collapses every prior onto its floor — the same failure mode as the missing
# Kish correction, and just as plausible-looking in the output.
NOISE_SCALE = {
    "goals_scored": 1.0,
    "assists": 0.85,
    "expected_goals": 0.30,
    "expected_assists": 0.20,
    "expected_goal_involvements": 0.25,
    "saves": 1.0,
    "defensive_contribution": 1.0,
}
DEFAULT_NOISE_SCALE = 1.0


def fit_gamma_prior(
    counts: np.ndarray, exposure: np.ndarray, noise_scale: float = DEFAULT_NOISE_SCALE
) -> GammaPrior:
    """Fit a Gamma prior to a population of players by moment matching.

    The subtlety is that the observed spread of per-90 rates is inflated by sampling noise,
    badly so for low-minute players. Naively matching the variance of observed rates would
    produce a far too wide prior and almost no shrinkage — the exact failure this module
    exists to prevent. So the expected noise component is subtracted off:

        Var(observed rate) = Var(true rate) + noise_scale * E[lambda / e]

    `noise_scale` is 1.0 for true counts and lower for xG-style measures; see `NOISE_SCALE`.
    Getting it wrong in either direction is costly: too high collapses the prior, too low
    leaves hot streaks unshrunk.
    """
    counts, exposure = np.asarray(counts, float), np.asarray(exposure, float)
    keep = exposure > 0
    counts, exposure = counts[keep], exposure[keep]
    if counts.size == 0 or exposure.sum() <= 0:
        # Nothing to learn from — a weak, near-zero prior that shrinks everything hard.
        return GammaPrior(alpha=MIN_PRIOR_VARIANCE, beta=1.0, between_player_variance=0.0)

    total_exposure = exposure.sum()
    mean = counts.sum() / total_exposure
    rates = counts / exposure

    # Exposure-weighted spread of observed rates, then remove the Poisson contribution.
    weighted_spread = float((exposure * (rates - mean) ** 2).sum())
    noise_component = noise_scale * mean * counts.size
    between_player_var = max(
        (weighted_spread - noise_component) / total_exposure, MIN_PRIOR_VARIANCE
    )

    # The correction itself is an estimate with its own sampling error: under a pure-noise
    # null the weighted spread has standard deviation of roughly mean*sqrt(2n). Declaring
    # "real variation" on anything smaller than a couple of those would find skill in pure
    # chance — which is exactly the trap in claiming players have finishing ability.
    noise_sd = mean * np.sqrt(2.0 * counts.size)
    detected = (weighted_spread - noise_component) > 2.0 * noise_sd
    if mean <= 0:
        return GammaPrior(alpha=MIN_PRIOR_VARIANCE, beta=1.0, between_player_variance=0.0)

    beta = mean / between_player_var
    return GammaPrior(
        alpha=mean * beta, beta=beta,
        between_player_variance=between_player_var, skill_detected=bool(detected),
    )


def shrink(counts: pd.Series, exposure: pd.Series, prior: GammaPrior) -> pd.DataFrame:
    """Posterior mean rate and a confidence weight, per player."""
    rate = (prior.alpha + counts) / (prior.beta + exposure)
    confidence = exposure / (exposure + prior.beta)
    return pd.DataFrame({"rate": rate, "confidence": confidence})


def price_tier(price: pd.Series, tiers: int = 4) -> pd.Series:
    """Bucket players by price within their position.

    Price is FPL's own estimate of expected output — set by the game's editors and adjusted
    by the market. For a player with no Premier League history it is the only quality signal
    available, so it is what the prior is conditioned on.
    """
    ranked = price.rank(pct=True, method="average")
    return pd.cut(ranked, bins=tiers, labels=False, include_lowest=True)


def decayed_totals(
    history: pd.DataFrame,
    *,
    as_of: float,
    stats: list[str],
    half_life_gws: float = 20.0,
    minutes_col: str = "minutes",
    key: str = "element",
    keep_last: tuple[str, ...] = ("position", "tier", "team", "web_name", "name"),
) -> pd.DataFrame:
    """Recency-weighted counts and exposure per player, Kish-rescaled.

    Shared by every rate-based model so the `w^2` correction lives in exactly one place.
    Anything consuming raw decayed sums without this rescaling will collapse its priors —
    see the module docstring.
    """
    df = history.copy()
    df["_idx"] = season_gw_index(df["season"], df["GW"])
    df["_w"] = decay_weights(df["_idx"], as_of, half_life_gws)
    df = df[df["_w"] > 0]
    if df.empty:
        raise ValueError(f"no history strictly before index {as_of}")

    raw_exposure = df[minutes_col] / MINUTES_PER_90
    df["_exposure"] = df["_w"] * raw_exposure
    df["_exposure2"] = df["_w"] ** 2 * raw_exposure

    agg = {"_exposure": "sum", "_exposure2": "sum"}
    for stat in stats:
        df[f"_c_{stat}"] = df["_w"] * df[stat].fillna(0.0)
        agg[f"_c_{stat}"] = "sum"
    # Exclude the grouping key: it becomes the index, and carrying it as a column too makes
    # any later reset_index() collide.
    carried = [c for c in keep_last if c in df.columns and c != key]
    players = df.sort_values("_idx").groupby(key).agg({**agg, **{c: "last" for c in carried}})

    e, e2 = players["_exposure"], players["_exposure2"]
    scale = (e / e2).where(e2 > 0, 0.0)
    players["_exposure"] = e * scale
    for stat in stats:
        players[f"_c_{stat}"] = players[f"_c_{stat}"] * scale
    return players


def rate_features(
    history: pd.DataFrame,
    *,
    as_of: float,
    stats: list[str],
    half_life_gws: float = 20.0,
    group_cols: tuple[str, ...] = ("position", "tier"),
    minutes_col: str = "minutes",
) -> pd.DataFrame:
    """Shrunk, recency-weighted per-90 rates for every player, as of a gameweek index.

    `history` needs one row per player-gameweek with `element`, `season`, `GW`, the stat
    columns, and whatever `group_cols` references. Only rows strictly before `as_of`
    contribute.
    """
    df = history.copy()
    df["_idx"] = season_gw_index(df["season"], df["GW"])
    df["_w"] = decay_weights(df["_idx"], as_of, half_life_gws)
    df = df[df["_w"] > 0]
    if df.empty:
        raise ValueError(f"no history strictly before index {as_of}")

    raw_exposure = df[minutes_col] / MINUTES_PER_90
    df["_exposure"] = df["_w"] * raw_exposure
    df["_exposure2"] = df["_w"] ** 2 * raw_exposure

    # One row per player: decayed counts, decayed exposure, and its second moment.
    agg = {"_exposure": "sum", "_exposure2": "sum"}
    for stat in stats:
        df[f"_c_{stat}"] = df["_w"] * df[stat].fillna(0.0)
        agg[f"_c_{stat}"] = "sum"
    keep_last = [c for c in ("position", "tier", "team", "web_name") if c in df.columns]
    players = df.sort_values("_idx").groupby("element").agg({**agg, **{c: "last" for c in keep_last}})

    # Kish rescaling — see module docstring. Without this every prior collapses.
    e, e2 = players["_exposure"], players["_exposure2"]
    scale = (e / e2).where(e2 > 0, 0.0)
    players["_exposure"] = e * scale
    for stat in stats:
        players[f"_c_{stat}"] = players[f"_c_{stat}"] * scale

    groups = [c for c in group_cols if c in players.columns]
    out = pd.DataFrame(index=players.index)
    for stat in stats:
        counts, exposure = players[f"_c_{stat}"], players["_exposure"]
        # Per-stat noise scale, not the 1.0 default. xG and xA are sums of probabilities and
        # carry far less sampling noise than counts do — passing the wrong scale here collapses
        # every prior onto its floor, which is the failure `NOISE_SCALE` exists to prevent.
        # This function was reaching the default and so silently ignoring it.
        noise = NOISE_SCALE.get(stat, DEFAULT_NOISE_SCALE)
        if groups:
            pieces = []
            for _, block in players.groupby(groups, dropna=False):
                prior = fit_gamma_prior(
                    counts.loc[block.index], exposure.loc[block.index], noise
                )
                pieces.append(shrink(counts.loc[block.index], exposure.loc[block.index], prior))
            shrunk = pd.concat(pieces).reindex(players.index)
        else:
            shrunk = shrink(counts, exposure, fit_gamma_prior(counts, exposure, noise))
        out[f"{stat}_per90"] = shrunk["rate"]
        out[f"{stat}_confidence"] = shrunk["confidence"]

    out["exposure_90s"] = players["_exposure"]
    for col in keep_last:
        out[col] = players[col]
    return out.reset_index()
