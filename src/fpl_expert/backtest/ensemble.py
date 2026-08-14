"""Run a season many times over PERTURBED DECISION PATHS, and compare distributions.

Why this exists
---------------

A season replay is one draw from a chaotic process. Perturbing the forecast by 0.1% — far
smaller than any modelling change ever tested here — moves a 2024-25 season total with a
standard deviation of 38 points and a range of 90, while the number of transfers barely
changes (64, 64, 64, 64, 64, 65). Nothing about the policy differs. One near-tie flips, a
different player is bought, and every subsequent decision is made from a different squad.

That noise floor is larger than most effects this project measures. It is why the horizon
(+111 / -174 / +15), valuation smoothing (-88 / +182 / +78), a corrected hit bar
(-54 / +11 / +128) and Free Hit (-11 / -16 / +11) all failed the per-season sign test on the
same day: they were never resolvable from single replays.

This is ground rule 1 — "never conclude from a single deterministic replay" — applied to the
DECISION path. `repeat_sim` resamples OUTCOMES while holding decisions fixed, which is the
opposite axis, and is exactly why it never exposed this.

Pairing is the point
--------------------

Averaging k paths cuts the standard error by sqrt(k), which helps. Differencing variants
*within* a shared perturbation helps far more: both variants meet identical jittered forecasts,
so the enormous common swing cancels and only the effect of the policy survives. Always compare
through `compare_paths`, never by differencing two independently-run means.

What the perturbation is, and is not
------------------------------------

It is a path randomiser, not a robustness test. At the default 0.1% the signal-to-noise ratio
is so large that decision quality is unaffected — the forecast is, to any practical purpose,
the same forecast. Raise `jitter` to a few percent and the question silently changes to "how
well does this policy tolerate a worse forecast", which is a legitimate question but a
different one. Realised outcomes are never touched.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .season_sim import simulate_season

log = logging.getLogger(__name__)

# Small enough that decision quality is untouched, large enough to break near-ties and explore
# the path space. Measured: 0.001 gives a season-total sd of ~38 points.
DEFAULT_JITTER = 0.001
DEFAULT_PATHS = 10


def _multipliers(forecasts: pd.DataFrame, jitter: float, seed: int) -> pd.Series:
    """One perturbation per player-gameweek, shared by every frame that mentions it.

    Keyed on `(gw, player_id)` rather than drawn per frame, because a decision week and the
    six horizon views that also contain that gameweek must agree. Independent draws would not
    be an alternative forecast, just noise layered on itself.
    """
    rng = np.random.default_rng(seed)
    key = pd.MultiIndex.from_arrays(
        [forecasts["gw"], forecasts["player_id"]], names=["gw", "player_id"]
    ).unique()
    return pd.Series(1.0 + rng.normal(0.0, jitter, len(key)), index=key)


def _apply(frame: pd.DataFrame, multipliers: pd.Series, points_col: str) -> pd.DataFrame:
    key = pd.MultiIndex.from_arrays([frame["gw"], frame["player_id"]])
    factors = multipliers.reindex(key).to_numpy()
    out = frame.copy()
    out[points_col] = out[points_col].to_numpy() * np.where(
        np.isnan(factors), 1.0, factors
    )
    return out


def perturb(
    forecasts: pd.DataFrame,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None,
    *,
    jitter: float = DEFAULT_JITTER,
    seed: int = 0,
    points_col: str = "expected_points",
) -> tuple[pd.DataFrame, dict[int, dict[int, pd.DataFrame]] | None]:
    """A coherently jittered copy of a season's forecasts. Outcomes are untouched."""
    multipliers = _multipliers(forecasts, jitter, seed)
    shifted = _apply(forecasts, multipliers, points_col)
    if lookahead is None:
        return shifted, None
    return shifted, {
        as_of: {gw: _apply(frame, multipliers, points_col) for gw, frame in window.items()}
        for as_of, window in lookahead.items()
    }


def ensemble_season(
    forecasts: pd.DataFrame,
    *,
    label: str = "model",
    paths: int = DEFAULT_PATHS,
    jitter: float = DEFAULT_JITTER,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
    points_col: str = "expected_points",
    **settings,
) -> pd.DataFrame:
    """Replay one season over `paths` perturbed decision paths. One row per path.

    `seed` is the path index, so two variants run with the same `paths` see byte-identical
    perturbations and can be differenced pairwise. That is not an incidental property — it is
    the only reason this instrument resolves anything at n=3.
    """
    rows = []
    for seed in range(paths):
        shifted, shifted_lookahead = perturb(
            forecasts, lookahead, jitter=jitter, seed=seed, points_col=points_col
        )
        result = simulate_season(
            shifted, label=label, lookahead=shifted_lookahead, points_col=points_col,
            **settings,
        )
        summary = result.summary()
        rows.append({
            "label": label, "path": seed,
            "points": float(summary["total_points"]),
            "transfers": int(summary["transfers"]),
            "hits": int(summary["hits_taken"]),
            "captain_points": float(summary["captain_points"]),
        })
        log.info("%s path %d: %.0f", label, seed, rows[-1]["points"])
    return pd.DataFrame(rows)


def compare_paths(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each variant against a baseline, differenced WITHIN each shared path.

    The unpaired standard deviation of a season total is ~38 points, so an unpaired comparison
    of two variants over 10 paths each has a standard error near 17 — bigger than most effects
    worth chasing. Pairing removes the perturbation both variants shared, and what is left is
    the policy difference.

    `wins` is the share of paths in which the variant came out ahead. Unlike the mean it makes
    no assumption about the shape of the difference, which matters here because season totals
    are visibly not normal — a single flipped transfer produces a bimodal spread.
    """
    if "season" in runs.columns:
        wide = runs.pivot_table(
            index=["season", "path"], columns="label", values="points"
        )
    else:
        wide = runs.pivot_table(index="path", columns="label", values="points")
    if baseline not in wide.columns:
        raise KeyError(f"no variant named {baseline!r}")

    rows = []
    for variant in wide.columns:
        difference = (wide[variant] - wide[baseline]).dropna()
        n = len(difference)
        standard_error = (
            float(difference.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan")
        )
        rows.append({
            "variant": variant,
            "mean": round(float(wide[variant].mean()), 1),
            "sd": round(float(wide[variant].std(ddof=1)), 1) if n > 1 else float("nan"),
            "mean_diff": round(float(difference.mean()), 1),
            "se": round(standard_error, 1),
            "ci_low": round(float(difference.mean()) - 1.96 * standard_error, 1),
            "ci_high": round(float(difference.mean()) + 1.96 * standard_error, 1),
            "wins": round(float((difference > 0).mean()), 2),
            "paths": n,
        })
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False)


def per_season_paired(runs: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Paired difference WITHIN each season, which is usually the informative cut.

    A pooled interval mixes two sources of spread: path noise inside a season, and the genuine
    difference between seasons. When a policy helps in one season and hurts in another, pooling
    averages them into a small number with a wide interval and reports "no effect" — when the
    truth is a large effect whose SIGN depends on the season. Those two situations call for
    completely different responses, and only this cut separates them.
    """
    if "season" not in runs.columns:
        raise KeyError("runs carry no season column")

    rows = []
    for season, block in runs.groupby("season", sort=True):
        table = compare_paths(block, baseline)
        table.insert(0, "season", season)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def resolvable(runs: pd.DataFrame, baseline: str, per_season: bool = True) -> pd.DataFrame:
    """Add the one column that decides whether a result may be adopted.

    A paired interval that excludes zero says the effect is real ACROSS THESE SEASONS. It does
    not say the effect will hold next season — for that the sign still has to agree in every
    season, which is ground rule 2 and is not superseded by any amount of path averaging.
    """
    pooled = compare_paths(runs, baseline).set_index("variant")
    pooled["excludes_zero"] = (pooled["ci_low"] > 0) | (pooled["ci_high"] < 0)

    if not per_season or "season" not in runs.columns:
        return pooled.reset_index()

    signs = {}
    for season, block in runs.groupby("season"):
        column = compare_paths(block, baseline).set_index("variant")["mean_diff"]
        signs[season] = np.sign(column)
    sign_table = pd.DataFrame(signs)
    pooled["seasons_agree"] = (sign_table.abs().sum(axis=1) > 0) & (
        sign_table.apply(lambda r: len(set(r[r != 0])) <= 1, axis=1)
    )
    pooled["adoptable"] = pooled["excludes_zero"] & pooled["seasons_agree"]
    return pooled.join(sign_table.add_prefix("sign_")).reset_index()
