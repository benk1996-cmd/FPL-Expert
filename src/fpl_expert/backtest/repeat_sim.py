"""Error bars for season simulations, by replaying each season many times.

Every measurement in this project so far has been a single number. One set of forecasts meets
one set of realised outcomes and returns, say, 2646 points. Nothing in that says whether 2646
is reliably better than 2643 — and decisions have been made on gaps that size. Three seasons
of one deterministic replay each is three data points, and the temptation to read a consistent
ordering across three seasons as evidence is exactly how a project ends up tuned on noise.

**What is resampled.** The realised season stays the point estimate; it is the only thing that
actually happened. What varies here is the *outcome draw*: each replay redraws every player's
gameweek score from that player's own points distribution
(`models/distribution.py`). The strategy still makes its decisions on the forecast alone,
knowing nothing about the draw, so what the spread measures is how much a season's result
depends on which way the coin-flips fell.

**Why that is the right question and not a circular one.** This does NOT test whether the
forecasts are accurate — it cannot, because it grades them against their own beliefs. Point
that out and the objection is answered by what the numbers are used for: the comparison
between two *strategies* under a shared set of outcomes. A strategy that wins under the
model's own view of the world in 52% of draws has not been shown to be better than one that
wins 48%, whatever a single realised season said. Absolute totals from this procedure are
worth nothing; differences between variants are worth a great deal.

## What this instrument does NOT establish (independent review, 2026-08-12)

`rescore` is linear in the drawn points and every draw has mean equal to the pmf mean, so the
expected difference between two variants is a DETERMINISTIC number — the model's own forecast
of which strategy it prefers. The draws estimate it; they do not test it. Confidence intervals
here are Monte-Carlo error, and because the planner picks its decisions to maximise the very
forecast being resampled, a favourable result is close to tautological for any variant that
optimises harder.

`forecast_implied_difference` computes the same quantity exactly and cheaply, and is the
honest way to report it. To learn whether a strategy is genuinely better it must be graded on
something it did not optimise against: realised outcomes (n=3 seasons) or a bootstrap over
realised player-gameweeks. Conclusions previously drawn through this module — the Free Hit
reversal above all — rest on the weaker footing and are flagged in `DECISIONS.md`.

**Pairing is what makes it cheap.** Every variant is scored against the SAME resampled
outcomes within a draw, so the comparison is paired and the enormous season-to-season variance
cancels out of the difference. Comparing independent runs would need orders of magnitude more
draws to resolve the same gap. This is the difference between "chips are worth 216 +/- 180"
and "chips are worth 216 +/- 25".

**Ranks are resampled too.** The field is scored on the same drawn outcomes, so rank stays
internally consistent — our squad and the 20,000 rivals live in one world per draw. The field
seed is varied as well, since two different fields produce two different ranks for identical
points, and that variation belongs in the error bar rather than outside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models.distribution import PointsDistribution, gameweek_distributions
from .field_sim import (
    DEFAULT_FIELD_SIZE,
    build_field,
    points_matrix,
    rank_metrics,
)
from .season_sim import rescore, simulate_season

log = logging.getLogger(__name__)

# Above this share of saturated draws, rank is suppressed entirely rather than reported from
# whichever draws happened to fall inside the field. A tenth is already generous: the
# surviving draws are the unluckiest seasons, so the surviving ranks are the worst ones.
MAX_SATURATION = 0.10


def ablation_variants(forecasts: pd.DataFrame) -> dict[str, dict]:
    """Variants that degrade one scoring component at a time, for the paired instrument.

    The ablation in `season_sim` answers "which components carry the decisions" with one
    deterministic replay per season. Re-measured on corrected data that turned out to be
    almost entirely noise — every component except attack changed SIGN between seasons — so
    the question needs the error bars this module provides rather than a bigger table.

    Each variant subtracts a component from `expected_points`, exactly as `ablate` does, but
    delivered as a variant dict so every one meets identical resampled outcomes.
    """
    from .season_sim import ABLATIONS

    variants: dict[str, dict] = {"full": {}}
    for name, columns in ABLATIONS.items():
        present = [c for c in columns if c in forecasts.columns]
        if present:
            variants[name] = {"_drop_components": present}
    return variants


# The chip sets with open questions. Kept here rather than in the caller so that the variants
# a result was measured under travel with the code that measured them.
STANDARD_VARIANTS: dict[str, dict] = {
    "bb_tc": {"use_chips": True, "allowed_chips": ("bench_boost", "triple_captain")},
    "bb_tc_fh": {
        "use_chips": True,
        "allowed_chips": ("bench_boost", "triple_captain", "free_hit"),
    },
    "bb_tc_wc": {
        "use_chips": True,
        "allowed_chips": ("bench_boost", "triple_captain", "wildcard"),
    },
    "all_chips": {"use_chips": True, "allowed_chips": None},
    "no_chips": {"use_chips": False},
}


@dataclass
class RepeatResult:
    """Per-draw outcomes, long format: one row per (draw, variant)."""

    draws: pd.DataFrame
    n_draws: int
    season: str = ""

    def summary(self) -> pd.DataFrame:
        """Mean and spread per variant.

        Absolute totals are model-relative and mean nothing on their own; see the module
        docstring. Rank is reported only where the simulated field was strong enough to
        contain us — currently it is not, and the column is dropped rather than filled with
        an extrapolation.
        """
        grouped = self.draws.groupby("variant")
        out = grouped.agg(points=("points", "mean"), points_sd=("points", "std"))
        # The standard error of the MEAN, which is what says whether more draws would move it.
        out["points_se"] = out["points_sd"] / np.sqrt(self.n_draws)

        saturated = float(self.draws.get("field_saturated", pd.Series([False])).mean())
        if saturated:
            log.warning(
                "field saturated in %.0f%% of draws — the simulated field is too weak to "
                "rank against", saturated * 100,
            )
        # A median taken over only the draws that happened not to saturate is a median over
        # the unluckiest seasons, which is worse than reporting nothing at all.
        if saturated <= MAX_SATURATION and self.draws["rank"].notna().any():
            out["rank"] = grouped["rank"].median()
        return out.sort_values("points", ascending=False).round(1)


def resample_actuals(
    forecasts: pd.DataFrame,
    distribution: PointsDistribution,
    keys: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One synthetic season: every player-gameweek score redrawn from its own distribution.

    Returns a copy of `forecasts` with `actual_points` replaced. Everything the strategy reads
    to make decisions is untouched, so the decisions are made in ignorance of the draw — which
    is the only way the exercise means anything.
    """
    drawn = keys.copy()
    drawn["resampled_points"] = distribution.sample(rng)

    merge_on = [c for c in keys.columns if c in forecasts.columns]
    out = forecasts.merge(drawn, on=merge_on, how="left")
    # A player with no distribution row (absent from the fixture-level frame for that week)
    # keeps a score of zero rather than his real one — carrying the realised value through
    # would mix two different worlds in a single season.
    out["actual_points"] = out["resampled_points"].fillna(0.0)
    return out.drop(columns=["resampled_points"])


def repeat_season(
    forecasts: pd.DataFrame,
    fixture_level: pd.DataFrame,
    *,
    variants: dict[str, dict] | None = None,
    n_draws: int = 30,
    seed: int = 0,
    field_size: int = DEFAULT_FIELD_SIZE,
    field_seeds: int = 2,
    rules: dict | None = None,
    season: str = "",
) -> RepeatResult:
    """Replay one season `n_draws` times, scoring every variant on each draw.

    `forecasts` is gameweek level (what the simulator consumes) and `fixture_level` is what it
    was aggregated from — the distribution needs the finer grain so double gameweeks convolve
    rather than being averaged away.
    """
    variants = variants or STANDARD_VARIANTS
    keys, distribution = gameweek_distributions(fixture_level, rules)

    # Each variant is SOLVED once. Every decision the simulator makes reads forecast columns
    # only — outcomes enter at scoring and nowhere else — so re-solving per draw would repeat
    # identical work. On a season with the rebuild chips enabled that is the difference
    # between seconds and an overnight job.
    traces = {}
    for name, settings in variants.items():
        settings = dict(settings)
        # A component ablation changes the FORECAST the variant decides on, not its settings.
        # Handled here so ablations and chip sets can be compared through one code path.
        dropped = settings.pop("_drop_components", None)
        frame = forecasts
        if dropped:
            frame = forecasts.copy()
            frame["expected_points"] = frame["expected_points"] - frame[dropped].sum(axis=1)
        result = simulate_season(frame, label=name, rules=rules, **settings)
        traces[name] = {
            "trace": result.trace,
            "transfers": int(result.gameweeks["transfers_made"].sum()),
            "hits": int(result.gameweeks["hit_cost"].sum() / 4),
        }
        log.info("solved %s: %d gameweeks decided", name, len(result.trace))

    # Fields are drawn up front for the same reason the variants are solved up front: who a
    # simulated manager holds depends on ownership and forecasts, never on outcomes. Drawing
    # one takes about 96 seconds; scoring an already-drawn one takes 0.06. Several seeds are
    # kept and rotated so that the spread between different fields — two fields give two ranks
    # for one score — stays inside the error bar rather than outside it.
    fields = [
        build_field(forecasts, field_size, seed=seed * 1000 + i) for i in range(field_seeds)
    ]
    log.info(
        "drew %d field(s) of %d managers; resampling %d player-gameweeks "
        "over %d draws x %d variants",
        len(fields), field_size, len(distribution), n_draws, len(variants),
    )

    rows = []
    for draw in range(n_draws):
        # One generator per draw, seeded from the draw index: reproducible, and independent of
        # how many variants happen to be measured.
        rng = np.random.default_rng([seed, draw])
        sampled = resample_actuals(forecasts, distribution, keys, rng)
        outcomes = dict(
            zip(
                zip(sampled["gw"], sampled["player_id"], strict=True),
                sampled["actual_points"],
                strict=True,
            )
        )

        # The field meets the same outcomes we do — our squad and its twenty thousand rivals
        # live in one world per draw, which is what keeps rank internally consistent.
        trace = fields[draw % len(fields)]
        field = trace.score(
            points_matrix(sampled, trace.player_ids, trace.gameweeks, "actual_points")
        )

        for name, solved in traces.items():
            total = rescore(solved["trace"], outcomes)
            metrics = rank_metrics(total, field)
            rows.append({
                "draw": draw,
                "variant": name,
                "points": total,
                "transfers": solved["transfers"],
                "hits": solved["hits"],
                # None when we beat every simulated manager: the field is currently too weak
                # to rank against, and inventing a number there would flatter us in exactly
                # proportion to how inadequate it is. See `field_sim`.
                "rank": metrics["rank_in_11m"],
                "field_saturated": metrics["saturated"],
            })
        if (draw + 1) % 10 == 0:
            log.info("draw %d/%d complete", draw + 1, n_draws)

    return RepeatResult(pd.DataFrame(rows), n_draws, season)


def forecast_implied_difference(traces: dict, distribution, keys: pd.DataFrame) -> pd.DataFrame:
    """The exact quantity `repeat_season` estimates, computed in closed form.

    **This is the finding that most changes how the resampled numbers should be read.**
    `rescore` is linear in the drawn points, and each draw has mean equal to that
    player-gameweek's pmf mean, so

        E[rescore(variant)] = sum of dist_mean over the variant's scoring ids  -  hits

    is a DETERMINISTIC number. The Monte-Carlo apparatus estimates it; the confidence interval
    `paired_comparison` reports is the sampling error of that estimate, not uncertainty about
    whether one strategy beats another. Two variants differing by "+16.2 [+14.5, +17.9] over
    600 draws" differ by exactly +16.2 in the model's own view, and no number of draws makes
    that a statement about the world.

    Worse, the estimand is the model's own forecast, and the planner CHOSE its decisions to
    maximise that forecast. A rebuild chip that the planner plays only when it forecasts a
    gain will show a positive resampled difference close to tautologically.

    Use this to get the number cheaply and honestly labelled. To learn whether a strategy is
    actually better, grade it on something it did not optimise against — realised outcomes
    (n=3 seasons, wide) or a bootstrap over realised player-gameweeks.
    """
    means = pd.Series(distribution.mean(), index=pd.MultiIndex.from_frame(keys))
    lookup = means.groupby(level=["gw", "player_id"]).sum() if "gw" in keys.columns else means

    rows = []
    for name, solved in traces.items():
        total = 0.0
        for week in solved["trace"]:
            gw = week["gw"]
            total += sum(float(lookup.get((gw, pid), 0.0)) for pid in week["scoring_ids"])
            if week["captain_id"] is not None:
                total += week["captain_multiplier"] * float(
                    lookup.get((gw, week["captain_id"]), 0.0)
                )
            total -= week["hit"]
        rows.append({"variant": name, "forecast_implied_points": total})
    return pd.DataFrame(rows).sort_values("forecast_implied_points", ascending=False)


def bootstrap_realised(
    forecasts: pd.DataFrame,
    traces: dict,
    *,
    n_draws: int = 400,
    seed: int = 0,
    block: int = 4,
) -> pd.DataFrame:
    """Compare variants on REALISED outcomes, resampled in blocks of gameweeks.

    The honest instrument. `repeat_season` draws outcomes from the model's own pmf, so it
    estimates the model's forecast of which strategy it prefers — a quantity the planner chose
    its decisions to maximise. This resamples the SEASON instead: gameweeks are drawn with
    replacement in contiguous blocks, and every variant is scored on the same drawn gameweeks
    using what actually happened.

    Nothing here is graded against the model's beliefs, so a variant only wins if it really
    collected more points in the weeks it was given. Blocks of four preserve the short-range
    structure that matters — fixture runs, injury spells, and the double gameweeks chips are
    timed around — which independent weekly draws would destroy.

    The limitation is honest and unavoidable: there are only three seasons of real outcomes, so
    this measures "would this have won on rearrangements of what happened", not "will it win
    next season". That is still a different and better question than the pmf resampler answers.
    """
    rng = np.random.default_rng(seed)
    outcomes = dict(
        zip(
            zip(forecasts["gw"], forecasts["player_id"], strict=True),
            forecasts["actual_points"].fillna(0.0),
            strict=True,
        )
    )

    by_gw = {name: {} for name in traces}
    gameweeks = sorted({w["gw"] for solved in traces.values() for w in solved["trace"]})
    for name, solved in traces.items():
        for week in solved["trace"]:
            total = sum(outcomes.get((week["gw"], p), 0.0) for p in week["scoring_ids"])
            if week["captain_id"] is not None:
                total += week["captain_multiplier"] * outcomes.get(
                    (week["gw"], week["captain_id"]), 0.0
                )
            by_gw[name][week["gw"]] = total - week["hit"]

    starts = np.arange(len(gameweeks))
    n_blocks = int(np.ceil(len(gameweeks) / block))
    rows = []
    for draw in range(n_draws):
        chosen = rng.choice(starts, size=n_blocks)
        picked = [
            gameweeks[i]
            for start in chosen
            for i in range(start, min(start + block, len(gameweeks)))
        ]
        for name in traces:
            rows.append({
                "draw": draw, "variant": name,
                "points": float(sum(by_gw[name].get(gw, 0.0) for gw in picked)),
            })
    return pd.DataFrame(rows)


def paired_comparison(draws: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each variant against a baseline, differenced WITHIN each draw.

    The paired difference is the whole point. Season totals swing by well over a hundred
    points between draws, and that swing is shared by every variant facing the same outcomes,
    so differencing removes it. What survives is the effect of the decisions themselves.

    **Read `mean_diff` and its interval with care.** The mean difference is deterministic
    given the decisions — see `forecast_implied_difference` — so `se`, `ci_low` and `ci_high`
    describe Monte-Carlo error on estimating the model's own forecast, NOT evidence that one
    strategy beats another. An independent review established this and showed that the guard
    previously used to defend it (a claimed-uniform gap between realised and resampled totals)
    does not hold: the spread is ~49 points and runs against the variants that optimise
    hardest against the forecast.

    `win_rate` is the one column here that is not deterministic — it is the share of drawn
    worlds in which the variant comes out ahead, and it does carry information about how much
    outcome luck separates two strategies. It is still measured under the model's own beliefs.
    """
    wide = draws.pivot(index="draw", columns="variant", values="points")
    if baseline not in wide.columns:
        raise KeyError(f"no variant named {baseline!r}")

    rows = []
    for variant in wide.columns:
        difference = wide[variant] - wide[baseline]
        n = len(difference)
        standard_error = float(difference.std(ddof=1)) / np.sqrt(n) if n > 1 else np.nan
        rows.append({
            "variant": variant,
            "mean_diff": float(difference.mean()),
            "sd_diff": float(difference.std(ddof=1)) if n > 1 else np.nan,
            "se": standard_error,
            # 1.96 rather than a t quantile: with 30+ draws the difference is immaterial,
            # and pretending to more precision than the resampling itself has would be false.
            "ci_low": float(difference.mean()) - 1.96 * standard_error,
            "ci_high": float(difference.mean()) + 1.96 * standard_error,
            "win_rate": float((difference > 0).mean()),
            "draws": n,
        })
    return pd.DataFrame(rows).sort_values("mean_diff", ascending=False).round(2)


def is_resolved(comparison: pd.DataFrame, variant: str) -> bool:
    """Whether a variant's advantage over the baseline is distinguishable from zero.

    Provided so the conclusion is a computed property rather than a judgement made afresh each
    time a table is read. A confidence interval spanning zero means the comparison has not
    been settled — not that the variants are equal.
    """
    row = comparison[comparison["variant"] == variant]
    if row.empty:
        raise KeyError(f"no variant named {variant!r}")
    return bool(row["ci_low"].iloc[0] > 0 or row["ci_high"].iloc[0] < 0)


def draws_needed(comparison: pd.DataFrame, variant: str, target_gap: float) -> int:
    """How many draws it would take to resolve a gap of `target_gap` points.

    Answers the question a table of overlapping intervals immediately raises — is this worth
    more compute, or is the effect too small to matter? Scales as the square of the ratio, so
    resolving a gap half as large costs four times as much.
    """
    row = comparison[comparison["variant"] == variant]
    if row.empty:
        raise KeyError(f"no variant named {variant!r}")
    spread = float(row["sd_diff"].iloc[0])
    if not np.isfinite(spread) or target_gap <= 0:
        return 0
    return int(np.ceil((1.96 * spread / target_gap) ** 2))
