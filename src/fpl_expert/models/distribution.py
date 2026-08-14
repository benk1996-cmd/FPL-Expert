"""The full points distribution for a player in a gameweek, not just its mean.

Everything upstream of this produces `expected_points` — one number per player. That is the
right input to a linear optimiser and the wrong input to almost every other decision FPL
actually asks you to make:

* **Captaincy** is a bet on the upside tail. Two players on 6.0 expected points are not
  interchangeable when one of them returns double digits a fifth of the time and the other
  never has. `points_variance` in the assembler was a stopgap admitting exactly this, and it
  ignores covariance and every threshold rule in the game.
* **Rank** depends on beating the field, not on scoring well. That is a question about
  distributions — `P(our score - their score > 0)` — and cannot be answered with two means.
* **Error bars.** Season simulations here are deterministic: one set of forecasts against one
  set of realised outcomes gives one number with no spread. Decisions have been made on gaps
  of ten season points with nothing to say whether ten points is signal. Resampling outcomes
  from this distribution is what gives those comparisons a standard error
  (`backtest/repeat_sim.py`).

**How it is built.** Points are integers, so the exact distribution is reachable by discrete
convolution rather than simulation — no sampling noise, and fast enough to run for every
player in every gameweek. Each component contributes a small pmf over integer point offsets
and they are convolved together, conditional on the minutes bucket, then mixed over the three
buckets by their probabilities.

Conditioning on the bucket is what makes this more faithful than the assembler's variance
term, and it is not cosmetic. A player's score is strongly bimodal: the zero-minutes outcome
puts a spike at exactly 0 points, which no unimodal approximation can express, and a
clean sheet is only reachable from the 60+ bucket. Treating minutes as a scaling factor on a
single distribution — which is what multiplying rates by expected minutes does — smears that
spike into the body and understates both the floor and the ceiling.

Two components are modelled jointly rather than independently, because they are not:

* **Clean sheets and goals conceded** both derive from one quantity, the goals the team
  concedes. Modelled independently, a defender could be credited with a clean sheet and a
  conceding deduction in the same fixture. Here both come from a single count distribution,
  re-anchored so `P(concede 0)` equals the match model's clean-sheet probability.
* **Yellow and red cards** are mutually exclusive outcomes of one draw, not two.

Known omissions, all rare and all documented rather than silently dropped: penalty saves,
penalty misses and own goals. Together they are worth well under 1% of scoring and none of
them has a rate model upstream to draw on.

**Bonus** deserves its own note because it is the least trustworthy input. The assembler
produces an expected bonus; turning that back into a pmf needs the conditional shape. Measured
across 186k player-gameweeks in the archive, bonus given that any was won is almost exactly
uniform over 1/2/3 (0.332 / 0.332 / 0.336, conditional mean 2.004), so the split is uniform
and `P(any bonus) = expected_bonus / 2`. That is measured, but it inherits every weakness of
the bonus model it is derived from.

## Measured calibration, 119,305 player-gameweeks over 2022-23 to 2025-26

Predicted exceedance probabilities against realised frequencies. The first column is what a
first attempt produced — independent components, Poisson counts — and it fails in the way
that matters most, growing worse the further into the tail it goes:

    event      independent   + bonus coupling   + dispersion   realised
    P(>= 2)         0.2302             0.2289         0.2286     0.2399
    P(>= 5)         0.0838             0.0821         0.0810     0.0961
    P(>= 8)         0.0217             0.0276         0.0274     0.0424
    P(>= 10)        0.0088             0.0127         0.0131     0.0224
    P(>= 15)        0.0012             0.0024         0.0029     0.0047
    P(>= 20)        0.0003             0.0003         0.0005     0.0008

    ratio to realised, P(>=10):   0.39x  ->  0.57x  ->  0.58x
    ratio to realised, P(>=20):   0.40x  ->  0.37x  ->  0.61x

The point is not the level, it is the SHAPE. As built, the ratio sits at roughly 0.6 across
every threshold from 5 points to 20; before the two corrections it decayed from 0.95 at the
low end to 0.40 at the high end. A flat ratio is a level error and a decaying one is a
structural failure, and only the second would make a haul look impossible.

The residual level shortfall was never this module's, and it has since been traced and largely
fixed in `models/attack.py` — a penalty-mass leak that deleted 6.1% of every team's expected
goals, and an assist rate of 0.72 where the archive says 0.905. After those fixes:

    event      before      after attack fixes   after the goalkeeper fix   target
    P(>= 5)     0.81x            0.95x                   1.04x              1.0
    P(>= 8)     0.65x            0.80x                   0.84x              1.0
    P(>=10)     0.58x            0.78x                   0.81x              1.0
    P(>=15)     0.61x            0.87x                   0.86x              1.0

The third column is the current state, re-measured 2026-08-12. The forecast's overall mean
went from 0.87x realised to 1.009x over the same two rounds of fixes — the second of which
(goalkeepers being given outfield attacking rates, see `pipeline.player_rates`) mattered more
here than anything done inside this module. P(>=5) has crossed from under to slightly over.

The table of raw probabilities above is kept as it was first measured, because it is what
motivated the two corrections in THIS module.

Independence assumptions that remain, all now measured (2026-08-11) and all left in place:

* **Goals and assists are drawn independently of each other.** P(assist) is 0.0888 in a match
  where a player does not score and 0.1542 where he does — a lift of 1.74x. Real, and the
  largest of the three, but modelling it means a joint distribution over two counts where the
  present code convolves two independent ones, and the tail it would fatten is already at
  0.78-0.95 of realised after the attack fixes.
* **Clean sheets are independent of returns.** 27.7% clean-sheet rate with no involvement
  against 37.9% with two — a lift of 1.37x, next to bonus's 24x which IS modelled.
* **Defensive contributions are Poisson.** They are genuinely overdispersed (residual
  variance/mean 1.78 / 1.60 / 1.49 by position) but a negative binomial improves defenders and
  makes midfielders and forwards worse — see `models/secondary.py` for the table. Not applied.

The common thread: each is a real dependence, each is an order of magnitude smaller than the
bonus coupling that was worth modelling, and each would cost more structure than the residual
error justifies. They are recorded so the next person does not have to re-measure them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from ..config import load_scoring_rules
from .minutes import BUCKET_LONG, BUCKET_MINUTES, BUCKET_SHORT
from .secondary import RED_CARD_FRACTION

log = logging.getLogger(__name__)

MINUTES_PER_90 = 90.0

# The reported grid. Scores outside it are folded into the end bins rather than dropped, so
# the pmf always sums to one; the mass involved is negligible (a 30-point gameweek is a
# handful of occurrences in the whole archive).
GRID_LOW, GRID_HIGH = -6, 32

# Event caps. Tail mass is folded into the top cell of each, so nothing is lost — these only
# bound the width of the convolution.
GOAL_CAP, ASSIST_CAP, CONCEDED_CAP, SAVE_CAP = 4, 4, 8, 15

# Negative-binomial dispersion for goals and assists: var = mean + mean^2 / k. Estimated by
# moments against realised counts across 4 seasons of forecasts, after first correcting the
# forecast mean so that dispersion is not confounded with calibration of the mean:
#
#     goals   k = 2.85    P(2+): Poisson 0.00275  NB 0.00332  realised 0.00356
#                         P(3+): Poisson 0.00027  NB 0.00046  realised 0.00048
#     assists k = 1.88    P(2+): Poisson 0.00165  NB 0.00229  realised 0.00259
#                         P(3+): Poisson 0.00008  NB 0.00021  realised 0.00019
#
# A player's true rate is not constant from match to match — form, role and matchup all move
# it, and the shrunk per-90 rate is a posterior mean rather than a known parameter. Poisson
# assumes it away and loses roughly a third of the multi-return tail as a result. The
# correction is mean-preserving, so `build_distribution` still reproduces the assembler's
# `expected_points`; only the shape changes.
#
# NOT applied here: the mean scale factors that same fit produced (goals 1.073, assists
# 1.268). They said the attack model under-predicts returns, especially assists — a real
# finding, but one belonging to `attack.py`, and applying it here would have silently put this
# module's mean at odds with the number the optimiser sees.
#
# That judgement paid off. Both were traced to specific defects in `attack.py` and fixed at
# source: a penalty-mass leak worth exactly 6.1% of team expected goals, and `ASSIST_RATE` set
# to 0.72 where seven seasons of archive say 0.905 (0.72 x 1.268 = 0.913). Had the scale
# factors been applied here as a fudge, both defects would still be in the model that actually
# makes decisions, hidden behind a correction in the model that only describes them.
GOAL_DISPERSION, ASSIST_DISPERSION = 2.85, 1.88

# Bonus given that any was won: uniform over 1/2/3, measured at 0.332/0.332/0.336 across the
# archive. The conditional mean of 2.0 is what converts an expected bonus into a probability.
BONUS_SHAPE = np.array([1 / 3, 1 / 3, 1 / 3])
BONUS_CONDITIONAL_MEAN = 2.0

# Expected bonus given goal involvements (goals + assists) in a 60+ minute appearance,
# measured over 55,234 such appearances in the archive:
#
#     involvements        0        1        2        3
#     n              45,162    8,370    1,414      288
#     P(any bonus)    0.055    0.479    0.910    0.972
#     E[bonus]        0.095    0.955    2.248    2.691
#
# This coupling is the single largest correction in the module. Treating bonus as independent
# of returns understated `P(10+ points)` by a factor of 2.5 against realised frequencies —
# and a haul is precisely a return *plus* the bonus that reliably accompanies it. Independence
# is a defensible approximation for a mean and a bad one for a tail.
BONUS_GIVEN_INVOLVEMENT = np.array([0.095, 0.955, 2.248, 2.691])
BONUS_GIVEN_MANY = 2.85         # four or more involvements: rare, and close to the ceiling


# --- small pmf algebra ----------------------------------------------------
#
# A component is a pair `(low, arr)`: `arr[i, j]` is the probability that player `i` earns
# `low + j` points from that component.


def _poisson_counts(lam: np.ndarray, cap: int, dispersion: float | None = None) -> np.ndarray:
    """(n, cap+1) count pmf with the tail folded into the last cell.

    `dispersion` switches to a negative binomial of the same mean, for counts whose rate is
    itself uncertain. `None` keeps a plain Poisson, which is right for events driven by a
    known team-level rate rather than a noisy per-player one.
    """
    counts = np.arange(cap + 1)
    mean = np.clip(lam, 0.0, None)[:, None]
    if dispersion is None:
        out = poisson.pmf(counts[None, :], mean)
    else:
        out = nbinom.pmf(counts[None, :], dispersion, dispersion / (dispersion + mean))
    out[:, -1] += np.clip(1.0 - out.sum(axis=1), 0.0, None)
    return out


def _spread(counts: np.ndarray, step: int) -> np.ndarray:
    """Map a count pmf to a points pmf worth `step` points per event."""
    if step == 0:
        return counts.sum(axis=1, keepdims=True)
    out = np.zeros((counts.shape[0], (counts.shape[1] - 1) * step + 1))
    out[:, ::step] = counts
    return out


def _gather(counts: np.ndarray, values: np.ndarray) -> tuple[int, np.ndarray]:
    """Map a count pmf through an arbitrary, possibly non-monotone point value per count.

    Needed wherever the scoring rule is not a multiple of the count — goals conceded pays
    -1 per *two* conceded, and the clean sheet sits at the same count of zero.
    """
    low, high = int(values.min()), int(values.max())
    out = np.zeros((counts.shape[0], high - low + 1))
    for k, value in enumerate(values):
        out[:, int(value) - low] += counts[:, k]
    return low, out


def _convolve(a: tuple[int, np.ndarray], b: tuple[int, np.ndarray]) -> tuple[int, np.ndarray]:
    """Sum of two independent components, row-wise.

    A loop over the *narrower* component rather than an FFT: these pmfs are a few dozen cells
    wide, where a direct pass is both faster and exact.
    """
    (a_low, a_arr), (b_low, b_arr) = a, b
    if b_arr.shape[1] > a_arr.shape[1]:
        (a_low, a_arr), (b_low, b_arr) = (b_low, b_arr), (a_low, a_arr)
    out = np.zeros((a_arr.shape[0], a_arr.shape[1] + b_arr.shape[1] - 1))
    for j in range(b_arr.shape[1]):
        out[:, j : j + a_arr.shape[1]] += a_arr * b_arr[:, j : j + 1]
    return a_low + b_low, out


def _project(low: int, arr: np.ndarray, grid_low: int, grid_high: int) -> np.ndarray:
    """Place a component on the reporting grid, folding overflow into the end bins."""
    width = grid_high - grid_low + 1
    out = np.zeros((arr.shape[0], width))
    for j in range(arr.shape[1]):
        out[:, min(max(low + j - grid_low, 0), width - 1)] += arr[:, j]
    return out


# --- the distribution -----------------------------------------------------


@dataclass(frozen=True)
class PointsDistribution:
    """Per-player pmf over integer FPL points, on a fixed grid.

    Rows align with the frame it was built from, and `player_id` carries that frame's keys so
    the distribution can be rejoined after any grouping.
    """

    pmf: np.ndarray
    low: int = GRID_LOW
    player_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.pmf.ndim != 2:
            raise ValueError("pmf must be 2-d (players x points)")

    @property
    def support(self) -> np.ndarray:
        return np.arange(self.low, self.low + self.pmf.shape[1])

    def __len__(self) -> int:
        return self.pmf.shape[0]

    def mean(self) -> np.ndarray:
        return self.pmf @ self.support

    def variance(self) -> np.ndarray:
        centred = self.support[None, :] - self.mean()[:, None]
        return (self.pmf * centred**2).sum(axis=1)

    def std(self) -> np.ndarray:
        return np.sqrt(self.variance())

    def quantile(self, q: float) -> np.ndarray:
        """Smallest score with cumulative probability at least `q`."""
        index = (self.pmf.cumsum(axis=1) >= q).argmax(axis=1)
        return self.support[index]

    def ceiling(self, q: float = 0.90) -> np.ndarray:
        """Realistic upside. The captaincy question is about this, not about the mean."""
        return self.quantile(q)

    def floor(self, q: float = 0.10) -> np.ndarray:
        """Realistic downside — what a differential costs when it fails."""
        return self.quantile(q)

    def p_at_least(self, threshold: int) -> np.ndarray:
        return self.pmf[:, self.support >= threshold].sum(axis=1)

    def p_haul(self, threshold: int = 10) -> np.ndarray:
        """Probability of a double-digit return, the standard captaincy criterion."""
        return self.p_at_least(threshold)

    def p_blank(self, threshold: int = 2) -> np.ndarray:
        """Probability of returning nothing beyond appearance points."""
        return self.pmf[:, self.support <= threshold].sum(axis=1)

    def sample(self, rng: np.random.Generator, draws: int = 1) -> np.ndarray:
        """Draw scores. Shape (draws, n_players), or (n_players,) when `draws` is 1.

        Inverse-CDF sampling against one uniform per player per draw, so the whole panel is
        drawn in a single vectorised pass.
        """
        cdf = self.pmf.cumsum(axis=1)
        cdf[:, -1] = 1.0
        uniform = rng.random((draws, self.pmf.shape[0]))
        index = (uniform[:, :, None] > cdf[None, :, :]).sum(axis=2)
        drawn = self.support[np.clip(index, 0, self.pmf.shape[1] - 1)]
        return drawn[0] if draws == 1 else drawn

    def to_frame(self) -> pd.DataFrame:
        """Summary statistics, one row per player — the form the report and tests want."""
        return pd.DataFrame({
            "player_id": self.player_id if self.player_id is not None else np.arange(len(self)),
            "dist_mean": self.mean(),
            "dist_std": self.std(),
            "floor": self.floor(),
            "median": self.quantile(0.5),
            "ceiling": self.ceiling(),
            "p_blank": self.p_blank(),
            "p_haul": self.p_haul(),
            "p_double_haul": self.p_at_least(15),
        })


# --- construction ---------------------------------------------------------


def _bucket_distribution(
    block: pd.DataFrame, bucket: int, position: str, rules: dict
) -> tuple[int, np.ndarray]:
    """Points pmf conditional on the player landing in one minutes bucket.

    Rates arrive already multiplied by *expected* minutes, so they are unconditional
    expectations. Dividing back out and re-scaling to the bucket's typical minutes is what
    conditions them — and because `expected_minutes` upstream is itself the bucket-weighted
    average of these same constants, the mixture reproduces the assembler's expectation
    rather than drifting from it.
    """
    minutes = BUCKET_MINUTES[bucket]
    exposure = np.clip(block["expected_minutes"].to_numpy(float), 1e-6, None) / MINUTES_PER_90
    scale = (minutes / MINUTES_PER_90) / exposure

    goals = block["expected_goals"].to_numpy(float) * scale
    assists = block["expected_assists"].to_numpy(float) * scale
    bonus = block["pts_bonus"].fillna(0.0).to_numpy(float) * scale

    total = _attack_and_bonus(goals, assists, bonus, position, rules)

    # Appearance points: a constant shift of the whole distribution.
    appearance = rules["appearance"]["sixty_plus" if bucket == BUCKET_LONG else "under_60"]
    total = (total[0] + int(appearance), total[1])

    if bucket == BUCKET_LONG:
        total = _convolve(total, _defence(block, position, rules))

    if position == "GK":
        saves = block["expected_saves"].to_numpy(float) * scale
        divisor = int(rules["saves_per_point"])
        counts = np.arange(SAVE_CAP + 1)
        total = _convolve(
            total, _gather(_poisson_counts(saves, SAVE_CAP), counts // divisor)
        )

    total = _convolve(total, _defcon(block, position, rules, minutes))
    total = _convolve(total, _cards(block, rules, minutes))
    return total


def _bonus_pmf(mean: np.ndarray) -> np.ndarray:
    """(n, 4) pmf over 0-3 bonus points with the given mean.

    Below a mean of 2 the conditional shape is the archive's measured one — uniform over
    1/2/3 — so the only free quantity is how often any bonus is won. Above 2 that shape
    cannot reach the mean however often bonus is won, so the distribution slides toward the
    maximum instead. Both branches match the target mean exactly, which is what keeps the
    assembled expectation intact while the mass moves to where returns actually put it.
    """
    mean = np.clip(mean, 0.0, 3.0)
    out = np.zeros((mean.size, 4))

    low = mean <= BONUS_CONDITIONAL_MEAN
    any_bonus = mean[low] / BONUS_CONDITIONAL_MEAN
    out[low, 0] = 1.0 - any_bonus
    out[np.ix_(low, [1, 2, 3])] = any_bonus[:, None] * BONUS_SHAPE[None, :]

    high = ~low
    tilt = mean[high] - BONUS_CONDITIONAL_MEAN
    out[np.ix_(high, [1, 2, 3])] = (1.0 - tilt)[:, None] * BONUS_SHAPE[None, :]
    out[high, 3] += tilt
    return out


def _attack_and_bonus(
    goals: np.ndarray, assists: np.ndarray, bonus: np.ndarray, position: str, rules: dict
) -> tuple[int, np.ndarray]:
    """Goals, assists and bonus as one joint distribution rather than three independent ones.

    Goals and assists are drawn independently — they are separate events and modelling their
    (mild, positive) correlation is not supportable from the archive — but bonus is
    conditioned on their sum, because that dependence is overwhelming rather than mild.

    The conditional means are the archive's, rescaled per player so that the *unconditional*
    expected bonus still equals the assembler's `pts_bonus`. That is the property worth
    protecting: the mean is untouched, so nothing downstream that reads `expected_points`
    changes, while the mass moves onto the outcomes where bonus is really earned. The rescale
    is clipped at three points per fixture, which is the actual maximum, so a heavy-bonus
    player's mean can fall slightly short of target — the alternative is predicting bonus that
    the rules cannot award.
    """
    per_goal, per_assist = int(rules["goal"][position]), int(rules["assist"])
    goal_counts = _poisson_counts(goals, GOAL_CAP, GOAL_DISPERSION)
    assist_counts = _poisson_counts(assists, ASSIST_CAP, ASSIST_DISPERSION)

    max_involvement = GOAL_CAP + ASSIST_CAP
    archive = np.append(
        BONUS_GIVEN_INVOLVEMENT,
        np.full(max_involvement + 1 - len(BONUS_GIVEN_INVOLVEMENT), BONUS_GIVEN_MANY),
    )

    # P(involvements = i) per player, and the expected bonus that the archive's conditional
    # means would imply for this player — the denominator of the rescale.
    involvement = np.zeros((goals.size, max_involvement + 1))
    for g in range(GOAL_CAP + 1):
        involvement[:, g : g + ASSIST_CAP + 1] += goal_counts[:, g : g + 1] * assist_counts
    implied = np.clip(involvement @ archive, 1e-9, None)
    conditional = np.clip(
        archive[None, :] * (bonus / implied)[:, None], 0.0, 3.0
    )

    # Every involvement slice lands on the same support — points from returns, plus 0-3 bonus
    # — so the mixture is a straight sum of equally-shaped arrays.
    width = GOAL_CAP * per_goal + ASSIST_CAP * per_assist + 1
    combined = np.zeros((goals.size, width + 3))
    for total_involvement in range(max_involvement + 1):
        cell = np.zeros((goals.size, width))
        for g in range(min(total_involvement, GOAL_CAP) + 1):
            a = total_involvement - g
            if a > ASSIST_CAP:
                continue
            cell[:, g * per_goal + a * per_assist] += goal_counts[:, g] * assist_counts[:, a]
        combined += _convolve((0, cell), (0, _bonus_pmf(conditional[:, total_involvement])))[1]
    return 0, combined


def _defence(block: pd.DataFrame, position: str, rules: dict) -> tuple[int, np.ndarray]:
    """Clean sheet and goals conceded from one count distribution.

    The two rules are the same event seen twice — a clean sheet *is* conceding zero — so they
    share a draw. `P(0)` is pinned to the match model's clean-sheet probability, which carries
    the Dixon-Coles low-score correction that a plain Poisson would discard, and the remaining
    mass keeps the Poisson shape.
    """
    clean_sheet = int(rules["clean_sheet"].get(position, 0))
    conceded = rules["goals_conceded"]
    deducts = position in conceded["applies_to"]
    n = len(block)

    if clean_sheet == 0 and not deducts:
        return 0, np.ones((n, 1))

    p_clean = np.clip(block["p_clean_sheet"].to_numpy(float), 0.0, 1.0)
    against = np.clip(block["expected_goals_against"].to_numpy(float), 1e-6, None)

    counts = _poisson_counts(against, CONCEDED_CAP)
    remainder = np.clip(1.0 - counts[:, :1], 1e-9, None)
    counts[:, 1:] *= (1.0 - p_clean)[:, None] / remainder
    counts[:, 0] = p_clean

    values = np.array([
        clean_sheet if k == 0
        else (conceded["points"] * (k // conceded["per_goals"]) if deducts else 0)
        for k in range(CONCEDED_CAP + 1)
    ])
    return _gather(counts, values)


def _defcon(
    block: pd.DataFrame, position: str, rules: dict, minutes: float
) -> tuple[int, np.ndarray]:
    """Defensive contribution: a threshold, so a Bernoulli on the count clearing it."""
    settings = rules.get("defensive_contribution", {})
    threshold = (settings.get("thresholds") or {}).get(position)
    n = len(block)
    if not settings.get("enabled", False) or threshold is None:
        return 0, np.ones((n, 1))

    lam = block["defcon_per90"].fillna(0.0).to_numpy(float) * (minutes / MINUTES_PER_90)
    hit = poisson.sf(int(threshold) - 1, np.clip(lam, 1e-9, None))
    award = int(settings["points"])
    return _gather(
        np.column_stack([1.0 - hit, hit]), np.array([0, award])
    )


def _cards(block: pd.DataFrame, rules: dict, minutes: float) -> tuple[int, np.ndarray]:
    """One draw with three outcomes, because a red and a yellow are alternatives."""
    lam = block["yellows_per90"].fillna(0.0).to_numpy(float) * (minutes / MINUTES_PER_90)
    booked = 1.0 - np.exp(-np.clip(lam, 0.0, None))
    red = booked * RED_CARD_FRACTION
    yellow = booked - red
    return _gather(
        np.column_stack([1.0 - booked, yellow, red]),
        np.array([0, int(rules["yellow_card"]), int(rules["red_card"])]),
    )


def build_distribution(
    frame: pd.DataFrame, rules: dict | None = None, *, key: str = "player_id"
) -> PointsDistribution:
    """Points pmf for every row of a forecast frame.

    `frame` is the assembler's output — one row per player-fixture — carrying the minutes
    probabilities, allocated expected goals and assists, the match model's clean-sheet and
    conceded numbers, and the secondary rates. Rows come back in the order they went in.

    Positions are processed as groups because the scoring table, the clean-sheet value, the
    conceded deduction and the defensive-contribution threshold are all constant within one
    and vary between them. That keeps every operation vectorised across a whole position at
    once instead of looping over players.
    """
    rules = rules or load_scoring_rules()
    width = GRID_HIGH - GRID_LOW + 1
    out = np.zeros((len(frame), width))

    # Positional row numbers, not index labels: a forecast frame concatenated across
    # gameweeks routinely carries duplicate index values, and a label-based scatter would
    # write some players' distributions into other players' rows.
    positions = frame["position"].to_numpy()
    for position in pd.unique(positions):
        rows = np.flatnonzero(positions == position)
        block = frame.iloc[rows]
        probabilities = {
            BUCKET_SHORT: block["p_short"].fillna(0.0).to_numpy(float),
            BUCKET_LONG: block["p_long"].fillna(0.0).to_numpy(float),
        }
        for bucket, weight in probabilities.items():
            low, arr = _bucket_distribution(block, bucket, str(position), rules)
            out[rows] += weight[:, None] * _project(low, arr, GRID_LOW, GRID_HIGH)

        # An unused player scores exactly zero — a spike, not a tail. Taking the residual
        # rather than the model's `p_zero` keeps every row a proper distribution even if the
        # three bucket probabilities upstream do not sum to one exactly.
        zero = np.clip(1.0 - probabilities[BUCKET_SHORT] - probabilities[BUCKET_LONG], 0.0, 1.0)
        out[rows, -GRID_LOW] += zero

    totals = out.sum(axis=1, keepdims=True)
    out = np.divide(out, totals, out=np.zeros_like(out), where=totals > 0)
    ids = frame[key].to_numpy() if key in frame.columns else None
    return PointsDistribution(out, GRID_LOW, ids)


def combine_fixtures(
    distribution: PointsDistribution, keys: pd.DataFrame
) -> tuple[pd.DataFrame, PointsDistribution]:
    """Collapse each key group's fixtures into one distribution by convolution.

    A double gameweek is the sum of two independent fixtures, so the pmfs convolve. Adding
    means and variances — the only thing a summary statistic allows — gets the centre right
    and the shape wrong, and a double gameweek is exactly when the shape is being asked
    about: Bench Boost and Triple Captain are timed on the ceiling, not the mean.

    `keys` identifies the groups, one row per row of `distribution`, typically
    season/gameweek/player. The returned keys frame is the distinct groups in first-seen
    order, aligned with the returned distribution.
    """
    if len(keys) != len(distribution):
        raise ValueError("keys and distribution must have the same number of rows")

    grouped = keys.reset_index(drop=True).groupby(list(keys.columns), sort=False)
    order = list(grouped.indices.items())
    width = distribution.pmf.shape[1]
    high = distribution.low + width - 1

    out = np.zeros((len(order), width))
    multiple = 0
    for row, (_, rows) in enumerate(order):
        if len(rows) == 1:
            out[row] = distribution.pmf[rows[0]]
            continue
        multiple += 1
        low, arr = distribution.low, distribution.pmf[rows[0]][None, :]
        for index in rows[1:]:
            low, arr = _convolve(
                (low, arr), (distribution.low, distribution.pmf[index][None, :])
            )
        out[row] = _project(low, arr, distribution.low, high)[0]

    if multiple:
        log.info("%d player-gameweeks had multiple fixtures and were convolved", multiple)

    combined_keys = pd.DataFrame(
        [key if isinstance(key, tuple) else (key,) for key, _ in order],
        columns=list(keys.columns),
    )
    ids = (
        combined_keys["player_id"].to_numpy() if "player_id" in combined_keys.columns else None
    )
    return combined_keys, PointsDistribution(out, distribution.low, ids)


def gameweek_distributions(
    fixture_level: pd.DataFrame,
    rules: dict | None = None,
    *,
    keys: tuple[str, ...] = ("season", "gw", "player_id"),
) -> tuple[pd.DataFrame, PointsDistribution]:
    """Per player-gameweek distributions from per-fixture forecasts.

    The convenience wrapper the backtest actually uses: build at fixture level, where every
    component model is defined, then convolve up to the gameweek, which is the unit decisions
    are made in.
    """
    present = [k for k in keys if k in fixture_level.columns]
    distribution = build_distribution(fixture_level, rules)
    return combine_fixtures(distribution, fixture_level[present])
