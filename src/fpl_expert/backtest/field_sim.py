"""Simulate the field, so rank can be measured instead of only total points.

Everything else in this project scores a season by points. That is the wrong objective for a
top-10k target, because rank depends on `X - Y` — your score minus the field's — and the
template largely cancels out of that difference. A player owned by 80% of managers barely
moves your rank when he hauls; he ruins it when you do not own him.

`lambda_rank` has sat at 0 since it was built for exactly one reason: nothing here could
measure whether raising it helped. This closes that.

**How the field is built.** The archive records `selected` — how many managers actually owned
each player in each gameweek. Rival squads are drawn from that distribution, so the simulated
field owns what the real field owned, including the concentration that makes differentials
hard: in GW10 of 2025-26, 8.6M managers held Haaland.

Sampling uses the Gumbel top-k trick — adding Gumbel noise to log-weights and taking the top
k is exactly equivalent to weighted sampling WITHOUT replacement, and it vectorises across
tens of thousands of managers at once. Sampling with replacement would be far simpler and
wrong: it would let a manager own the same player twice, which understates how concentrated
real squads are and therefore makes differentials look cheaper than they are.

**Budget and the club limit — built, measured, and OFF by default.** Drawn squads are not
legal as often as one might hope: 47% of GW1 squads come out over £100m and 20% hold four or
more from one club, because the mean drawn squad costs £99.5m and half a distribution centred
on the limit sits above it. `repair_squads` fixes that almost completely (0.7% and 0.3%
remaining) but costs more than it buys:

    field            GW1 cost   illegal          median   top 10k   ownership anchor
    as drawn          £99.5m    47% / 20%         2076     2504      1985 (target 1991)
    repaired          £95.4m    0.7% / 0.3%       2054     2444      1964

Repairing trades away the exact calibration this whole model is anchored on. The reason is
structural: ownership marginals are reproduced correctly but the JOINT is not — real managers
who own one premium cannot own three, and independent sampling does not know that. Repairing
afterwards is not the same as sampling from the right joint, and it over-corrects by dropping
precisely the expensive, high-scoring players.

So the field keeps illegal squads and its exact ownership calibration, rather than legal
squads and a 1.4% miss. `enforce_legality=True` switches it on for questions about squad
structure, where legality matters more than the anchor.

**Rival managers play chips.** Each plays one Bench Boost and one Triple Captain per
half-season, timed with probability proportional to how many fixtures a gameweek carries, so
they aim at double gameweeks the way real managers do. Deliberately NOT timed with our
forecasts: chip timing is a genuine skill and handing the whole field ours would make every
rival as good at it as the system being measured. Worth +28 points at the median and +43 at
the top-10k threshold — a field that played none while we played three was flattering our
rank by about that much.

## An exact anchor to calibrate against

Most simulation work has nothing to check itself against. This one does, and it falls out of
an identity rather than an assumption. Every manager owns exactly fifteen players, so summing
ownership counts over all players counts each manager fifteen times:

    managers        = sum(selected) / 15
    mean squad pts  = sum(selected * points) / managers

The first recovers 10.77M managers for 2024-25, which is the real figure.

**But neither is exact, and an independent review caught it.** Both hold only when every
player a manager holds appears in the frame. In a BLANK gameweek players without a fixture
have no row, so the sum loses their ownership, the implied manager count comes out low, and
the anchor comes out high — by 1.5-2.1% per season:

    season    naive anchor   blank-corrected
    2023-24      2014            1974
    2024-25      1991            1956
    2025-26      1958            1928

`manager_count` applies the correction, using the fact that the real count is monotone
non-decreasing within a season. Any field model can then be checked against it: draw its
squads and see whether they hold the same points the real field held — a test of composition
alone, independent of who each manager started or captained.

A second correction, this one in the model's favour. The simulator can only draw players that
are IN the frame, so the fair target is the achievable anchor given that frame, not the whole
league's. Measured against it the sampler lands within 0.04% on 2024-25 rather than the 0.3%
previously claimed. The residual gap was missing players, not the sampler.

## How squads are drawn, and why the obvious method fails

The first version drew squads with the Gumbel top-k trick — weighted sampling without
replacement, which is exactly Plackett-Luce. It is correct as sampling and wrong for this
purpose, because **Plackett-Luce does not reproduce the ownership marginals it is given**.
For a fixed sample size the inclusion probability of a heavily-weighted item saturates below
its weight share, and in FPL the heavily-owned players are precisely the high scorers. Drawn
that way the field held 1928 points against the true 1991, and it lost them all at the top of
the ownership distribution.

Pareto order sampling fixes it by targeting inclusion probabilities directly. Give each
player a key `(U/(1-U)) / (pi/(1-pi))` and take the k smallest; the resulting inclusion
probabilities match `pi` closely, and the sample size is exactly k. The target `pi` is the
real ownership fraction, and it is feasible without adjustment: ownership fractions within a
position sum to exactly that position's quota (2.000, 5.000, 5.000, 3.000 — measured, not
imposed), because every manager owns exactly that many.

## Persistence and transfers, from one construction

A manager needs to be the same manager in GW38 as in GW1 — persistence is what creates the
tail, and redrawing weekly crushes the distribution toward the mean. But a frozen squad is
just as wrong in the other direction: it accumulates zeros from players who are injured,
dropped or sold, while real managers make 30-40 transfers a season largely to avoid that.

Both come from one idea. Each manager holds a FIXED uniform `U` per player for the whole
season, while the ownership targets `pi` move week to week. The squad is then whatever those
fixed uniforms select against this week's marginals: it tracks the crowd's migration toward
in-form players, yet stays the same squad from one week to the next. Transfers are not
scripted — they emerge, at a rate the ownership data itself dictates.

Measured against 2024-25, with fixed uniforms versus fresh ones each week:

    fresh uniforms   mean 1983   sd  79   p99 2163      (marginals right, tail destroyed)
    fixed uniforms   mean 1984   sd 187   p99 2391      (marginals right, tail intact)

Same mean, and the mean is the anchor's; more than twice the spread.

## What the finished model reproduces

Squad composition against the exact anchor, and the resulting spread of manager totals:

    season     squad-15 (anchor)     median    p99    top 10k    best of 20,000
    2023-24     2014  (2014)          2039     2418     2494          2547
    2024-25     1985  (1991)          2053     2425     2500          2571
    2025-26     1958  (1958)          1919     2223     2281          2370

Composition is within 0.3% everywhere and exact in two of the three seasons. It is not fitted:
the only tuned quantity in the model is the skill spread, which barely moves the median. The
2025-26 field is genuinely weaker rather than mis-modelled — that season's mean squad is 56
points lower before any modelling starts.

## Read saturation as a warning, not a result

With 20,000 rivals, one of them stands for 550 real managers, so that is the finest rank this
resolves. Our own squads land at:

    2023-24   2617 pts   beats the entire field      (better than ~550, unresolved)
    2024-25   2467 pts   rank ~35,200
    2025-26   2359 pts   rank ~550                   (one simulated manager ahead)

Two of three seasons sit at or above the top of a field that now reproduces real ownership
exactly and reaches a realistic top-10k threshold. The likelier explanation is that the
BACKTEST flatters us, not that these squads would have finished in the top few hundred. It
takes selling price to be market price, applies no availability gate because the archive has
no injury data, and chooses the XI with the same forecast it is scored on. A saturated rank
should send you to those assumptions rather than into the record as an achievement.

## History

Before 2026-08-10 this file drew eleven players once from season-average ownership and held
them for 38 gameweeks. Its field had a median of 1640 against the true 1991, and its best
manager in twenty thousand would not have made the real top 10k. Our own squads beat 100% of
it, so every rank it produced was an extrapolation off the top of a distribution that was too
low — including "mean simulated rank 7,089", which is withdrawn.

That weakness had been masked by the double-gameweek duplication bug (see
`backtest/historical_forecast.py`), which inflated realised points for the field as much as
for us and propped the distribution up far enough to look plausible.

Three separate errors had to be fixed to get from that model to this one, and each was worth
more than the last: the sampler did not reproduce the marginals it was given, squads were
frozen for the season, and every manager was equally skilled. Only the third needed a fitted
parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_FIELD_SIZE = 20_000

# The squad every FPL manager holds. These are also the per-position sums of ownership
# fractions, which is what makes them a feasible set of Pareto inclusion targets.
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = 15

# Starting XI: the minimum each position must field, and the most it may.
FORMATION_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
FORMATION_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
STARTING_XI = 11

# Spread of manager skill. Set to 0 for a homogeneous field, where every rival is equally
# good and the only thing separating them is luck in the draw.
#
# CALIBRATED, NOT DERIVED — and it is the one number here that does not come from the
# archive. Squad composition is pinned exactly by the ownership identity; the *tail* is not,
# because nothing in the data records how individual managers finished.
#
# Fitted on 2024-25 so the simulated top-10k threshold matches FPL's published one of roughly
# 2500 points, with the median left free as a check. Re-fitted once the field started playing
# chips, which supply part of the spread skill previously had to supply alone:
#
#     dispersion   median   top 10k   best of 40,000
#         0.15      2070      2461         2541
#         0.20      2072      2482         2570
#         0.25      2076      2505         2571      <- chosen
#         0.30      2078      2531         2590
#         0.35      2082      2545         2626      <- was chosen, before field chips
#
# The median barely moves across that range, which is the point: skill spreads the field
# without shifting its centre, so the ownership anchor survives the calibration instead of
# being traded against it.
#
# Any rank quoted from this field inherits the 2500 assumption and should be read as "against
# a field calibrated to reach 2500 at 10k". Set to 0 for a homogeneous field.
SKILL_DISPERSION = 0.25


def pareto_sample(
    inclusion: np.ndarray, k: int, uniforms: np.ndarray
) -> np.ndarray:
    """Fixed-size sample of `k` whose inclusion probabilities match `inclusion`.

    Pareto order sampling: rank players by `(U/(1-U)) / (pi/(1-pi))` and keep the k smallest.
    Unlike Plackett-Luce it hits the marginals it is given, which matters here because the
    marginals ARE the data — `pi` is the ownership the real field actually had, and a sampler
    that compresses the top of it systematically drops the best players.

    `uniforms` is `(n_managers, n_players)`. Holding it fixed across gameweeks while `pi`
    moves is what makes a manager persist as the same manager while still transferring.

    `inclusion` may be one row shared by everyone, or one row per manager when skill makes
    them want different squads; both broadcast.
    """
    odds_target = inclusion / np.clip(1.0 - inclusion, 1e-9, None)
    odds_uniform = uniforms / np.clip(1.0 - uniforms, 1e-9, None)
    keys = odds_uniform / np.clip(odds_target, 1e-12, None)
    return np.argpartition(keys, kth=k - 1, axis=1)[:, :k]


def skill_tilt(quality: np.ndarray, skill: np.ndarray) -> np.ndarray:
    """Per-manager multiplier on ownership, so that some managers pick better than others.

    Without this every simulated manager is equally good and the field has no top end — the
    only thing separating them is luck in the draw. Real top-10k managers are not lucky
    average managers, and a field of average managers cannot be ranked against.

    A manager with skill `s` tilts toward players with a high quality score `z` by
    `exp(s * z)`. Keeping the population average on the real ownership is NOT done here — see
    `balanced_inclusion`, which enforces it directly rather than by an analytic correction.
    """
    return np.exp(skill[:, None] * quality[None, :])


def balanced_inclusion(
    inclusion: np.ndarray, tilt: np.ndarray, quota: int, iterations: int = 3
) -> np.ndarray:
    """Tilted inclusion probabilities that keep BOTH constraints that matter.

    Two things have to hold at once and they pull against each other:

    * every manager owns exactly `quota` players in the position (rows sum to the quota);
    * the field as a whole owns what the real field owned (columns average to `inclusion`).

    Enforcing them in sequence does not work. An analytic Jensen correction makes the average
    *tilt* one, but the row normalisation that follows — needed to keep squads legal — is
    itself correlated with the tilt and puts the bias straight back. Measured, that cost 12%
    of the field's points at a skill dispersion of 0.4, breaking the calibration the whole
    model is anchored on.

    Alternating the two normalisations (Sinkhorn) converges to a matrix satisfying both. The
    loop ends on the row condition, so legality is exact and the ownership match is very close
    rather than the other way round — a squad of the wrong size is nonsense, whereas ownership
    off in the fourth decimal is not.

    Three iterations is not a guess. Measured on a realistic 20,000 x 250 block, the largest
    ownership error falls 2.3e-4 -> 1e-5 -> below 1e-6 over the first three passes and stops
    improving; row sums are exact at every one.
    """
    scaled = inclusion[None, :] * tilt
    for _ in range(iterations):
        scaled = _fill_to_quota(scaled, quota)
        scaled *= inclusion / np.clip(scaled.mean(axis=0), 1e-12, None)
    return _fill_to_quota(scaled, quota)


def _fill_to_quota(weights: np.ndarray, quota: int, cap: float = 0.999) -> np.ndarray:
    """Scale each row to sum to `quota` with no entry above `cap`.

    Water-filling rather than scale-then-clip. A strongly skilled manager's tilt can push a
    premium player's inclusion above 1 — he wants Salah more than certainly — and simply
    clipping that back loses the excess, leaving the squad short. Measured, scale-then-clip
    left rows summing to 1.16 instead of 2. The excess has to go somewhere, and where it
    belongs is the rest of that manager's squad: cap whoever is over, then redistribute what
    is left over the players still free to move, and repeat until nothing is over.
    """
    filled = np.clip(weights, 0.0, None)
    for _ in range(8):
        filled = filled * (
            quota / np.clip(filled.sum(axis=1, keepdims=True), 1e-12, None)
        )
        over = filled > cap
        if not over.any():
            break
        capped_mass = np.where(over, cap, 0.0).sum(axis=1, keepdims=True)
        free = np.where(over, 0.0, filled)
        free_total = np.clip(free.sum(axis=1, keepdims=True), 1e-12, None)
        remaining = np.clip(quota - capped_mass, 0.0, None)
        filled = np.where(over, cap, free * (remaining / free_total))
    return np.clip(filled, 0.0, cap)


def _quality_score(values: np.ndarray) -> np.ndarray:
    """Standardised player quality within a position, as a skilled manager would rank it.

    Our own expected points are the only available stand-in for what a good manager sees.
    That makes skilled rivals good at the same things we are, which understates how a real
    strong manager might beat us in ways our model cannot see — the resulting rank is
    therefore optimistic, and is the right direction to be wrong about a competitor.
    """
    spread = values.std()
    if spread <= 0:
        return np.zeros_like(values)
    return np.clip((values - values.mean()) / spread, -3.0, 3.0)


def manager_count(frame: pd.DataFrame, ownership_col: str = "selected") -> pd.Series:
    """Managers per gameweek, corrected for players with no fixture.

    `sum(selected)/15` is exact only when every player a manager holds appears in the frame.
    In a BLANK gameweek — 2 to 5 per season — players without a fixture have no row at all, so
    the sum loses their ownership and the implied manager count comes out too low. Everything
    divided by it, including the anchor this whole field model is calibrated against, is then
    too high: measured at 1.5-2.1% per season, and up to 2.3x in the worst individual week.

    The real count is monotone non-decreasing within a season — FPL managers do not un-enter —
    so a running maximum of the naive estimate is a valid floor, and a far better one than the
    naive figure in exactly the weeks the naive figure breaks.
    """
    naive = frame.groupby("gw")[ownership_col].sum() / SQUAD_SIZE
    return naive.sort_index().cummax()


def _inclusion_targets(counts: np.ndarray, quota: int) -> np.ndarray:
    """Ownership counts to inclusion probabilities summing to the position's quota.

    The scale factor is `1 / n_managers`, but rather than take that from elsewhere it is
    derived from the constraint itself: within a position the fractions must sum to the quota,
    because every manager owns exactly that many. Normalising rather than dividing also makes
    this robust to a forecast frame that omits a few fringe players, which would otherwise
    understate the manager count and inflate every share.
    """
    total = counts.sum()
    if total <= 0:
        return np.full(len(counts), quota / max(len(counts), 1))
    # Clipped below 1: a player cannot be owned by more than every manager, and an inclusion
    # probability of exactly 1 makes the Pareto odds infinite.
    return np.clip(counts / total * quota, 0.0, 0.999)


def _pick_starting_xi(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Which of each position's squad members start, given a value per player.

    Returns a boolean mask per position. The rule is the legal one and needs no search: field
    every position's minimum (1 GK, 3 DEF, 2 MID, 1 FWD = 7), then fill the remaining four
    places with the best players left over. For outfielders the greedy fill is exactly optimal
    rather than an approximation, because 5 DEF, 5 MID and 3 FWD are simultaneously the squad
    quotas and the formation maxima, so no combination of four can breach one.

    GOALKEEPERS ARE THE EXCEPTION and must be excluded from that pool. A squad holds two and a
    team may field exactly one, so the reserve keeper is the single player in the fifteen who
    can never be promoted. Leaving him in the pool fields twelve-man teams with two keepers.
    """
    starters, spare_values, spare_index = {}, [], []
    for position, held in values.items():
        # Descending by value, so the first columns are the ones a manager would field.
        ranking = np.argsort(-held, axis=1)
        minimum = FORMATION_MIN[position]
        starters[position] = np.zeros(held.shape, dtype=bool)
        np.put_along_axis(starters[position], ranking[:, :minimum], True, axis=1)

        if FORMATION_MAX[position] <= minimum:
            continue        # no discretion here — the reserve keeper cannot start

        # Everyone below the minimum, up to the position's maximum, competes for the
        # remaining places.
        rest = ranking[:, minimum : FORMATION_MAX[position]]
        if rest.shape[1]:
            spare_values.append(np.take_along_axis(held, rest, axis=1))
            spare_index.append((position, rest))

    if not spare_index:
        return starters

    pooled = np.concatenate(spare_values, axis=1)
    remaining = STARTING_XI - sum(FORMATION_MIN.values())
    chosen = np.argpartition(-pooled, kth=remaining - 1, axis=1)[:, :remaining]

    # Map the pooled winners back to the position they came from.
    offset = 0
    rows = np.arange(pooled.shape[0])[:, None]
    for position, rest in spare_index:
        width = rest.shape[1]
        local = chosen - offset
        hit = (local >= 0) & (local < width)
        # Out-of-range entries are clamped so the scatter stays in bounds, then masked off.
        columns = np.take_along_axis(rest, np.clip(local, 0, width - 1), axis=1)
        np.logical_or.at(starters[position], (rows, columns), hit)
        offset += width
    return starters


@dataclass
class FieldTrace:
    """Who every simulated manager held and started, gameweek by gameweek.

    Separated from scoring for the same reason our own season is: **none of it depends on
    realised points.** Squads come from ownership, the XI from forecasts, the armband from
    ownership again — so a field can be built once and then scored against as many different
    outcome draws as wanted. That is what keeps repeated simulation affordable now that
    building a field is the expensive part.

    Indices are positions in `player_ids`, not FPL ids, so scoring is a pure gather.
    """

    player_ids: np.ndarray          # (n_players,) the id each column index refers to
    gameweeks: list[int]
    squad: np.ndarray               # (n_gw, n_managers, 15) indices into player_ids
    starting: np.ndarray            # (n_gw, n_managers, 15) whether each is in the XI
    captain: np.ndarray             # (n_gw, n_managers) index into player_ids
    bench_boost: np.ndarray | None = None      # (n_gw, n_managers) played this week?
    triple_captain: np.ndarray | None = None   # (n_gw, n_managers) played this week?

    @property
    def n_managers(self) -> int:
        return self.squad.shape[1]

    def score_gameweek(self, week_points: np.ndarray, step: int) -> np.ndarray:
        """Every manager's score in one gameweek, from a `(n_players,)` points vector.

        The unit a weekly decision is judged in. Kept separate from `score` so that a single
        gameweek can be evaluated against a resampled draw without rebuilding a season.
        """
        held = week_points[self.squad[step]]
        return (held * self.starting[step]).sum(axis=1) + week_points[self.captain[step]]

    def score(self, points: np.ndarray, *, squad_only: bool = False) -> np.ndarray:
        """Season totals against a `(n_gw, n_players)` matrix of realised points.

        `squad_only` sums all fifteen and skips the armband — the quantity the ownership
        identity pins exactly, and therefore the calibration test.
        """
        totals = np.zeros(self.n_managers)
        for index in range(len(self.gameweeks)):
            week = points[index]
            held = week[self.squad[index]]
            if squad_only:
                totals += held.sum(axis=1)
                continue

            # Bench Boost scores the whole fifteen rather than the eleven.
            fielded = self.starting[index]
            if self.bench_boost is not None:
                fielded = fielded | self.bench_boost[index][:, None]
            totals += (held * fielded).sum(axis=1)

            armband = week[self.captain[index]]
            if self.triple_captain is not None:
                # The armband already grants a second copy; Triple Captain adds a third.
                armband = armband * np.where(self.triple_captain[index], 2, 1)
            totals += armband
        return totals


def points_matrix(
    frame: pd.DataFrame, player_ids: np.ndarray, gameweeks: list[int], points_col: str
) -> np.ndarray:
    """`(n_gw, n_players)` realised points, aligned to a trace's indexing."""
    lookup = pd.Series(np.arange(len(player_ids)), index=player_ids)
    grouped = frame.groupby(["gw", "player_id"])[points_col].sum()
    out = np.zeros((len(gameweeks), len(player_ids)))
    gw_position = {gw: i for i, gw in enumerate(gameweeks)}
    for (gw, player), value in grouped.items():
        if gw in gw_position and player in lookup.index:
            out[gw_position[gw], lookup[player]] = value
    return out


def repair_squads(
    picks: np.ndarray,
    column_position: np.ndarray,
    player_position: np.ndarray,
    ownership: np.ndarray,
    price: np.ndarray,
    club: np.ndarray,
    *,
    budget: float,
    max_per_club: int,
    rounds: int = 3,
) -> np.ndarray:
    """Swap out picks that break the budget or the club limit, keeping the squad shape.

    Ownership sampling produces squads that are close to legal without being told to be, but
    not close enough to ignore: measured on 2024-25, 48% of GW1 squads came out over £100m and
    20% held four or more from one club, because the mean drawn squad costs £99.7m and half of
    a distribution centred on the limit sits above it. A field made of illegal squads is
    systematically stronger than the one being ranked against.

    This is a repair rather than a constrained solve. Replacements are drawn from the SAME
    COLUMN's position, so the 2/5/5/3 shape survives, and are taken in the field's own
    preference order — most-owned first among those that fit. A squad still broken after a few
    rounds is left as it is rather than mangled; putting an optimiser inside a field simulator
    would cost more than the error it removes.

    The pick dropped is the least-owned of the offending group, which is the one a real
    manager is least attached to.
    """
    picks = picks.copy()
    # Candidate lists per position, most-owned first: the order the field itself prefers.
    preference = {}
    for position in np.unique(column_position):
        members = np.flatnonzero(player_position == position)
        preference[position] = members[np.argsort(-ownership[members])]

    for _ in range(rounds):
        held_price = price[picks].sum(axis=1)
        held_club = club[picks]
        counts = np.apply_along_axis(
            np.bincount, 1, held_club, None, int(club.max()) + 1
        )
        over_club = counts.max(axis=1) > max_per_club
        broken = np.flatnonzero((held_price > budget) | over_club)
        if broken.size == 0:
            break

        for row in broken:
            held = picks[row]
            if over_club[row]:
                worst = int(counts[row].argmax())
                offending = np.flatnonzero(club[held] == worst)
            else:
                # Over budget: trade down from the most expensive pick held.
                offending = np.array([int(np.argmax(price[held]))])
            column = int(offending[np.argmin(ownership[held[offending]])])
            drop = held[column]

            owned = set(held.tolist())
            headroom = budget - (held_price[row] - price[drop])

            # Two candidates are tracked, because one swap is often not enough. A squad £5m
            # over budget cannot be fixed by replacing its most expensive player if the rest
            # already exceed the budget — headroom comes out NEGATIVE and nothing fits. Giving
            # up there is what made the repair stall at 16% still-illegal however many rounds
            # it was given. Falling back to the cheapest legal alternative guarantees the
            # squad gets cheaper every round, so successive rounds converge.
            affordable = cheapest = None
            for candidate in preference[column_position[column]]:
                if candidate in owned:
                    continue
                same_club = int((club[held] == club[candidate]).sum())
                same_club -= int(club[drop] == club[candidate])
                if same_club >= max_per_club:
                    continue
                if price[candidate] <= headroom:
                    affordable = candidate      # most-owned that fits: the field's choice
                    break
                if cheapest is None or price[candidate] < price[cheapest]:
                    cheapest = candidate

            chosen = affordable
            if chosen is None and cheapest is not None and price[cheapest] < price[drop]:
                chosen = cheapest
            if chosen is not None:
                picks[row, column] = chosen

    return picks


def _enforce_legality(held_global, held_values, held_ownership, week, legality, global_index):
    """Repair one gameweek's squads, then put the per-position views back in step.

    The repair works on the whole fifteen because budget and the club limit are squad-wide,
    but everything downstream — the XI, the armband — reads per-position blocks. So the
    blocks are concatenated, repaired, and split again, with values and ownership re-read for
    whoever came in.
    """
    order = [p for p in SQUAD_QUOTA if p in held_global]
    widths = [held_global[p].shape[1] for p in order]
    combined = np.concatenate([held_global[p] for p in order], axis=1)
    column_position = np.concatenate(
        [np.full(w, p) for p, w in zip(order, widths, strict=True)]
    )

    repaired = repair_squads(
        combined, column_position, legality["position"], legality["ownership"],
        legality["price"], legality["club"],
        budget=legality["budget"], max_per_club=legality["max_per_club"],
    )

    # Values and ownership for whoever is now held, looked up by global index.
    values = week.groupby("player_id")[legality["value_col"]].sum().reindex(
        global_index.index
    ).fillna(0.0).to_numpy()
    owned = week.groupby("player_id")[legality["ownership_col"]].max().reindex(
        global_index.index
    ).fillna(0.0).to_numpy()

    out_global, out_values, out_owned = {}, {}, {}
    start = 0
    for position, width in zip(order, widths, strict=True):
        block = repaired[:, start : start + width]
        out_global[position] = block
        out_values[position] = values[block]
        out_owned[position] = owned[block]
        start += width
    return out_global, out_values, out_owned


def _chip_weeks(
    fixture_counts: np.ndarray,
    gameweeks: list[int],
    n_managers: int,
    rng: np.random.Generator,
    windows: tuple[tuple[int, int], ...] = ((1, 19), (20, 38)),
) -> np.ndarray:
    """When each manager plays one chip, one use per half-season window.

    Real managers do not scatter chips uniformly — they aim them at double gameweeks, which is
    when a bench is worth fielding and a captain worth tripling. Timing is therefore drawn
    with probability proportional to how many fixtures that gameweek carries, which makes
    doubles roughly twice as likely as singles without needing chip-usage data the archive
    does not record.

    They are NOT timed with our forecasts. Chip timing is a genuine skill and handing the
    whole field ours would make every rival as good at it as the system being measured.
    """
    played = np.zeros((len(gameweeks), n_managers), dtype=bool)
    for start, stop in windows:
        eligible = [i for i, gw in enumerate(gameweeks) if start <= gw <= stop]
        if not eligible:
            continue
        weights = fixture_counts[eligible].astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else None
        chosen = rng.choice(eligible, size=n_managers, p=weights)
        played[chosen, np.arange(n_managers)] = True
    return played


def build_field(
    forecasts: pd.DataFrame,
    n_managers: int = DEFAULT_FIELD_SIZE,
    *,
    seed: int = 0,
    ownership_col: str = "selected",
    selection_col: str = "expected_points",
    skill_dispersion: float = SKILL_DISPERSION,
    play_chips: bool = True,
    enforce_legality: bool = False,
    budget: float = 100.0,
    max_per_club: int = 3,
) -> FieldTrace:
    """Draw a field: who holds whom, week by week, with no reference to outcomes."""
    rng = np.random.default_rng(seed)
    frame = forecasts.copy()
    frame[ownership_col] = frame[ownership_col].fillna(0.0)
    if selection_col not in frame.columns:
        selection_col = ownership_col

    # Fixed for the season, like the uniforms: a manager who is good in August is still good
    # in May. Drawing skill weekly would average it away and leave the field homogeneous.
    skill = rng.normal(0.0, skill_dispersion, n_managers) if skill_dispersion > 0 else None

    gameweeks = sorted(frame["gw"].unique())
    player_ids = frame["player_id"].unique()
    global_index = pd.Series(np.arange(len(player_ids)), index=player_ids)

    # One uniform per manager per player, drawn once and reused all season. float32 halves
    # the memory for what is otherwise the largest array here.
    uniforms: dict[str, np.ndarray] = {}
    pools: dict[str, pd.Index] = {}
    for position in SQUAD_QUOTA:
        players = pd.Index(frame.loc[frame["position"] == position, "player_id"].unique())
        if players.empty:
            continue
        pools[position] = players
        uniforms[position] = rng.random(
            (n_managers, len(players)), dtype=np.float32
        ).astype(np.float64)

    # Everything the legality repair needs, indexed the same way as `player_ids`.
    legality = None
    if enforce_legality:
        info = frame.drop_duplicates("player_id").set_index("player_id").reindex(player_ids)
        if "price" in info.columns and "team" in info.columns:
            legality = {
                "position": info["position"].to_numpy(),
                "price": info["price"].fillna(4.0).to_numpy(float),
                "club": pd.factorize(info["team"].fillna("?"))[0],
                "ownership": frame.groupby("player_id")[ownership_col].max()
                .reindex(player_ids).fillna(0.0).to_numpy(),
                "budget": budget,
                "max_per_club": max_per_club,
                "value_col": selection_col,
                "ownership_col": ownership_col,
            }

    width = sum(min(q, len(pools[p])) for p, q in SQUAD_QUOTA.items() if p in pools)
    squad = np.zeros((len(gameweeks), n_managers, width), dtype=np.int32)
    starting = np.zeros((len(gameweeks), n_managers, width), dtype=bool)
    captain = np.zeros((len(gameweeks), n_managers), dtype=np.int32)

    for step, gw in enumerate(gameweeks):
        week = frame[frame["gw"] == gw]
        held_values, held_global, held_ownership = {}, {}, {}

        for position, quota in SQUAD_QUOTA.items():
            if position not in pools:
                continue
            players = pools[position]
            block = week[week["position"] == position].groupby("player_id").agg(
                own=(ownership_col, "max"), value=(selection_col, "sum"),
            ).reindex(players).fillna(0.0)

            k = min(quota, len(players))
            inclusion = _inclusion_targets(block["own"].to_numpy(float), k)
            if skill is not None:
                quality = _quality_score(block["value"].to_numpy(float))
                inclusion = balanced_inclusion(inclusion, skill_tilt(quality, skill), k)
            picks = pareto_sample(inclusion, k, uniforms[position])

            held_values[position] = block["value"].to_numpy(float)[picks]
            held_ownership[position] = block["own"].to_numpy(float)[picks]
            held_global[position] = global_index.reindex(players).to_numpy()[picks]

        if not held_global:
            continue

        if enforce_legality and legality is not None:
            held_global, held_values, held_ownership = _enforce_legality(
                held_global, held_values, held_ownership, week, legality, global_index
            )

        starters = _pick_starting_xi(held_values)
        offset = 0
        best_own = np.full(n_managers, -1.0)
        rows = np.arange(n_managers)
        for position in SQUAD_QUOTA:
            if position not in held_global:
                continue
            block_width = held_global[position].shape[1]
            slot = slice(offset, offset + block_width)
            squad[step, :, slot] = held_global[position]
            starting[step, :, slot] = starters[position]
            offset += block_width

            # The armband goes on the most-owned player they are actually starting.
            owned = np.where(starters[position], held_ownership[position], -1.0)
            index = owned.argmax(axis=1)
            improved = owned[rows, index] > best_own
            best_own = np.where(improved, owned[rows, index], best_own)
            captain[step] = np.where(
                improved, held_global[position][rows, index], captain[step]
            )

    if not play_chips:
        return FieldTrace(player_ids, gameweeks, squad, starting, captain)

    # How many fixtures each gameweek carries, as a proxy for where the crowd aims its chips.
    counts = (
        frame.groupby("gw")["fixture"].nunique().reindex(gameweeks).fillna(1).to_numpy()
        if "fixture" in frame.columns
        else frame.groupby("gw")["player_id"].nunique().reindex(gameweeks).fillna(1).to_numpy()
    )
    return FieldTrace(
        player_ids, gameweeks, squad, starting, captain,
        bench_boost=_chip_weeks(counts, gameweeks, n_managers, rng),
        triple_captain=_chip_weeks(counts, gameweeks, n_managers, rng),
    )


def simulate_field_season(
    forecasts: pd.DataFrame,
    n_managers: int = DEFAULT_FIELD_SIZE,
    *,
    seed: int = 0,
    ownership_col: str = "selected",
    points_col: str = "actual_points",
    selection_col: str = "expected_points",
    skill_dispersion: float = SKILL_DISPERSION,
    play_chips: bool = True,
    return_squad_points: bool = False,
) -> np.ndarray:
    """Season totals for a simulated field of persistent, transferring managers.

    Each manager holds fifteen players, re-selected every gameweek against that week's real
    ownership by a fixed set of uniforms. Fixing the uniforms is what makes them persist as a
    manager; letting the ownership move is what makes them transfer. Neither is scripted —
    see the module docstring for why one construction gives both.

    The starting XI is chosen on `selection_col` (our own expected points, where present),
    not on ownership. This is deliberate and it is the one place the field borrows from us.
    Ownership moves far too slowly to notice that a player is injured this week, so choosing
    on it forces simulated managers to field players who are not playing — the single largest
    reason the old model scored so far below the real field. Deciding which of your fifteen
    to start is an easy task that real managers do well, so giving them a competent XI is more
    faithful than giving them a blind one. Squad COMPOSITION, which is the hard part and the
    thing being modelled, still comes entirely from real ownership.

    Captaincy stays on ownership: the field really does pile onto the most-owned premium, and
    that behaviour is exactly what makes a differential armband the main lever on rank.

    Convenience wrapper over `build_field` and `FieldTrace.score`. Anything scoring the same
    field against more than one set of outcomes should call those directly and keep the trace,
    since drawing the field is the expensive half by three orders of magnitude.
    """
    trace = build_field(
        forecasts, n_managers, seed=seed, ownership_col=ownership_col,
        selection_col=selection_col, skill_dispersion=skill_dispersion,
        play_chips=play_chips,
    )
    points = points_matrix(
        forecasts.assign(**{points_col: forecasts[points_col].fillna(0.0)}),
        trace.player_ids, trace.gameweeks, points_col,
    )
    return trace.score(points, squad_only=return_squad_points)


def rank_metrics(our_total: float, field_totals: np.ndarray) -> dict:
    """Where our season lands against the field.

    `percentile` is the share of the field we beat, so higher is better; `rank_in_11m`
    translates it to a position among a real-sized FPL field for interpretability.

    `saturated` is true when we beat every simulated manager. Extrapolating a rank from that
    is not a measurement — it says only that our score is off the top of a distribution whose
    top is known to be too low, and the extrapolated figure gets better the more inadequate
    the field is. Callers must check it before quoting a rank.
    """
    beaten = float((field_totals < our_total).mean())
    saturated = beaten >= 1.0
    # Beating every sampled manager does not mean first place; it means the answer is finer
    # than this field can resolve. With 20,000 simulated rivals one manager stands for 550 of
    # the real 11M, so that is the floor.
    resolution = round(11_000_000 / max(len(field_totals), 1))
    return {
        "our_points": round(our_total, 1),
        "field_mean": round(float(field_totals.mean()), 1),
        "field_p99": round(float(np.percentile(field_totals, 99)), 1),
        "field_p999": round(float(np.percentile(field_totals, 99.9)), 1),
        "field_max": round(float(field_totals.max()), 1),
        "percentile": round(beaten, 5),
        "top_pct": round((1 - beaten) * 100, 3),
        "rank_in_11m": None if saturated else round((1 - beaten) * 11_000_000),
        "saturated": saturated,
        "resolution": resolution,
        # What can still be said when saturated: better than this, but not by how much.
        "rank_better_than": resolution if saturated else None,
        "beats_p99": our_total > float(np.percentile(field_totals, 99)),
    }


def attach_ownership(
    forecasts: pd.DataFrame, history: pd.DataFrame, ownership_col: str = "selected"
) -> pd.DataFrame:
    """Join realised ownership counts onto a forecast frame, keyed by name and gameweek."""
    if ownership_col not in history.columns:
        raise KeyError(f"history has no {ownership_col} column")
    owned = (
        history.groupby(["season", "GW", "name"], as_index=False)[ownership_col].max()
        .rename(columns={"GW": "gw"})
    )
    return forecasts.merge(owned, on=["season", "gw", "name"], how="left")
