"""Decisions judged by beating the field, using the points distribution and a simulated field.

`risk.py` approximates a rank objective by discounting expected points by ownership. That is a
heuristic standing in for a quantity nobody could compute: the probability of finishing above
a rival. Two things that did not exist when it was written now do — a full points distribution
per player (`models/distribution.py`) and a field of twenty thousand simulated managers
calibrated against real ownership (`backtest/field_sim.py`) — and together they make the real
quantity reachable by simulation.

## The thing worth being clear about first

**Under a total-points objective the existing captaincy rule is already optimal, provably.**
The armband adds a second copy of the captain's score, so the objective is linear and
`E[base + X_c]` is maximised by `argmax E[X_c]` whatever the shape of `X_c`. A steady six and
a one-in-four twenty-four are worth exactly the same. No distribution, however good, can
improve on picking the highest mean.

So this module is not an improvement on captaincy in general. It only does something a mean
cannot when the objective is RANK, and then the direction is conditional rather than fixed:

    versus a field averaging 60, holding a base of 55
        steady captain   (mean 6, sd 1)     beat-rate 0.5215
        volatile captain (mean 6, sd 10)    beat-rate 0.5064

Ahead of the field, the steady captain wins because variance can only cost you. Behind it,
the ordering reverses, because a manager who cannot win by standing still has to gamble.
Any advice that does not depend on where you stand is not really a rank objective.

## How it is computed

Monte Carlo, with the draws SHARED between us and the field. That sharing is the whole reason
this beats an ownership heuristic: when the field owns the player being evaluated, a big score
for him lifts their totals as much as ours and cancels out of the difference, automatically and
in the right proportion. Nothing has to be assumed about how much a template captain is worth,
because the simulated field either owns him or does not.

Season-to-date enters on both sides, so the objective knows whether it is protecting a lead or
chasing. That is what `lambda_rank` structurally could not express: a fixed ownership discount
gives the same advice in GW3 and GW38.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_DRAWS = 200


def expected_beat_rate(
    our_scores: np.ndarray, field_scores: np.ndarray
) -> float:
    """Share of the field we finish above, averaged over draws.

    `our_scores` is one total per draw; `field_scores` is `(n_draws, n_managers)` or a single
    `(n_managers,)` row shared by every draw. Higher is better, and it is already the
    percentile — no rank conversion is applied here, because converting to a position among
    11M managers implies a precision the field size does not support.
    """
    ours = np.asarray(our_scores, dtype=float).reshape(-1, 1)
    field = np.atleast_2d(np.asarray(field_scores, dtype=float))
    if field.shape[0] == 1:
        field = np.broadcast_to(field, (ours.shape[0], field.shape[1]))
    return float((field < ours).mean())


def evaluate_captains(
    candidates: dict,
    starters: dict,
    week_draws: np.ndarray,
    field_week: np.ndarray,
    *,
    our_baseline: float = 0.0,
    field_baseline: np.ndarray | None = None,
) -> pd.DataFrame:
    """Beat-rate for each candidate armband, on shared outcome draws.

    `candidates` and `starters` map player id to that player's COLUMN in `week_draws`, which
    is `(n_draws, n_players)` of sampled gameweek scores. `field_week` is `(n_draws,
    n_managers)` from the same draws — the same sampled world both sides live in.

    Returns one row per candidate with its expected points and its beat-rate, sorted by the
    latter. `beat_rate_delta` is against whichever candidate has the highest mean, i.e. the
    pick a points objective would make, so the table reads as "what does going off-piste buy".
    """
    if not candidates:
        return pd.DataFrame(columns=["player_id", "expected_points", "beat_rate"])

    starting_columns = list(starters.values())
    base = week_draws[:, starting_columns].sum(axis=1) + our_baseline
    field = field_week + (0.0 if field_baseline is None else field_baseline[None, :])

    rows = []
    for player_id, column in candidates.items():
        totals = base + week_draws[:, column]
        rows.append({
            "player_id": player_id,
            "expected_points": float(week_draws[:, column].mean()),
            "beat_rate": expected_beat_rate(totals, field),
        })

    table = pd.DataFrame(rows)
    points_pick = table.loc[table["expected_points"].idxmax()]
    table["beat_rate_delta"] = table["beat_rate"] - points_pick["beat_rate"]
    table["points_delta"] = table["expected_points"] - points_pick["expected_points"]
    return table.sort_values("beat_rate", ascending=False).reset_index(drop=True)


class RankCaptainSelector:
    """Chooses the armband to maximise the share of the field beaten.

    Built as a callable object rather than a function because it has to carry state a season
    simulator cannot supply: the field it is competing against, the distributions to draw
    from, and the field's own running total. It is passed to `simulate_season` as
    `captain_selector` and is otherwise a drop-in for the argmax rule.

    `min_points_delta` is a guard rail with a purpose. A pure beat-rate objective will hand
    the armband to a genuinely worse player for a third-decimal gain, and beat-rate is
    estimated from a few hundred draws, so those gains are frequently noise. Requiring the
    alternative to cost less than a fixed number of expected points keeps the search to
    decisions where the two objectives genuinely disagree.
    """

    def __init__(
        self,
        field_trace,
        distributions: dict,
        realised_points: np.ndarray,
        *,
        n_draws: int = DEFAULT_DRAWS,
        seed: int = 0,
        pool: int = 6,
        min_points_delta: float = 1.0,
    ) -> None:
        self.trace = field_trace
        self.distributions = distributions      # gw -> (keys frame, PointsDistribution)
        # `(n_gw, n_players)` of what the field actually scores, so their running total tracks
        # the real season rather than a resampled one. Only OUR decision is made under
        # uncertainty; their history is already settled by the time we choose.
        self.realised_points = realised_points
        self.n_draws = n_draws
        self.rng = np.random.default_rng(seed)
        self.pool = pool
        self.min_points_delta = min_points_delta
        self.field_baseline = np.zeros(field_trace.n_managers)
        self.decisions: list[dict] = []
        self._gw_position = {gw: i for i, gw in enumerate(field_trace.gameweeks)}
        self._player_column = pd.Series(
            np.arange(len(field_trace.player_ids)), index=field_trace.player_ids
        )

    def __call__(self, gw, held, points_col, points_so_far):
        """Pick an armband, and advance the field's running total by this gameweek."""
        step = self._gw_position.get(gw)
        entry = self.distributions.get(gw)
        if step is None or entry is None or held.empty:
            return held.nlargest(1, points_col)["player_id"].iloc[0] if not held.empty else None

        keys, distribution = entry
        # Columns in the field's player indexing, for everyone this gameweek.
        columns = self._player_column.reindex(keys["player_id"]).to_numpy()
        valid = ~np.isnan(columns)
        draws = distribution.sample(self.rng, draws=self.n_draws)

        week = np.zeros((self.n_draws, len(self._player_column)))
        week[:, columns[valid].astype(int)] = draws[:, valid]
        field_week = np.stack(
            [self.trace.score_gameweek(week[d], step) for d in range(self.n_draws)]
        )

        position = pd.Series(np.arange(len(keys)), index=keys["player_id"].to_numpy())
        squad_columns = {
            int(pid): int(columns[position[pid]])
            for pid in held["player_id"]
            if pid in position.index and not np.isnan(columns[position[pid]])
        }
        contenders = held.nlargest(self.pool, points_col)["player_id"]
        candidates = {int(p): squad_columns[int(p)] for p in contenders if int(p) in squad_columns}

        table = evaluate_captains(
            candidates, squad_columns, week, field_week,
            our_baseline=points_so_far, field_baseline=self.field_baseline,
        )
        # Advance the field regardless of what we chose — they play the week either way, and
        # their history is settled by the time the next decision is made.
        self.field_baseline = self.field_baseline + self.trace.score_gameweek(
            self.realised_points[step], step
        )

        if table.empty:
            return held.nlargest(1, points_col)["player_id"].iloc[0]

        affordable = table[table["points_delta"] >= -self.min_points_delta]
        choice = (affordable if not affordable.empty else table).iloc[0]
        self.decisions.append({"gw": gw, **choice.to_dict()})
        return choice["player_id"]
