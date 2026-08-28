"""What a bench place is actually worth: the chance an autosub reaches it.

The problem this replaces
-------------------------

`select_squad` valued every bench place at a flat `DEFAULT_BENCH_WEIGHT = 0.10`, a guess.
`recommend_transfers` valued bench places at **1.0** — it has no concept of an XI at all, so a
fourth-choice defender counted exactly as much as the captain. The two optimisers therefore
priced the same squad differently, and only one of them could be right.

Neither number is derived from anything. This module derives one.

The model
---------

A bench player scores only when an autosub fires, and an autosub fires when a STARTER plays
zero minutes. So for the k-th outfield bench slot:

    weight_k = P(at least k starters blank)

`N`, the number of blanking starters, is a sum of independent Bernoulli draws with different
probabilities — a Poisson-binomial — computed exactly here by convolution rather than
approximated. The bench goalkeeper is separate: he plays only if the starting keeper blanks,
so his weight is that single probability.

Deliberately NOT multiplied by the bench player's own `p_appear`: his expected points already
integrate over his own minutes distribution, and multiplying again would count it twice.

Known simplifications, both small and in opposite directions
------------------------------------------------------------

* **Formation legality is ignored.** FPL skips a bench player whose entry would leave an
  illegal XI (fewer than 3 defenders, say). This overstates the weight for a player whose
  position is already at its minimum.
* **A bench player ahead in the queue who did not play is skipped**, which promotes those
  behind him. This understates the weight for slots 2 and 3.

Neither is worth modelling until the headline effect is shown to matter, and the errors do not
compound in the same direction.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def blank_distribution(p_zero) -> np.ndarray:
    """Exact pmf of how many of these players fail to appear.

    Poisson-binomial by convolution: start with "nobody has blanked, probability 1", then fold
    in one player at a time. Exact and fast at this size — an XI is ten outfield players.
    """
    pmf = np.array([1.0])
    for p in np.clip(np.asarray(p_zero, dtype=float), 0.0, 1.0):
        stay = pmf * (1.0 - p)
        blank = pmf * p
        pmf = np.append(stay, 0.0) + np.append(0.0, blank)
    return pmf


def autosub_slot_weights(xi_p_zero, n_slots: int = 3) -> list[float]:
    """P(the k-th outfield bench slot is reached), for k = 1..n_slots.

    This is the survival function of the blank count, so the weights fall away steeply: with a
    typical XI it is roughly 0.6 / 0.25 / 0.07, not the flat 0.10 that was there before. The
    first bench slot was badly under-valued and the third slightly over-valued.
    """
    if n_slots <= 0:
        return []
    pmf = blank_distribution(xi_p_zero)
    tail = 1.0 - np.cumsum(pmf)                  # tail[k-1] = P(N > k-1) = P(N >= k)
    return [float(np.clip(tail[k - 1], 0.0, 1.0)) if k - 1 < len(tail) else 0.0
            for k in range(1, n_slots + 1)]


def squad_value(
    held: pd.DataFrame,
    rules: dict,
    *,
    points_col: str = "horizon_points",
    p_zero_col: str = "p_zero",
    xi_points_col: str | None = None,
) -> float:
    """Expected value of a fifteen: the XI in full, plus each bench place at its autosub odds.

    This is the quantity a transfer should be judged on. The sum over all fifteen — which is
    what `recommend_transfers` maximises — treats a bench slot as a starting slot and so
    over-values squad depth, which is how a hit gets taken to upgrade a player who never plays.

    `xi_points_col` lets the XI be CHOSEN on one column and VALUED on another: you pick the
    team you would actually field on this week's points, then value the squad over the horizon.
    """
    from ..backtest.season_sim import pick_xi

    if held.empty:
        return 0.0
    ranked, starters = pick_xi(held, rules, xi_points_col or points_col)
    bench = ranked.drop(index=starters.index)
    value = float(starters[points_col].fillna(0).sum())
    if bench.empty:
        return value

    p_zero = (
        starters[p_zero_col].fillna(0.0) if p_zero_col in starters
        else pd.Series(0.0, index=starters.index)
    )
    outfield_starters = starters[starters["position"] != "GK"]
    outfield_p_zero = p_zero.reindex(outfield_starters.index).fillna(0.0)

    # The bench keeper covers exactly one event: the starting keeper not playing.
    keepers = starters[starters["position"] == "GK"]
    gk_blank = float(p_zero.reindex(keepers.index).fillna(0.0).iloc[0]) if len(keepers) else 0.0

    bench_gk = bench[bench["position"] == "GK"]
    bench_outfield = bench[bench["position"] != "GK"].sort_values(points_col, ascending=False)

    value += gk_blank * float(bench_gk[points_col].fillna(0).sum())
    weights = autosub_slot_weights(outfield_p_zero, n_slots=len(bench_outfield))
    value += float(
        (bench_outfield[points_col].fillna(0).to_numpy() * np.asarray(weights)).sum()
    )
    return value


def xi_value(squad: pd.DataFrame, rules: dict, points_col: str = "expected_points") -> float:
    """This gameweek's score for a squad: the best legal XI plus the armband again.

    The armband is counted twice because the captain scores twice — the same convention the
    manifest and `SquadSolution.expected_points` use, so the two are comparable.
    """
    from ..backtest.season_sim import pick_xi

    if squad.empty:
        return 0.0
    _, starters = pick_xi(squad, rules, points_col)
    if starters.empty:
        return 0.0
    return float(starters[points_col].sum() + starters[points_col].max())
