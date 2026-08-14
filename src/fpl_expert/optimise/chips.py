"""Chips: what each is worth in a given gameweek, and when to play it.

Eight chips over a season — a Wildcard, Free Hit, Bench Boost and Triple Captain in each
half — and none of them carry over. A chip left unplayed at the GW19 deadline is simply
thrown away, which makes this an **optimal stopping** problem rather than a valuation one.
Knowing a chip is worth 18 points this week is useless without knowing whether 24 is likely
later, and how many chances remain.

Each chip's value is a different quantity, and conflating them is the usual mistake:

    Bench Boost      the four bench players score      -> value = expected bench points
    Triple Captain   captain scores 3x rather than 2x  -> value = ONE extra copy of his score
    Free Hit         any legal XI for one week, then revert
    Wildcard         unlimited transfers, squad persists

Bench Boost and Triple Captain are worth most in **double** gameweeks, where players play
twice; Free Hit is worth most in **blanks**, where much of your squad has no fixture. That is
why fixture structure drives chip timing far more than form does.

The stopping rule is the honest core here. Playing whenever a chip looks "good" burns it in
week three; holding out for perfection wastes it at expiry. The policy plays when the current
value beats what remains achievable, and becomes progressively less fussy as the deadline
approaches — because a chip worth 15 now beats a chip worth 25 you never get to use.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHIPS = ("bench_boost", "triple_captain", "free_hit", "wildcard")

# Value below which a chip is not worth playing except to avoid losing it entirely.
DEFAULT_FLOOR = {
    "bench_boost": 8.0,
    "triple_captain": 6.0,
    "free_hit": 12.0,
    "wildcard": 15.0,
}


def bench_boost_value(squad: pd.DataFrame, bench_ids: set, points_col: str) -> float:
    """Expected points from the four bench players, who normally score nothing."""
    bench = squad[squad["player_id"].isin(bench_ids)]
    return float(bench[points_col].fillna(0).sum())


def triple_captain_value(xi: pd.DataFrame, captain_id, points_col: str) -> float:
    """The armband already doubles; the chip adds a THIRD copy, not a third of the score.

    Valuing this as `3 x points` instead of one extra copy overstates it by a factor of
    three and will burn the chip on a good-but-not-exceptional week.
    """
    row = xi[xi["player_id"] == captain_id]
    if row.empty:
        row = xi.nlargest(1, points_col)
    return float(row[points_col].fillna(0).iloc[0]) if not row.empty else 0.0


def free_hit_value(
    current_xi_points: float, best_possible_points: float
) -> float:
    """Gain from fielding an unconstrained legal XI for one week, then reverting.

    Worth most in a blank gameweek: when half your squad has no fixture your baseline
    collapses while the best available XI does not.
    """
    return max(0.0, best_possible_points - current_xi_points)


def wildcard_value(
    current_horizon_points: float, best_horizon_points: float, hits_avoided: int = 0,
    hit_cost: float = 4.0,
) -> float:
    """Gain from restructuring permanently, measured over the planning horizon.

    Unlike a Free Hit the new squad persists, so it is valued across the horizon rather than
    a single week, plus the hits it saves you from taking one at a time.
    """
    return max(0.0, best_horizon_points - current_horizon_points) + hits_avoided * hit_cost


# How much better the BEST remaining week is than a typical one, per chip. Fixture structure
# drives this: Bench Boost and Triple Captain roughly double in a DOUBLE gameweek because
# players play twice, and Free Hit spikes hardest in a BLANK because the baseline collapses
# while the best available XI does not.
#
# This encodes what a fixture-aware planner would read directly off the calendar — double and
# blank gameweeks are announced weeks ahead. Without it the policy compares "this week" to
# "an average week" and fires almost immediately, which measurably burns chips: playing Bench
# Boost in GW2 of 2025-26 returned +21 where GW26 returned +70.
BEST_WEEK_MULTIPLIER = {
    "bench_boost": 2.0,
    "triple_captain": 2.0,
    "free_hit": 2.5,
    "wildcard": 1.5,
}


def expected_best_remaining(
    known_future: list[float],
    unknown_weeks: int = 0,
    unknown_estimate: float = 0.0,
    chip: str | None = None,
) -> float:
    """Best value still achievable before the chip expires.

    Forecasts only reach a few gameweeks ahead, so weeks beyond that are represented by
    `unknown_estimate` scaled by what a good week looks like for this chip. Ignoring them
    entirely makes the policy far too eager, because it behaves as though the season ends at
    the edge of the forecast horizon.
    """
    candidates = list(known_future)
    if unknown_weeks > 0:
        multiplier = BEST_WEEK_MULTIPLIER.get(chip, 1.0)
        # More remaining chances mean a higher expected maximum, but with strongly
        # diminishing returns — a fifth chance adds far less than a second.
        candidates.append(
            unknown_estimate * multiplier * (1.0 + 0.15 * np.log1p(unknown_weeks))
        )
    return max(candidates) if candidates else 0.0


def should_play(
    chip: str,
    current_value: float,
    known_future: list[float],
    *,
    unknown_weeks: int = 0,
    unknown_estimate: float = 0.0,
    weeks_to_expiry: int = 99,
    floor: dict[str, float] | None = None,
    urgency_weeks: int = 3,
) -> tuple[bool, str]:
    """Whether to play the chip now, with the reason.

    Three tests, in order:
      1. is it above the floor at which the chip is worth using at all;
      2. does it beat the best remaining opportunity;
      3. is expiry close enough that holding out is a bad bet.
    """
    floor = {**DEFAULT_FLOOR, **(floor or {})}
    minimum = floor.get(chip, 0.0)
    best_remaining = expected_best_remaining(
        known_future, unknown_weeks, unknown_estimate, chip
    )

    if weeks_to_expiry <= 0:
        return current_value > 0, "final chance before expiry"

    if weeks_to_expiry <= urgency_weeks:
        # Near expiry the comparison is against what is actually left, not an ideal week.
        # A chip worth 15 now beats a chip worth 25 you never get to play.
        if current_value >= best_remaining:
            return True, f"expiring in {weeks_to_expiry} gw, best remaining {best_remaining:.1f}"
        return False, f"holding: {best_remaining:.1f} still reachable before expiry"

    if current_value < minimum:
        return False, f"below floor ({current_value:.1f} < {minimum:.1f})"
    if current_value < best_remaining:
        return False, f"holding for {best_remaining:.1f}"
    return True, f"best available ({current_value:.1f} vs {best_remaining:.1f} remaining)"


def chip_windows(rules: dict) -> dict[str, list[tuple[int, int]]]:
    """Play windows per chip, from the verified scoring rules."""
    return {
        name: [tuple(window) for window in config["windows"]]
        for name, config in rules.get("chips", {}).items()
    }


def active_window(windows: list[tuple[int, int]], gw: int) -> tuple[int, int] | None:
    """The window a gameweek falls in, if any."""
    for start, stop in windows:
        if start <= gw <= stop:
            return (start, stop)
    return None


def plan_summary(schedule: dict[str, int]) -> pd.DataFrame:
    """Tabulate a chip schedule for the weekly brief."""
    return pd.DataFrame(
        [{"chip": chip, "gameweek": gw} for chip, gw in sorted(schedule.items(), key=lambda x: x[1])]
    )
