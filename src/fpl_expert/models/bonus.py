"""Bonus points.

Bonus is 7.6% of all FPL points — more than clean sheets are worth to a midfielder, and
systematically ignored by simple models. It is awarded 3/2/1 to the top three players by BPS
in each match, so it is fundamentally a *ranking within a match*, not an independent per-player
event.

**This is the crudest model in the system, deliberately.** A proper treatment recomputes BPS
from its published components and simulates the within-match ranking. That is not possible to
calibrate right now for a specific reason: FPL retuned the BPS formula for 2026/27 —
clearances/blocks/interceptions moved from 1 point per 2 to 1 per 3, the penalty for being
tackled was removed, the outside-the-box save metric was dropped, and penalty saves fell from
8 BPS to 7. Every historical `bps` value is therefore on superseded rules, and a component
model fitted to them would be confidently wrong about exactly the player types the retune was
designed to shift: goalkeepers, attacking full-backs, and creative players.

So the sprint model uses realised `bonus` rather than `bps`: a shrunk bonus-per-90 rate,
scaled by minutes and by how well the player's team is expected to do. Bonus concentrates in
winning teams and among goalscorers, which this captures crudely through the team's expected
goals.

SPRINT — revisit by rebuilding BPS from raw components under the 2026/27 weights and
simulating the match ranking. Until then, treat bonus forecasts as the least trustworthy
column in the points assembler.
"""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

MINUTES_PER_90 = 90.0
LEAGUE_GOALS_PER_TEAM = 1.43

# Only three players per match earn bonus at all, so a per-90 rate cannot exceed this by
# much; the cap stops a small-sample player being projected absurdly.
MAX_BONUS_PER_90 = 1.5


def expected_bonus(
    bonus_per90: pd.Series,
    expected_minutes: pd.Series,
    team_expected_goals: pd.Series,
    p_clean_sheet: pd.Series | None = None,
) -> pd.Series:
    """Expected bonus points for a player in a fixture.

    Scaled by team expected goals because bonus follows goal involvement and winning teams
    collect most of it. Clean sheets add a smaller lift for defensive players, who pick up
    BPS from defensive actions when the team keeps the opposition out.
    """
    base = bonus_per90.clip(upper=MAX_BONUS_PER_90) * (expected_minutes / MINUTES_PER_90)
    attack_scale = (team_expected_goals / LEAGUE_GOALS_PER_TEAM).clip(0.4, 2.5)
    scaled = base * attack_scale
    if p_clean_sheet is not None:
        # A modest lift: a clean sheet raises BPS for the whole defensive unit.
        scaled = scaled * (1.0 + 0.25 * (p_clean_sheet - 0.25).clip(lower=-0.25))
    return scaled.clip(lower=0.0)


BONUS_PER_MATCH = 6.0   # 3 + 2 + 1, awarded to exactly three players


def allocate_match_bonus(propensity: pd.Series, per_match: float = BONUS_PER_MATCH) -> pd.Series:
    """Distribute a match's six bonus points across its players by relative propensity.

    Bonus is a *ranking within a match*, not an independent per-player event: exactly six
    points exist per fixture no matter how many strong players are on the pitch. Modelling
    each player independently violates that — two goalscoring midfielders in the same match
    cannot both collect 3 points, and an unconstrained model happily predicts they will.

    Allocating a fixed pool also sharpens discrimination, which is what the ablation
    actually rewards: a component that shifts every player equally changes no decision.
    """
    total = float(propensity.clip(lower=0).sum())
    if total <= 0:
        return pd.Series(0.0, index=propensity.index)
    return propensity.clip(lower=0) / total * per_match


def bonus_confidence_note() -> str:
    """Machine-readable caveat for the weekly report, so the caveat travels with the number."""
    return (
        "Bonus is modelled from realised bonus rates, not BPS components. FPL retuned the "
        "BPS formula for 2026/27, so historical BPS is on superseded rules. Treat bonus as "
        "the least reliable component."
    )
