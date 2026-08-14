"""Secondary scoring components: saves, defensive contributions, and cards.

Individually small, collectively decisive. Defensive contributions alone are worth 2 points
and a defender who reliably hits the threshold earns more from it than from the occasional
goal. All three are modelled as rates per 90, shrunk with the same machinery as everywhere
else, then converted to points through the actual scoring rules — which are threshold-based,
not linear, so expectations must be taken over the distribution rather than applied to the
mean.

The distinction matters most for saves. A goalkeeper expected to make 2.8 saves does NOT earn
2.8/3 points: the rule is one point per *three* saves, so the expectation has to run over the
whole count distribution. Applying the rule to the mean instead over-rewards keepers who
hover just under a multiple of three.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..features.rates import NOISE_SCALE, fit_gamma_prior, shrink
from .minutes import BUCKET_LONG, BUCKET_MINUTES, BUCKET_SHORT

log = logging.getLogger(__name__)

MINUTES_PER_90 = 90.0

# Representative minutes per bucket, shared with the minutes model so the threshold is
# evaluated on the same two worlds the rest of the system conditions on.
BUCKET_SHORT_MINUTES = BUCKET_MINUTES[BUCKET_SHORT]
BUCKET_LONG_MINUTES = BUCKET_MINUTES[BUCKET_LONG]
MAX_COUNT = 40  # well past the observed maximum for any of these stats

# League-average team expected goals, used to scale shots faced by fixture difficulty.
LEAGUE_GOALS_PER_TEAM = 1.43

# SPRINT: red cards are modelled as a fixed fraction of yellows rather than separately.
# They are rare (roughly 1 in 25 bookings) and the points impact is small.
RED_CARD_FRACTION = 0.04


def shrunk_rate(
    counts: pd.Series, exposure: pd.Series, stat: str, groups: pd.Series | None = None
) -> pd.DataFrame:
    """Per-90 rate for a countable stat, shrunk within groups."""
    noise = NOISE_SCALE.get(stat, 1.0)
    if groups is None:
        prior = fit_gamma_prior(counts, exposure, noise)
        return shrink(counts, exposure, prior)

    pieces = []
    for index in groups.groupby(groups).groups.values():
        prior = fit_gamma_prior(counts.loc[index], exposure.loc[index], noise)
        pieces.append(shrink(counts.loc[index], exposure.loc[index], prior))
    return pd.concat(pieces).reindex(counts.index)


# --- saves ----------------------------------------------------------------


def expected_save_points(expected_saves: pd.Series, per_point: int = 3) -> pd.Series:
    """E[floor(saves / 3)], taken over the count distribution rather than at the mean.

    Applying the threshold to the mean is wrong and biased: a keeper expected to make 2.8
    saves would score zero, when in fact he clears three saves nearly half the time.
    """
    counts = np.arange(MAX_COUNT + 1)
    points = counts // per_point
    return pd.Series(
        [float((points * poisson.pmf(counts, max(lam, 0.0))).sum()) for lam in expected_saves],
        index=expected_saves.index,
    )


def expected_saves(
    saves_per90: pd.Series, expected_minutes: pd.Series, opponent_expected_goals: pd.Series
) -> pd.Series:
    """Saves scale with how much the opponent threatens.

    A keeper's own history sets his baseline workload; the fixture scales it. Without this a
    keeper facing the strongest attack in the league would be forecast the same volume as one
    facing the weakest, which gets goalkeeper selection exactly backwards — save points and
    clean-sheet points pull in opposite directions.
    """
    difficulty = (opponent_expected_goals / LEAGUE_GOALS_PER_TEAM).clip(0.3, 3.0)
    return saves_per90 * (expected_minutes / MINUTES_PER_90) * difficulty


# --- defensive contributions ---------------------------------------------


def defensive_contribution_points(
    defcon_per90: pd.Series,
    expected_minutes: pd.Series,
    position: pd.Series,
    thresholds: dict[str, int],
    points: int = 2,
    p_short: pd.Series | None = None,
    p_long: pd.Series | None = None,
) -> pd.Series:
    """P(clearing the CBIT/CBIRT threshold) x 2, per fixture.

    Threshold-based and capped at one award per match, so this is a tail probability rather
    than a rate: `P(count >= threshold)`. Defenders need 10, midfielders and forwards 12,
    and goalkeepers do not score it at all. League means sit at roughly 7.5 and 7.9, so most
    players clear the bar only sometimes — which is precisely why the distribution matters
    more than the average.

    **No opponent term, and that is deliberate — checked 2026-08-14.** It looks like an
    omission: a defender facing Man City should surely make more tackles and blocks than one
    facing a promoted side, and clean sheets are fixture-dependent, so why not this? Measured
    on 7,815 player-matches of 60+ minutes in 2025-26, the only season carrying the stat:

        TEAM level     mean defcon/90 vs goals conceded per match    r = +0.46
                       Arsenal 6.46 / 0.64        Burnley 7.26 / 1.78
        WITHIN player  defcon/90 vs goals conceded that match        r = -0.055
                       on a clean sheet 6.93      when conceding 6.92

    The team effect is real — weak defences do defend more — but it is entirely BETWEEN teams
    and players, not within them. The same player produces the same defensive volume whether
    his side keeps a clean sheet or ships three. So there is no fixture-level variation to
    model, and the between-team part is already captured: `defcon_per90` is an empirical
    per-player rate, so a Burnley defender's 7.26 and an Arsenal defender's 6.46 are measured
    directly rather than inferred. An opponent adjustment on top would fit an effect the data
    says does not exist.

    Related, for anyone weighing defcon against clean sheets: across regular defenders the two
    are barely substitutes, correlating at only -0.07. Defcon varies far more by ROLE — a
    ball-winning centre-back against an attacking full-back — than by team quality, so a
    strong defence does not systematically mean low defcon.

    Counts are treated as Poisson. They ARE overdispersed — measured against our own predicted
    rates on 2025-26, residual variance/mean is 1.78 for defenders, 1.60 for midfielders and
    1.49 for forwards — but modelling that does not consistently improve the number this
    function actually produces:

        position   threshold   Poisson   negative binomial   realised
        DEF           10        0.2633        0.2749          0.2794
        MID           12        0.1804        0.1961          0.1871
        FWD           12        0.0129        0.0238          0.0165

    A negative binomial helps defenders and makes midfielders and forwards worse, because
    fattening the tail overshoots once the threshold sits well above the mean. Shipping it
    would trade one position's accuracy for two, so Poisson stays and the measurement is
    recorded here instead. Worth revisiting when a second season of the stat exists — this
    rests on one.
    """
    limits = position.map(thresholds)

    def _award(minutes: pd.Series) -> pd.Series:
        counts = defcon_per90 * (minutes / MINUTES_PER_90)
        out = pd.Series(0.0, index=defcon_per90.index)
        for i, (lam, limit) in enumerate(zip(counts, limits, strict=True)):
            if pd.isna(limit) or lam <= 0:
                continue
            out.iloc[i] = points * float(poisson.sf(int(limit) - 1, lam))
        return out

    if p_short is None or p_long is None:
        # Threshold applied at MEAN minutes. Biased low, and kept only so callers without the
        # bucket probabilities still work — `points.assemble` always supplies them.
        return _award(expected_minutes)

    # Conditional on the minutes bucket, then mixed — the same treatment the saves docstring
    # above insists on, applied here at last. `P(N >= T)` is sharply convex in lambda while T
    # sits above it (10 against a league mean near 7.5), so evaluating it at average minutes
    # is not an approximation of the mixture, it is systematically about HALF of it:
    #
    #     2025-26        at mean minutes   bucket-conditional   realised
    #     total defcon        1394               2599             2834
    #
    # That 1,440-point gap is essentially the whole of the season's residual under-prediction
    # (expected 32,978 against actual 34,409), which had been wrongly attributed to the match
    # model. `models/distribution.py` had always done this correctly, so the two paths
    # disagreed by 4.2% on 2025-26 defenders — and the r=0.997 correlation used to cross-check
    # them is exactly the statistic that cannot see a level shift.
    return (
        p_short.fillna(0.0) * _award(pd.Series(BUCKET_SHORT_MINUTES, index=defcon_per90.index))
        + p_long.fillna(0.0) * _award(pd.Series(BUCKET_LONG_MINUTES, index=defcon_per90.index))
    )


# --- cards ----------------------------------------------------------------


def expected_card_points(
    yellows_per90: pd.Series,
    expected_minutes: pd.Series,
    yellow_points: int = -1,
    red_points: int = -3,
) -> pd.Series:
    """Expected disciplinary points. Small, but consistently negative for some players."""
    yellows = yellows_per90 * (expected_minutes / MINUTES_PER_90)
    reds = yellows * RED_CARD_FRACTION
    return yellows * yellow_points + reds * red_points


# --- clean sheets and goals conceded --------------------------------------


def defensive_points(
    p_clean_sheet: pd.Series,
    p_long: pd.Series,
    expected_conceded_penalty: pd.Series,
    position: pd.Series,
    clean_sheet_points: dict[str, int],
) -> pd.Series:
    """Clean-sheet and goals-conceded points.

    A clean sheet only pays if the player is on the pitch for 60+ minutes, so the team's
    clean-sheet probability is gated by `p_long` from the minutes model. This is the join
    between the match model and the minutes model, and getting it wrong would credit
    substitutes with full clean-sheet value.
    """
    per_clean_sheet = position.map(clean_sheet_points).fillna(0).astype(float)
    earned = p_clean_sheet * p_long * per_clean_sheet
    # The conceded deduction applies to keepers and defenders whenever they are playing.
    conceded_applies = position.isin(["GK", "DEF"]).astype(float)
    return earned + expected_conceded_penalty * conceded_applies * p_long
