"""The recent-form baseline: the bar a forecasting system actually has to clear.

It replaces the price-only benchmark, which turned out to be unusable. Maximising a
price-derived objective under a budget that binds at exactly £100m is close to degenerate —
two solutions 0.8% apart in objective shared 5 of 15 players and scored 90 against 36 in the
same gameweek — so there was no single number to beat. Form is not collinear with the budget,
and it ranks players at 0.641 against price's 0.381 (our model: 0.692), which makes it both
meaningful and hard.

Everything here guards the one property that would silently invalidate it: form must be built
only from gameweeks that had already been played.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.season_sim import (
    FORM_PRICE_SCALE,
    attach_form,
    form_scores,
)


def _frame(points_by_gw, price=5.0, player_id=1):
    return pd.DataFrame([
        {"gw": gw, "player_id": player_id, "actual_points": pts, "price": price}
        for gw, pts in sorted(points_by_gw.items())
    ])


def test_form_never_contains_the_gameweek_it_scores():
    """The guard that matters. A player's own result must not inform the squad picked to
    include him — that is hindsight, and it would inflate the baseline without limit."""
    quiet = _frame({1: 0.0, 2: 0.0, 3: 0.0})
    explosive = _frame({1: 0.0, 2: 0.0, 3: 99.0})

    assert form_scores(quiet).tolist() == pytest.approx(form_scores(explosive).tolist())


def test_a_later_result_changes_only_later_form():
    frame = _frame({1: 0.0, 2: 6.0, 3: 0.0, 4: 0.0})
    scores = form_scores(frame)

    altered = frame.copy()
    altered.loc[altered["gw"] == 2, "actual_points"] = 0.0
    changed = form_scores(altered)

    assert scores.iloc[0] == pytest.approx(changed.iloc[0])
    assert scores.iloc[1] == pytest.approx(changed.iloc[1])
    assert scores.iloc[2] > changed.iloc[2]      # GW3 can see GW2, and should


def test_the_opening_gameweek_falls_back_to_price():
    """With no history the baseline picks on price, which is the information a real manager
    has in August. Stated rather than hidden: GW1 is the one week it degenerates."""
    scores = form_scores(_frame({1: 0.0}, price=8.0))
    assert scores.iloc[0] == pytest.approx(8.0 * FORM_PRICE_SCALE)


def test_evidence_progressively_outweighs_the_price_prior():
    """A £10m player who never returns must decay toward zero, not sit on his price."""
    scores = form_scores(_frame(dict.fromkeys(range(1, 9), 0.0), price=10.0)).tolist()
    assert scores[0] == pytest.approx(10.0 * FORM_PRICE_SCALE)
    assert all(later <= earlier for earlier, later in pairwise(scores))
    assert scores[-1] < scores[0] / 3


def test_the_window_forgets_old_form():
    """Six gameweeks, not the whole season: a player who stopped returning in October should
    not still be priced on August."""
    hot_then_cold = _frame({1: 20.0, **dict.fromkeys(range(2, 10), 0.0)})
    always_cold = _frame(dict.fromkeys(range(1, 10), 0.0))

    hot = form_scores(hot_then_cold).tolist()
    cold = form_scores(always_cold).tolist()
    assert hot[2] > cold[2]                       # still remembered
    assert hot[-1] == pytest.approx(cold[-1])     # forgotten


def test_players_are_scored_independently():
    frame = pd.concat([
        _frame({1: 0.0, 2: 10.0, 3: 0.0}, player_id=1),
        _frame({1: 0.0, 2: 0.0, 3: 0.0}, player_id=2),
    ], ignore_index=True)
    scores = form_scores(frame)
    frame = frame.assign(form=scores)

    scorer = frame[(frame["player_id"] == 1) & (frame["gw"] == 3)]["form"].iloc[0]
    blank = frame[(frame["player_id"] == 2) & (frame["gw"] == 3)]["form"].iloc[0]
    assert scorer > blank


def test_form_is_returned_aligned_to_the_input_rows():
    """`form_scores` sorts internally to compute the rolling window. If it returned the sorted
    order the caller would silently attach one player's form to another."""
    frame = pd.concat([
        _frame({1: 0.0, 2: 10.0}, player_id=1),
        _frame({1: 0.0, 2: 0.0}, player_id=2),
    ], ignore_index=True).sample(frac=1.0, random_state=3)

    scored = frame.assign(form=form_scores(frame))
    for row in scored.itertuples():
        rebuilt = form_scores(frame[frame["player_id"] == row.player_id])
        expected = rebuilt[frame[frame["player_id"] == row.player_id]["gw"] == row.gw]
        assert row.form == pytest.approx(float(expected.iloc[0]))


# --- the horizon views ------------------------------------------------------------------


def _season(n_gws=4):
    rows = []
    for gw in range(1, n_gws + 1):
        for pid in (1, 2):
            rows.append({
                "gw": gw, "player_id": pid, "price": 5.0 + pid,
                "actual_points": float(gw * pid),
            })
    return pd.DataFrame(rows)


def test_horizon_views_carry_the_decision_weeks_form_not_the_target_weeks():
    """The trap that caught the price baseline. A GW2 manager judging GW4 knows how his
    players have played up to GW2 and nothing more. Giving those frames GW4's own form would
    rebuild, on the baseline, exactly the lookahead this project spent a session removing."""
    season = _season()
    lookahead = {2: {2: season[season["gw"] == 2], 4: season[season["gw"] == 4]}}

    scored, forward = attach_form(season, lookahead)
    at_gw2 = scored[scored["gw"] == 2].set_index("player_id")["form_score"]
    view_of_gw4 = forward[2][4].set_index("player_id")["form_score"]

    pd.testing.assert_series_equal(view_of_gw4, at_gw2, check_names=False)
    # and it must NOT equal what GW4 itself knew
    at_gw4 = scored[scored["gw"] == 4].set_index("player_id")["form_score"]
    assert not np.allclose(view_of_gw4.to_numpy(), at_gw4.to_numpy())


def test_attach_form_without_a_lookahead_returns_none():
    scored, forward = attach_form(_season(), None)
    assert forward is None
    assert "form_score" in scored.columns


def test_a_player_blanking_in_the_decision_week_falls_back_to_price():
    """A blank gameweek leaves a player with no row at all that week, so there is no form to
    carry forward. He must still be scored on something knowable rather than dropped or left
    null — otherwise a blank silently removes him from the baseline's candidate pool."""
    season = _season()
    season = season[~((season["gw"] == 2) & (season["player_id"] == 2))]
    lookahead = {2: {2: season[season["gw"] == 2], 4: season[season["gw"] == 4]}}

    _, forward = attach_form(season, lookahead)
    blanked = forward[2][4].set_index("player_id").loc[2]
    assert blanked["form_score"] == pytest.approx(blanked["price"] * FORM_PRICE_SCALE)
