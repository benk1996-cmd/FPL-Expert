"""Simulating the field, so rank can be measured rather than only total points.

The field model is unusual in having an exact thing to be right about. Every FPL manager owns
fifteen players, so ownership counts pin the average manager's squad points precisely, with no
modelling in between. Most of these tests are about not breaking that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.backtest.field_sim import (
    SQUAD_QUOTA,
    STARTING_XI,
    _inclusion_targets,
    _pick_starting_xi,
    attach_ownership,
    balanced_inclusion,
    pareto_sample,
    rank_metrics,
    simulate_field_season,
)

# Roughly the real shape of an FPL player pool. Size matters here and is not padding: a
# manager owning 5 of 14 defenders has almost no room to differentiate, so skill makes every
# simulated manager converge on the same names and NARROWS the field instead of spreading it.
# Measured on a 14-defender pool the spread fell from 27 to 5; at 60 it rose from 27 to 46.
# Real FPL carries around 200 defenders, so only the larger pool tests anything real.
POOL = (("GK", 20), ("DEF", 60), ("MID", 60), ("FWD", 30))


def _forecasts(n_gws=4, seed=0, drift=False, pool=POOL):
    """A season where ownership spans two orders of magnitude, as it really does."""
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for position, count in pool:
        for i in range(count):
            share = (count - i) / count
            base = 10 ** (1 + 3 * share)
            for gw in range(1, n_gws + 1):
                # With `drift`, ownership migrates over the season the way it really does,
                # which is what the field is supposed to follow.
                shift = (gw / n_gws) if drift and i % 2 == 0 else 1.0
                rows.append({
                    "gw": gw, "player_id": pid, "position": position,
                    "selected": float(base * shift),
                    "expected_points": 0.5 + 5.0 * share,
                    "actual_points": float(rng.poisson(0.5 + 5.0 * share)),
                })
            pid += 1
    return pd.DataFrame(rows)


# --- sampling -------------------------------------------------------------


def test_pareto_sampling_hits_the_inclusion_probabilities_it_is_given():
    """The property Plackett-Luce lacks, and the reason this sampler replaced it. Weighted
    sampling without replacement compresses heavily-weighted items below their share — in FPL
    those are the most-owned players, who are also the highest scorers, so the field quietly
    lost the players that matter most.

    Pareto sampling is close but not exact. Measured here: max error 0.012 on a moderate
    target and 0.032 on an extreme one, against a Plackett-Luce error large enough to cost the
    field 3% of its season points. The aggregate effect is checked properly by the anchor test
    below, which lands within 0.4% on real seasons.
    """
    for target, tolerance in (
        (np.array([0.7, 0.5, 0.4, 0.25, 0.15]), 0.02),
        (np.array([0.9, 0.6, 0.3, 0.15, 0.05]), 0.04),
    ):
        uniforms = np.random.default_rng(0).random((40_000, len(target)))
        picks = pareto_sample(target, 2, uniforms)

        realised = np.zeros(len(target))
        for column in range(picks.shape[1]):
            realised += np.bincount(picks[:, column], minlength=len(target))
        realised /= len(uniforms)
        assert np.allclose(realised, target, atol=tolerance)


def test_sampling_never_picks_the_same_player_twice():
    uniforms = np.random.default_rng(0).random((500, 5))
    picks = pareto_sample(np.array([0.9, 0.6, 0.6, 0.3, 0.1]), 3, uniforms)
    assert picks.shape == (500, 3)
    assert all(len(set(row)) == 3 for row in picks)


def test_fixed_uniforms_keep_the_same_squad_when_ownership_does_not_move():
    """Persistence: a manager is the same manager in May as in August."""
    target = np.array([0.8, 0.5, 0.4, 0.2, 0.1])
    uniforms = np.random.default_rng(1).random((300, 5))
    first = pareto_sample(target, 2, uniforms)
    second = pareto_sample(target, 2, uniforms)
    assert (np.sort(first, axis=1) == np.sort(second, axis=1)).all()


def test_squads_follow_ownership_when_it_moves():
    """...and transfers: the same manager, holding different players, because the crowd moved.
    Neither behaviour is scripted — both fall out of fixed uniforms against moving targets."""
    uniforms = np.random.default_rng(2).random((2000, 5))
    early = pareto_sample(np.array([0.9, 0.6, 0.3, 0.15, 0.05]), 2, uniforms)
    late = pareto_sample(np.array([0.05, 0.15, 0.3, 0.6, 0.9]), 2, uniforms)
    changed = [set(a) != set(b) for a, b in zip(early, late, strict=True)]
    assert np.mean(changed) > 0.5


def test_inclusion_targets_sum_to_the_quota():
    """Feasibility for a fixed-size sample, and it is not imposed — ownership fractions within
    a position really do sum to that position's quota, because every manager owns that many.

    No individual target can exceed 1 either, and that is a property of the data rather than
    of the clip: exceeding it would take a player holding more than `1/quota` of all ownership
    in his position, which is a player owned by over 100% of managers.
    """
    counts = np.array([9e6, 6e6, 3e6, 1.5e6, 5e5])
    targets = _inclusion_targets(counts, 2)
    assert targets.sum() == pytest.approx(2.0)
    assert (targets < 1.0).all()


def test_empty_ownership_falls_back_to_a_uniform_target():
    assert _inclusion_targets(np.zeros(4), 2).sum() == pytest.approx(2.0)


# --- skill ----------------------------------------------------------------


def test_balancing_keeps_squads_legal_and_the_field_honest():
    """Both constraints at once. Enforcing them in sequence was measured to cost 12% of the
    field's points, because the row normalisation that keeps squads legal is itself correlated
    with the skill tilt and puts back the bias the tilt correction removed.

    Sized like a real position block. The iteration count is tuned for a pool of a couple of
    hundred players, where it converges below 1e-6; a five-player pool is a far harder
    Sinkhorn problem than anything this will ever see and would only test the tuning.
    """
    rng = np.random.default_rng(0)
    n_players = 200
    inclusion = np.sort(rng.random(n_players))[::-1]
    inclusion = np.clip(inclusion / inclusion.sum() * 5, 1e-6, 0.999)
    quality = np.clip(rng.normal(0, 1, n_players), -3, 3)
    skill = rng.normal(0, 0.6, 5000)
    balanced = balanced_inclusion(inclusion, np.exp(skill[:, None] * quality[None, :]), 5)

    assert np.allclose(balanced.sum(axis=1), 5.0, atol=1e-9)      # every squad legal, exactly
    assert np.allclose(balanced.mean(axis=0), inclusion, atol=1e-4)  # field owns what it owned


def test_a_skilled_manager_cannot_own_a_player_more_than_certainly():
    """Water-filling, not clipping. A strong tilt pushes a premium above probability one, and
    the excess has to be redistributed across the rest of that manager's squad — discarding it
    leaves him short of a full fifteen. Measured before the fix: rows summing to 1.16 of 2."""
    inclusion = np.array([0.8, 0.5, 0.4, 0.2, 0.1])
    quality = np.array([3.0, 1.0, 0.0, -1.0, -3.0])
    tilt = np.exp(np.array([2.5])[:, None] * quality[None, :])     # one very skilled manager
    balanced = balanced_inclusion(inclusion, tilt, 2)

    assert (balanced <= 1.0).all()
    assert balanced.sum(axis=1)[0] == pytest.approx(2.0)


def test_skill_spreads_the_field_without_moving_its_centre():
    """What skill is for. If it shifted the centre it would break the ownership anchor; the
    whole point is that it separates good managers from bad ones around that anchor.

    Only the spread and the centre are asserted, not the upper tail. In this fixture ownership
    tracks quality perfectly, so a skilled manager is already near the best squad ownership
    allows and has nowhere to climb — the extra dispersion appears almost entirely below the
    median. Real ownership is far looser, and there the top end does rise: measured on
    2024-25, p99 went from 2323 to 2497 when skill was switched on, with the median steady at
    ~2055. That is recorded against `SKILL_DISPERSION` rather than asserted here, because it
    is a property of real ownership data and not of the mechanism.
    """
    forecasts = _forecasts(n_gws=20, drift=True)
    flat = simulate_field_season(forecasts, 4000, seed=1, skill_dispersion=0.0)
    varied = simulate_field_season(forecasts, 4000, seed=1, skill_dispersion=0.8)

    assert varied.std() > 1.3 * flat.std()
    assert np.median(varied) == pytest.approx(np.median(flat), rel=0.02)
    assert abs(np.mean(varied) - np.mean(flat)) < 0.05 * np.mean(flat)


# --- the anchor -----------------------------------------------------------


def test_simulated_squads_hold_the_points_ownership_says_they_should():
    """The calibration that makes this model checkable rather than merely plausible.

        managers       = sum(selected) / 15
        mean squad pts = sum(selected * points) / managers

    Both exact. Measured on real seasons the simulator lands within 0.4% of it.
    """
    forecasts = _forecasts(n_gws=6, drift=True)
    per_gw = forecasts.groupby("gw").apply(
        lambda d: (d["selected"] * d["actual_points"]).sum()
        / (d["selected"].sum() / 15),
        include_groups=False,
    )
    anchor = float(per_gw.sum())

    squads = simulate_field_season(
        forecasts, 6000, seed=3, skill_dispersion=0.0, return_squad_points=True
    )
    assert squads.mean() == pytest.approx(anchor, rel=0.03)


def test_the_anchor_survives_skill_being_switched_on():
    forecasts = _forecasts(n_gws=6, drift=True)
    flat = simulate_field_season(
        forecasts, 6000, seed=3, skill_dispersion=0.0, return_squad_points=True
    )
    varied = simulate_field_season(
        forecasts, 6000, seed=3, skill_dispersion=0.6, return_squad_points=True
    )
    assert varied.mean() == pytest.approx(flat.mean(), rel=0.03)


# --- squads and the XI ----------------------------------------------------


def test_the_starting_xi_is_eleven_and_legal():
    rng = np.random.default_rng(0)
    values = {p: rng.random((400, k)) for p, k in SQUAD_QUOTA.items()}
    starters = _pick_starting_xi(values)

    fielded = sum(mask.sum(axis=1) for mask in starters.values())
    assert (fielded == STARTING_XI).all()
    assert (starters["GK"].sum(axis=1) == 1).all()
    assert (starters["DEF"].sum(axis=1) >= 3).all()
    assert (starters["MID"].sum(axis=1) >= 2).all()
    assert (starters["FWD"].sum(axis=1) >= 1).all()


def test_the_best_players_are_the_ones_fielded():
    """A manager benches his worst. Choosing the XI on ownership instead was the largest single
    reason the old field scored so low — ownership moves far too slowly to notice that someone
    is injured this week, so simulated managers kept fielding players who were not playing."""
    values = {p: np.tile(np.arange(k, dtype=float), (50, 1)) for p, k in SQUAD_QUOTA.items()}
    starters = _pick_starting_xi(values)
    # Within each position the lowest-valued held players must be the benched ones.
    for position, mask in starters.items():
        started = values[position][mask.reshape(50, -1)].reshape(50, -1)
        benched_count = SQUAD_QUOTA[position] - mask.sum(axis=1)[0]
        if benched_count:
            assert started.min() >= benched_count - 1e-9


def test_the_field_plays_chips_and_scores_more_for_it():
    """Rival managers used to play no chips at all while we played three, which flattered our
    rank by roughly the value of a chip season."""
    forecasts = _forecasts(n_gws=24)
    without = simulate_field_season(forecasts, 3000, seed=1, play_chips=False)
    with_chips = simulate_field_season(forecasts, 3000, seed=1, play_chips=True)
    assert with_chips.mean() > without.mean()


def test_each_chip_is_played_once_per_half_season():
    """Two of each per season, one per window, exactly as the rules allow."""
    from fpl_expert.backtest.field_sim import build_field

    trace = build_field(_forecasts(n_gws=38), 500, seed=2)
    for chip in (trace.bench_boost, trace.triple_captain):
        first = chip[[i for i, gw in enumerate(trace.gameweeks) if gw <= 19]].sum(axis=0)
        second = chip[[i for i, gw in enumerate(trace.gameweeks) if gw >= 20]].sum(axis=0)
        assert (first == 1).all()
        assert (second == 1).all()


def test_chips_do_not_disturb_the_ownership_anchor():
    """Chips change what a manager SCORES, not who he owns, so the composition calibration
    must be untouched by them."""
    forecasts = _forecasts(n_gws=12, drift=True)
    plain = simulate_field_season(
        forecasts, 3000, seed=3, play_chips=False, return_squad_points=True
    )
    chipped = simulate_field_season(
        forecasts, 3000, seed=3, play_chips=True, return_squad_points=True
    )
    assert chipped.mean() == pytest.approx(plain.mean(), rel=1e-9)


def test_repair_brings_an_over_budget_squad_under_the_limit():
    """The failure that made a first version stall: dropping the priciest player is often not
    enough, headroom comes out NEGATIVE, and nothing fits. Falling back to the cheapest legal
    alternative guarantees the squad gets cheaper each round so successive rounds converge."""
    from fpl_expert.backtest.field_sim import repair_squads

    positions = np.array(["MID"] * 6)
    price = np.array([12.0, 11.0, 10.0, 5.0, 4.5, 4.0])
    ownership = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    club = np.arange(6)

    repaired = repair_squads(
        np.array([[0, 1, 2]]), np.array(["MID"] * 3), positions, ownership, price, club,
        budget=20.0, max_per_club=3, rounds=8,
    )
    assert price[repaired].sum() <= 20.0


def test_repair_breaks_up_a_club_that_is_over_represented():
    from fpl_expert.backtest.field_sim import repair_squads

    positions = np.array(["MID"] * 6)
    price = np.array([12.0, 11.0, 10.0, 5.0, 4.5, 4.0])
    ownership = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
    club = np.array([0, 0, 0, 0, 1, 2])

    repaired = repair_squads(
        np.array([[0, 1, 2]]), np.array(["MID"] * 3), positions, ownership, price, club,
        budget=99.0, max_per_club=2, rounds=8,
    )
    counts = np.bincount(club[repaired][0], minlength=3)
    assert counts.max() <= 2


def test_repair_keeps_the_squad_shape():
    """Replacements come from the same column's position, so 2/5/5/3 survives — swapping a
    defender for a cheap midfielder would fix the budget and break the squad."""
    from fpl_expert.backtest.field_sim import repair_squads

    positions = np.array(["DEF", "DEF", "MID", "MID"])
    price = np.array([9.0, 4.0, 9.0, 4.0])
    ownership = np.array([90.0, 40.0, 80.0, 30.0])
    club = np.arange(4)

    repaired = repair_squads(
        np.array([[0, 2]]), np.array(["DEF", "MID"]), positions, ownership, price, club,
        budget=9.0, max_per_club=3, rounds=6,
    )
    assert positions[repaired[0, 0]] == "DEF"
    assert positions[repaired[0, 1]] == "MID"


def test_field_has_a_spread_not_a_spike():
    field = simulate_field_season(_forecasts(), 3000, seed=1)
    assert field.std() > 0
    assert np.percentile(field, 99) > np.percentile(field, 50)


def test_a_season_beats_a_single_gameweek():
    short = simulate_field_season(_forecasts(n_gws=2), 2000, seed=1)
    long = simulate_field_season(_forecasts(n_gws=8), 2000, seed=1)
    assert long.mean() > short.mean()


# --- metrics --------------------------------------------------------------


def test_rank_metrics_translate_percentile_to_a_position():
    field = np.arange(1000, dtype=float)
    metrics = rank_metrics(990.0, field)
    assert metrics["percentile"] == pytest.approx(0.99, abs=0.01)
    assert metrics["rank_in_11m"] < 150_000


def test_beating_the_field_top_is_reported():
    field = np.arange(1000, dtype=float)
    assert rank_metrics(999.0, field)["beats_p99"]
    assert not rank_metrics(500.0, field)["beats_p99"]


def test_no_rank_is_reported_when_we_beat_the_entire_field():
    """Beating every simulated manager does not put us first — it says the answer is finer
    than this field can resolve. With 20,000 rivals one of them stands for 550 real managers,
    so `rank_better_than` is all that can honestly be said."""
    field = np.arange(1000, dtype=float)
    metrics = rank_metrics(5000.0, field)
    assert metrics["saturated"]
    assert metrics["rank_in_11m"] is None
    assert metrics["rank_better_than"] == 11_000


def test_a_rank_is_still_reported_when_the_field_contains_us():
    field = np.arange(1000, dtype=float)
    metrics = rank_metrics(990.0, field)
    assert not metrics["saturated"]
    assert metrics["rank_in_11m"] is not None
    assert metrics["rank_better_than"] is None


def test_resolution_improves_with_a_bigger_field():
    small = rank_metrics(500.0, np.arange(1000, dtype=float))["resolution"]
    large = rank_metrics(500.0, np.arange(50_000, dtype=float))["resolution"]
    assert large < small


def test_attach_ownership_joins_on_season_gameweek_and_name():
    forecasts = pd.DataFrame({
        "season": ["2025-26"], "gw": [3], "name": ["A Player"], "actual_points": [6],
    })
    history = pd.DataFrame({
        "season": ["2025-26"], "GW": [3], "name": ["A Player"], "selected": [1_000_000],
    })
    joined = attach_ownership(forecasts, history)
    assert joined["selected"].iloc[0] == 1_000_000


def test_attach_ownership_requires_the_column():
    with pytest.raises(KeyError, match="selected"):
        attach_ownership(pd.DataFrame(), pd.DataFrame({"season": [], "GW": [], "name": []}))
