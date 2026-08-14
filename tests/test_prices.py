"""Price changes: ordered outcomes, no lookahead, and a baseline that is hard to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.prices import (
    FALL,
    PRICE_STEP,
    RISE,
    SAME,
    NoChangeBaseline,
    PriceModel,
    build_features,
    team_value_path,
)


def _history(n_players=60, n_gws=12, seed=0):
    """Synthetic archive where price follows transfer pressure with a one-gameweek lag."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(n_players):
        value = 40 + (player % 10) * 8
        owners = 10_000 * (1 + player % 12)
        for gw in range(1, n_gws + 1):
            pressure = rng.normal(0, 0.12)
            balance = pressure * owners
            rows.append({
                "season": "2024-25", "GW": gw, "name": f"p{player}", "value": value,
                "selected": owners, "transfers_balance": balance,
                "transfers_in": max(balance, 0), "transfers_out": max(-balance, 0),
            })
            # The move lands NEXT gameweek, which is what the model has to learn.
            value += 1 if pressure > 0.15 else (-1 if pressure < -0.15 else 0)
    return pd.DataFrame(rows)


def test_features_never_see_the_change_they_predict():
    """The target is next gameweek's move. Shifting it the other way would produce a model
    that explains price changes from the transfers those changes caused."""
    frame = build_features(_history())
    one = frame[frame["name"] == "p0"].sort_values("GW")
    assert (one["delta"] == one["next_value"] - one["value"]).all()
    # The last gameweek has no successor and must be dropped, not filled.
    assert one["GW"].max() < 12


def test_targets_are_the_three_ordered_outcomes():
    frame = build_features(_history())
    assert set(frame["target"]) <= {FALL, SAME, RISE}
    rising = frame[frame["delta"] > 0]
    assert (rising["target"] == RISE).all()


def test_net_fraction_is_relative_to_the_ownership_base():
    """FPL moves prices on net transfers relative to owners: the same absolute flow moves a
    fringe player and barely touches a template one."""
    frame = build_features(_history())
    assert "net_fraction" in frame
    big = frame[frame["selected"] == frame["selected"].max()]
    small = frame[frame["selected"] == frame["selected"].min()]
    assert big["net_fraction"].abs().mean() < small["net_fraction"].abs().mean() * 5


# --- the model -------------------------------------------------------------


def test_probabilities_are_a_proper_distribution():
    frame = build_features(_history())
    probabilities = PriceModel().fit(frame).predict_proba(frame)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= -1e-9).all()


def test_buying_pressure_raises_the_chance_of_a_rise_and_lowers_a_fall():
    """The ordering an ordered model exists to guarantee. Two independent classifiers could
    make a player simultaneously likely to rise and likely to fall."""
    frame = build_features(_history())
    model = PriceModel().fit(frame)

    heavy_buy = frame.nlargest(200, "net_fraction")
    heavy_sell = frame.nsmallest(200, "net_fraction")
    bought, sold = model.predict_proba(heavy_buy), model.predict_proba(heavy_sell)

    assert bought[:, RISE].mean() > sold[:, RISE].mean()
    assert bought[:, FALL].mean() < sold[:, FALL].mean()


def test_cutpoints_cannot_cross():
    """A crossed pair would invert the ordering and delete the middle category. The fit
    exponentiates the gap so it stays positive whatever the optimiser does."""
    model = PriceModel().fit(build_features(_history()))
    assert model.cutpoints[0] < model.cutpoints[1]
    probabilities = model.predict_proba(build_features(_history()))
    assert probabilities[:, SAME].min() >= 0.0


def test_expected_change_is_signed_and_bounded_by_one_increment():
    frame = build_features(_history())
    change = PriceModel().fit(frame).expected_change(frame)
    assert np.abs(change).max() <= PRICE_STEP + 1e-9
    assert change.min() < 0 < change.max()


def test_the_model_beats_the_base_rate_baseline():
    """92.5% of player-gameweeks see no change, so a model can be accurate and useless. Log
    loss is the metric that notices."""
    from fpl_expert.backtest.metrics import log_loss

    frame = build_features(_history(n_players=120, n_gws=16))
    train = frame[frame["GW"] <= 10]
    test = frame[frame["GW"] > 10]

    model, baseline = PriceModel().fit(train), NoChangeBaseline().fit(train)
    y = test["target"].to_numpy()
    assert log_loss(y, model.predict_proba(test)) < log_loss(y, baseline.predict_proba(test))


def test_predicting_before_fitting_is_an_error():
    with pytest.raises(RuntimeError):
        PriceModel().predict_proba(build_features(_history()))


# --- team value ------------------------------------------------------------


def test_team_value_path_accumulates_over_the_horizon():
    frame = build_features(_history())
    model = PriceModel().fit(frame)
    by_gw = {
        gw: block.assign(player_id=block["name"].str.removeprefix("p").astype(int))
        for gw, block in frame.groupby("GW")
    }
    path = team_value_path(set(range(10)), by_gw, model)

    assert len(path) == len(by_gw)
    assert path["cumulative"].iloc[-1] == pytest.approx(path["expected_change"].sum())


def test_team_value_path_ignores_players_not_held():
    frame = build_features(_history())
    model = PriceModel().fit(frame)
    by_gw = {
        gw: block.assign(player_id=block["name"].str.removeprefix("p").astype(int))
        for gw, block in frame.groupby("GW")
    }
    assert team_value_path(set(), by_gw, model).empty
