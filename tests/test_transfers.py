"""Selling prices, squad reconstruction, and transfer recommendations."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.data.my_team import purchase_prices, selling_price_tenths
from fpl_expert.optimise.transfers import horizon_points, recommend_transfers

# --- selling price --------------------------------------------------------


@pytest.mark.parametrize(
    ("purchase", "current", "expected"),
    [
        (70, 75, 72),   # +0.5 profit -> keep half, rounded DOWN: 7.2 not 7.25
        (70, 74, 72),   # +0.4 -> 7.2
        (70, 73, 71),   # +0.3 -> half is 0.15, rounds down to 0.1
        (70, 70, 70),   # unchanged
        (70, 65, 65),   # losses are borne in full
        (70, 71, 70),   # +0.1 -> half rounds down to nothing
    ],
)
def test_selling_price_keeps_half_the_profit_rounded_down(purchase, current, expected):
    """Getting this wrong silently inflates the budget and produces transfers you cannot
    actually afford."""
    assert selling_price_tenths(purchase, current) == expected


# --- purchase price reconstruction ---------------------------------------


def test_purchase_prices_from_initial_squad_and_transfers():
    initial = pd.DataFrame({"element": [1, 2, 3], "purchase_price": [50, 60, 70]})
    transfers = pd.DataFrame({
        "element_in": [4], "element_out": [3],
        "element_in_cost": [85], "element_out_cost": [72], "event": [5],
    })
    paid = purchase_prices(initial, transfers, current_squad={1, 2, 4})

    assert paid == {1: 50, 2: 60, 4: 85}
    assert 3 not in paid                      # sold


def test_repurchase_uses_the_latest_price_paid():
    """A player bought, sold, then bought again must carry the SECOND purchase price."""
    initial = pd.DataFrame({"element": [1], "purchase_price": [50]})
    transfers = pd.DataFrame({
        "element_in": [2, 1], "element_out": [1, 2],
        "element_in_cost": [60, 55], "element_out_cost": [50, 61], "event": [3, 7],
    })
    paid = purchase_prices(initial, transfers, current_squad={1})
    assert paid == {1: 55}


def test_purchase_prices_with_no_transfers():
    initial = pd.DataFrame({"element": [1, 2], "purchase_price": [50, 60]})
    paid = purchase_prices(initial, pd.DataFrame(), current_squad={1, 2})
    assert paid == {1: 50, 2: 60}


# --- transfer recommendation ---------------------------------------------


def _squad():
    rows = []
    pid = 0
    for position, count in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for i in range(count):
            rows.append({
                "player_id": pid, "web_name": f"held{pid}", "position": position,
                "team": f"club{pid % 6}", "selling_price": 5.0,
                "horizon_points": 10.0 + i,
            })
            pid += 1
    return pd.DataFrame(rows)


def _candidates(bonus=0.0, price=5.0):
    rows = []
    for pid, position in enumerate(["GK", "DEF", "MID", "FWD"], start=100):
        rows.append({
            "player_id": pid, "web_name": f"target{position}", "position": position,
            "team": "club7", "price": price, "horizon_points": 10.0 + bonus,
        })
    return pd.DataFrame(rows)


def test_no_transfer_when_nothing_clears_its_cost():
    """Rolling the transfer is a real option and often the right one."""
    plan = recommend_transfers(
        _squad(), _candidates(bonus=-5.0), bank=0.0, free_transfers=1
    )
    assert plan.n_transfers == 0
    assert "ROLL" in plan.summary()


def test_takes_a_clearly_profitable_free_transfer():
    plan = recommend_transfers(
        _squad(), _candidates(bonus=8.0), bank=0.0, free_transfers=1, max_transfers=1
    )
    assert plan.n_transfers == 1
    assert plan.hits == 0
    assert plan.net_gain > 0


def test_takes_multiple_hits_when_each_clears_its_cost():
    """A hit is worth taking whenever the gain exceeds 4 points, and they compound — the
    optimiser should not stop at one just because only one is free."""
    plan = recommend_transfers(
        _squad(), _candidates(bonus=8.0), bank=0.0, free_transfers=1, max_transfers=3
    )
    assert plan.n_transfers > 1
    assert plan.hits == plan.n_transfers - 1
    assert plan.net_gain > 8.0        # better than the single free transfer alone


def test_a_hit_must_clear_four_points_to_be_worth_taking():
    """A second transfer costs 4 points. A 3-point gain is not worth it; a 12-point gain is."""
    marginal = recommend_transfers(
        _squad(), _candidates(bonus=3.0), bank=0.0, free_transfers=0, max_transfers=2
    )
    assert marginal.n_transfers == 0

    worthwhile = recommend_transfers(
        _squad(), _candidates(bonus=12.0), bank=0.0, free_transfers=0, max_transfers=2
    )
    assert worthwhile.n_transfers >= 1
    assert worthwhile.net_gain > 0


def test_budget_uses_selling_price_not_market_price():
    """You receive the SELLING price, which is lower than market after a price rise. An
    optimiser using market prices proposes transfers you cannot fund."""
    squad = _squad()
    squad["selling_price"] = 4.0
    expensive = _candidates(bonus=20.0, price=9.0)

    plan = recommend_transfers(squad, expensive, bank=0.0, free_transfers=1)
    assert plan.n_transfers == 0          # 4.0 out + 0 bank cannot buy 9.0


def test_positional_quota_is_preserved():
    plan = recommend_transfers(_squad(), _candidates(bonus=8.0), bank=5.0, free_transfers=3,
                               max_transfers=3)
    if plan.n_transfers:
        assert (plan.transfers_out["position"].value_counts().sort_index()
                .equals(plan.transfers_in["position"].value_counts().sort_index()))


def test_club_limit_holds_after_the_transfer():
    """The constraint applies to the POST-transfer squad, which is what makes it awkward."""
    squad = _squad()
    squad.loc[squad.index[:3], "team"] = "club7"      # already 3 from club7
    plan = recommend_transfers(squad, _candidates(bonus=15.0), bank=10.0, free_transfers=2,
                               max_transfers=2)
    after = pd.concat([
        squad[~squad["player_id"].isin(plan.transfers_out["player_id"])],
        plan.transfers_in,
    ])
    assert (after["team"] == "club7").sum() <= 3


def test_bank_after_is_reported():
    plan = recommend_transfers(_squad(), _candidates(bonus=8.0, price=4.0),
                               bank=1.0, free_transfers=1, max_transfers=1)
    assert plan.n_transfers == 1
    assert plan.bank_after == pytest.approx(1.0 + 5.0 - 4.0)


# --- horizon ---------------------------------------------------------------


def test_horizon_discounts_later_gameweeks():
    """A transfer is a durable change, so it is judged over a horizon — but later weeks are
    discounted because forecasts decay and squads churn.

    `captaincy_weight=0` isolates the discounting; the armband premium is covered in
    test_captaincy.py.
    """
    frames = {
        1: pd.DataFrame({"player_id": [1], "expected_points": [10.0]}),
        2: pd.DataFrame({"player_id": [1], "expected_points": [10.0]}),
    }
    out = horizon_points(frames, decay=0.5, captaincy_weight=0.0)
    assert out["horizon_points"].iloc[0] == pytest.approx(10.0 + 5.0)


def test_horizon_sums_across_players():
    frames = {
        1: pd.DataFrame({"player_id": [1, 2], "expected_points": [4.0, 6.0]}),
        2: pd.DataFrame({"player_id": [1, 2], "expected_points": [4.0, 6.0]}),
    }
    out = horizon_points(frames, decay=1.0, captaincy_weight=0.0).set_index("player_id")
    assert out.loc[1, "horizon_points"] == pytest.approx(8.0)
    assert out.loc[2, "horizon_points"] == pytest.approx(12.0)


def test_horizon_includes_the_captaincy_premium_by_default():
    """A durable captain is worth more than his raw points, and the default valuation must
    say so — otherwise every transfer decision underprices reliable armband options."""
    frames = {
        1: pd.DataFrame({"player_id": [1, 2], "expected_points": [10.0, 3.0]}),
        2: pd.DataFrame({"player_id": [1, 2], "expected_points": [10.0, 3.0]}),
    }
    out = horizon_points(frames, decay=1.0).set_index("player_id")
    assert out.loc[1, "horizon_points"] > 20.0
    assert out.loc[1, "captaincy_uplift"] > out.loc[2, "captaincy_uplift"]
