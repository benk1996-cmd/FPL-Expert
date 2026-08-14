"""Squad flexibility: positional reach and the cost of pivoting."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.optimise.flexibility import (
    flexibility_score,
    pivot_cost,
    positional_reach,
    reachable_share,
    summarise,
)


def _squad(prices=(5.0, 15.0, 6.0, 4.5)):
    gk, fwd, mid, defender = prices
    return pd.DataFrame({
        "web_name": ["keeper", "premium", "midder", "back"],
        "position": ["GK", "FWD", "MID", "DEF"],
        "selling_price": [gk, fwd, mid, defender],
        "price": [gk, fwd, mid, defender],
    })


def test_reach_is_the_most_spendable_on_one_replacement():
    reach = positional_reach(_squad(), bank=2.0)
    assert reach["FWD"] == pytest.approx(17.0)     # 15.0 premium + 2.0 bank
    assert reach["DEF"] == pytest.approx(6.5)


def test_holding_a_premium_buys_reach_in_that_position_only():
    """The core asymmetry — and its limit. A premium striker opens the forward market and
    does nothing for defence."""
    premium = positional_reach(_squad(prices=(5.0, 15.0, 6.0, 4.5)))
    budget = positional_reach(_squad(prices=(5.0, 7.0, 6.0, 4.5)))

    assert premium["FWD"] > budget["FWD"]
    assert premium["DEF"] == budget["DEF"]


def test_bank_raises_reach_in_every_position():
    """Cash dominates a premium on flexibility alone: it reaches everywhere. Its drawback is
    that it scores no points while it sits there."""
    without = positional_reach(_squad(), bank=0.0)
    with_bank = positional_reach(_squad(), bank=3.0)
    assert (with_bank - without).round(6).eq(3.0).all()


def test_selling_price_is_preferred_over_market_price():
    """Reach must be computed on what you RECEIVE. Using market price overstates it for any
    player who has risen, and proposes transfers you cannot fund."""
    squad = _squad()
    squad["selling_price"] = squad["price"] - 0.4
    assert positional_reach(squad)["FWD"] == pytest.approx(14.6)


def test_pivot_cost_is_zero_when_the_target_is_reachable():
    target = pd.Series({"position": "FWD", "price": 14.0})
    assert pivot_cost(_squad(), target) == 0.0


def test_pivot_cost_reports_the_funding_gap():
    """A positive gap means a second transfer and a -4 hit, or selling someone you wanted."""
    target = pd.Series({"position": "FWD", "price": 14.0})
    assert pivot_cost(_squad(prices=(5.0, 7.0, 6.0, 4.5)), target) == pytest.approx(7.0)


def test_pivot_cost_is_infinite_without_a_player_in_that_position():
    squad = _squad()[lambda d: d["position"] != "FWD"]
    assert pivot_cost(squad, pd.Series({"position": "FWD", "price": 9.0})) == float("inf")


def test_reachable_share_measures_access_to_the_useful_market():
    """More decision-relevant than raw reach: not how much you could spend, but how much of
    the genuinely useful market it opens."""
    candidates = pd.DataFrame({
        "position": ["FWD"] * 4,
        "price": [6.0, 9.0, 12.0, 15.0],
        "expected_points": [4.0, 5.0, 6.0, 7.0],
    })
    rich = reachable_share(_squad(prices=(5.0, 15.0, 6.0, 4.5)), candidates)
    poor = reachable_share(_squad(prices=(5.0, 6.0, 6.0, 4.5)), candidates)

    assert rich == 1.0
    assert poor < rich


def test_flexibility_score_aggregates_reach():
    assert flexibility_score(_squad(), bank=1.0) == pytest.approx(
        positional_reach(_squad(), bank=1.0).sum()
    )


def test_summarise_names_the_anchor_player():
    table = summarise(_squad(), bank=1.0).set_index("position")
    assert table.loc["FWD", "anchor"] == "premium"
    assert table.loc["FWD", "reach"] == pytest.approx(16.0)
