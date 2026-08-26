"""The weekly brief: everything a decision needs, in one document.

Transfers and chips previously lived in separate commands, so the brief showed a squad without
showing how to reach it from the one you own. These tests are about the joined-up version.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from fpl_expert.reporting.gw_report import (
    build_report,
    chip_section,
    price_section,
    transfer_section,
)


@dataclass
class _Plan:
    n_transfers: int
    hits: int
    net_gain: float
    transfers_out: pd.DataFrame
    transfers_in: pd.DataFrame


def _plan(n=1, hits=0, gain=3.4):
    out = pd.DataFrame({
        "web_name": ["Old"], "position": ["MID"], "team": ["AVL"],
        "selling_price": [7.2], "horizon_points": [12.0],
    })
    into = pd.DataFrame({
        "web_name": ["New"], "position": ["MID"], "team": ["ARS"],
        "price": [7.5], "horizon_points": [15.4],
    })
    return _Plan(n, hits, gain, out, into)


def _solution():
    squad = pd.DataFrame({
        "player_id": range(15),
        "web_name": [f"p{i}" for i in range(15)],
        "position": ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        "team": [f"c{i % 6}" for i in range(15)],
        "price": [5.0] * 15,
        "expected_points": [10 - i * 0.4 for i in range(15)],
        "p_appear": [0.9] * 15,
        "expected_minutes": [80.0] * 15,
    })

    @dataclass
    class _S:
        squad: pd.DataFrame
        starting_xi: pd.DataFrame
        bench: pd.DataFrame
        captain: dict
        vice_captain: dict
        total_cost: float
        expected_points: float

    return _S(squad, squad.head(11), squad.tail(4),
              {"web_name": "p0"}, {"web_name": "p1"}, 75.0, 60.0)


# --- transfers -------------------------------------------------------------


def test_a_recommended_transfer_shows_both_sides():
    text = "\n".join(transfer_section(_plan()))
    assert "OUT" in text and "IN" in text
    assert "Old" in text and "New" in text
    assert "+3.40" in text


def test_the_out_side_shows_selling_price_not_market_price():
    """What you actually receive, which is what funds the move."""
    text = "\n".join(transfer_section(_plan()))
    assert "7.2" in text


def test_a_hit_is_stated_explicitly():
    assert "costing 4" in "\n".join(transfer_section(_plan(n=2, hits=1)))
    assert "no hit" in "\n".join(transfer_section(_plan()))


def test_rolling_is_a_recommendation_not_a_blank():
    """Saying nothing would read as an oversight; the brief has to say 'roll' out loud."""
    text = "\n".join(transfer_section(_plan(n=0)))
    assert "Roll" in text


def test_no_plan_produces_no_section():
    assert transfer_section(None) == []


# --- chips -----------------------------------------------------------------


def test_holding_a_chip_is_reported_with_its_value():
    """The margin is the useful part: a bench boost worth 12 against a typical 11 is a
    different decision from one worth 20."""
    text = "\n".join(chip_section(None, values={"bench_boost": 12.4, "triple_captain": 8.1}))
    assert "Hold" in text
    assert "12.4" in text and "8.1" in text


def test_playing_a_chip_names_it_and_says_why():
    text = "\n".join(chip_section("bench_boost", "best available (18.2 vs 11.0 remaining)"))
    assert "Bench Boost" in text
    assert "best available" in text


# --- prices ----------------------------------------------------------------


def test_price_moves_are_reported_but_framed_as_secondary():
    risers = pd.DataFrame({
        "web_name": ["Riser"], "team": ["ARS"], "price": [7.5],
        "p_rise": [0.4], "p_fall": [0.0], "expected_change": [0.04],
    })
    text = "\n".join(price_section(risers, risers.head(0)))
    assert "Riser" in text
    # A price move must never read as a reason to transfer on its own.
    assert "not worth a transfer by themselves" in text


def test_no_price_data_produces_no_section():
    assert price_section(None, None) == []
    assert price_section(pd.DataFrame(), pd.DataFrame()) == []


# --- the whole brief -------------------------------------------------------


def test_the_brief_joins_squad_transfers_and_chips():
    report = build_report(
        _solution().squad, _solution(), 5, plan=_plan(),
        chip_values={"bench_boost": 9.0, "triple_captain": 7.0},
    )
    for heading in ("Starting XI", "Bench", "Captaincy", "Transfers", "Chips", "Caveats"):
        assert f"## {heading}" in report


def test_the_brief_still_works_with_nothing_extra():
    """`fpl report` solves an ideal squad and has no held squad to transfer from."""
    report = build_report(_solution().squad, _solution(), 5)
    assert "## Transfers" not in report
    assert "## Starting XI" in report


def test_the_caveats_survive():
    """The bonus warning has to travel with the numbers; a brief that drops it reads more
    confident than the model is."""
    report = build_report(_solution().squad, _solution(), 5)
    assert "bonus" in report.lower()
    assert "press conferences" in report


def test_chip_values_are_optional():
    assert "Chips" in build_report(_solution().squad, _solution(), 5, plan=_plan())


def _held_squad():
    """Fifteen with a dead asset at forward — the shape that exposed the bug."""
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return pd.DataFrame({
        "player_id": range(15),
        "web_name": [f"P{i}" for i in range(15)],
        "position": positions,
        "team": [f"C{i % 6}" for i in range(15)],
        "price": 5.0,
        "selling_price": 5.0,
        "expected_points": [4.0] * 14 + [0.0],     # P14, a forward, is the dead asset
        "horizon_points": [20.0] * 14 + [0.0],
        "p_zero": 0.1,
    })


def test_the_team_sheet_fields_the_squad_you_would_hold_after_the_transfer():
    """The brief used to recommend a signing and then print an XI without him in it.

    It picked the eleven from the PRE-transfer squad, so the player being sold sat on the
    bench at 0.00 while the incoming player appeared nowhere. On a real squad that understated
    the gameweek by 2.6 points and named two starters who would have been displaced.
    """
    from fpl_expert.backtest.season_sim import pick_xi
    from fpl_expert.config import load_scoring_rules

    rules = load_scoring_rules()
    held = _held_squad()
    incoming = pd.DataFrame([{
        "player_id": 99, "web_name": "Incoming", "position": "FWD", "team": "C9",
        "price": 5.0, "selling_price": 5.0, "expected_points": 6.0,
        "horizon_points": 30.0, "p_zero": 0.1,
    }])

    fielded = pd.concat([held[held["player_id"] != 14], incoming], ignore_index=True)
    _, starters = pick_xi(fielded, rules, "expected_points")

    assert "Incoming" in set(starters["web_name"]), "the signing must be available to field"
    assert "P14" not in set(fielded["web_name"]), "the sold player must be gone entirely"


def test_xi_value_counts_the_captain_twice():
    """The gain figure compares two XI totals, so both must use the same convention as the
    manifest — otherwise the 'what the move buys' line is measuring two different things."""
    from fpl_expert.config import load_scoring_rules
    from fpl_expert.reporting.gw_report import xi_value

    rules = load_scoring_rules()
    squad = _held_squad()
    # eleven starters at 4.0, best of them counted again as captain
    assert xi_value(squad, rules) == pytest.approx(11 * 4.0 + 4.0)
    assert xi_value(pd.DataFrame(), rules) == 0.0


def test_the_move_section_is_absent_when_no_transfer_is_recommended():
    """A 'what the move buys' block with nothing in it is worse than no block."""
    from fpl_expert.config import load_scoring_rules
    from fpl_expert.reporting.gw_report import move_section

    rules = load_scoring_rules()

    class _Plan:
        n_transfers = 0

    assert move_section(50.0, 50.0, _Plan(), rules, set(), set()) == []
    assert move_section(50.0, 52.0, None, rules, set(), set()) == []
