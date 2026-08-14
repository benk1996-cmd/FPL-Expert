"""Config and scoring rules load correctly and carry the values models depend on."""

from __future__ import annotations

from fpl_expert.config import load_config, load_scoring_rules


def test_config_loads_with_committed_objective():
    cfg = load_config()
    assert cfg.optimise.objective.target_rank == 10_000
    assert cfg.ownership.overall_league_id == 314
    assert cfg.optimise.horizon_gws >= 1


def test_scoring_rules_are_verified_against_the_live_game():
    rules = load_scoring_rules()
    assert rules["verified"] is True
    assert rules["season"] == "2026/27"


def test_goalkeeper_goals_worth_ten():
    """Verified from the API's own game_config; easy to assume 6 by analogy with defenders."""
    assert load_scoring_rules()["goal"]["GK"] == 10
    assert load_scoring_rules()["goal"]["DEF"] == 6


def test_two_of_every_chip():
    """Eight chips total, one of each per half — not the pre-2024/25 single-set structure."""
    chips = load_scoring_rules()["chips"]
    assert {name: c["count"] for name, c in chips.items()} == {
        "wildcard": 2, "free_hit": 2, "bench_boost": 2, "triple_captain": 2
    }
    # Wildcard and Free Hit cannot be played in GW1; Bench Boost and Triple Captain can.
    assert chips["wildcard"]["windows"][0][0] == 2
    assert chips["bench_boost"]["windows"][0][0] == 1


def test_squad_rules_match_the_api():
    squad = load_scoring_rules()["squad"]
    assert squad["budget"] == 100.0
    assert squad["positions"] == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert squad["max_per_club"] == 3
