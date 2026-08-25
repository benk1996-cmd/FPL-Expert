"""Hand-entered availability corrections: they must apply, expire, and never fail silently."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from fpl_expert.data.overrides import apply_availability_overrides, load_overrides

TODAY = date(2026, 8, 24)


@pytest.fixture
def players():
    return pd.DataFrame({
        "id": [55, 1, 2],
        "web_name": ["Watkins", "Smith", "Smith"],
        "status": ["a", "a", "a"],
        "chance_of_playing_next_round": [None, None, None],
    })


def test_an_override_changes_status(players):
    out = apply_availability_overrides(
        players, [{"web_name": "Watkins", "status": "u", "reason": "left the club"}],
        today=TODAY,
    )
    assert out.loc[out["web_name"] == "Watkins", "status"].iloc[0] == "u"
    assert players["status"].tolist() == ["a", "a", "a"]      # the input frame is untouched


def test_an_expired_override_is_ignored_and_says_so(players, caplog):
    """The dangerous failure is an override outliving the situation it describes."""
    with caplog.at_level(logging.WARNING):
        out = apply_availability_overrides(
            players,
            [{"web_name": "Watkins", "status": "u", "until": "2026-08-01"}],
            today=TODAY,
        )
    assert out.loc[out["web_name"] == "Watkins", "status"].iloc[0] == "a"
    assert "EXPIRED" in caplog.text


def test_an_override_matching_nobody_warns(players, caplog):
    """A typo that silently does nothing is exactly the class of bug ground rule 7 is about."""
    with caplog.at_level(logging.WARNING):
        apply_availability_overrides(players, [{"web_name": "Watkinz", "status": "u"}], today=TODAY)
    assert "matched no player" in caplog.text


def test_an_ambiguous_name_is_skipped_rather_than_guessed(players, caplog):
    """Two players share `Smith`. Picking one would be worse than applying neither."""
    with caplog.at_level(logging.WARNING):
        out = apply_availability_overrides(
            players, [{"web_name": "Smith", "status": "i"}], today=TODAY
        )
    assert out["status"].tolist() == ["a", "a", "a"]
    assert "matches 2 players" in caplog.text

    by_id = apply_availability_overrides(players, [{"id": 2, "status": "i"}], today=TODAY)
    assert by_id.loc[by_id["id"] == 2, "status"].iloc[0] == "i"
    assert by_id.loc[by_id["id"] == 1, "status"].iloc[0] == "a"


def test_an_unknown_status_is_rejected(players, caplog):
    """FPL's vocabulary, not a free-text field — `apply_availability_gate` matches on it."""
    with caplog.at_level(logging.WARNING):
        out = apply_availability_overrides(
            players, [{"web_name": "Watkins", "status": "gone"}], today=TODAY
        )
    assert out.loc[out["web_name"] == "Watkins", "status"].iloc[0] == "a"
    assert "unknown status" in caplog.text


def test_applying_an_override_always_warns(players, caplog):
    """A fallback that changes the model must say so — this one changes a squad."""
    with caplog.at_level(logging.WARNING):
        apply_availability_overrides(players, [{"id": 55, "status": "u"}], today=TODAY)
    assert "OVERRIDE APPLIED" in caplog.text


def test_no_overrides_is_the_normal_case(players, tmp_path):
    assert load_overrides(tmp_path / "absent.yaml") == []
    assert apply_availability_overrides(players, [], today=TODAY).equals(players)


def test_every_shipped_override_carries_an_expiry_and_a_reason():
    """The shipped file may hold live corrections — this is a single-owner project — but an
    override with no `until` outlives the situation it describes, and one with no `reason`
    cannot be checked by the next reader. Staleness is the whole risk here, so it is guarded
    rather than left to discipline."""
    for entry in load_overrides():
        assert entry.get("until"), f"override without an expiry: {entry}"
        assert entry.get("reason"), f"override without a reason: {entry}"
        assert entry.get("id") or entry.get("web_name"), f"override matches nobody: {entry}"
