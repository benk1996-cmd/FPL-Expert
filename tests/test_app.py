"""The Streamlit front end, exercised headlessly.

`AppTest` runs the script the way Streamlit does and surfaces exceptions, which an HTTP 200 on
the root does not — the page shell loads before the script has run a line.

The bundle is built in a tmp_path so these tests never depend on `fpl publish` having been run,
and never read the developer's real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class _Solution:
    """The parts of `SquadSolution` the bundle writer reads."""

    def __init__(self, squad, starting_xi):
        self.squad = squad
        self.starting_xi = starting_xi
        self.bench = squad[~squad["player_id"].isin(starting_xi["player_id"])]
        self.captain = starting_xi.iloc[0]
        self.vice_captain = starting_xi.iloc[1]
        self.total_cost = float(squad["price"].sum())
        self.expected_points = float(starting_xi["expected_points"].sum())


def _players(n=40):
    rows = []
    for i in range(n):
        position = ["GK", "DEF", "MID", "FWD"][i % 4]
        rows.append({
            "player_id": i, "web_name": f"P{i}", "name": f"Player {i}",
            "position": position, "team": f"Club{i % 6}", "price": 4.0 + (i % 8) * 0.5,
            "expected_points": 2.0 + (i % 7) * 0.4,
            "horizon_points": 12.0 + (i % 7) * 2.0,
            "points_variance": 4.0, "p_appear": 0.9, "p_long": 0.5 + (i % 5) * 0.1,
            "expected_minutes": 70.0,
            "pts_appearance": 1.5, "pts_goals": 0.8, "pts_assists": 0.4,
            "pts_clean_sheet": 0.3, "pts_bonus": 0.2, "pts_defcon": 0.1,
            "pts_saves": 0.0, "pts_cards": -0.1,
            "expected_goals": 0.2, "expected_assists": 0.1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def bundle(tmp_path):
    from fpl_expert.serving import write_bundle

    players = _players()
    squad = players.head(15)
    solution = _Solution(squad, squad.head(11))
    fixtures = pd.DataFrame([
        {"team": f"Club{c}", "gw": gw, "opponent": f"Club{(c + 1) % 6}",
         "is_home": gw % 2 == 0, "kickoff_time": "2026-08-21T17:30:00Z"}
        for c in range(6) for gw in (1, 2, 3)
    ])
    write_bundle(
        tmp_path, gw=1, span=6, players=players, solution=solution,
        brief="# Brief\n\nSomething useful.", fixtures=fixtures,
    )
    return tmp_path


def _run(bundle_dir, monkeypatch):
    """Point the app at a test bundle by patching the module-level BUNDLE path."""
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["_test_bundle"] = str(bundle_dir)
    monkeypatch.setenv("FPL_SERVING_DIR", str(bundle_dir))
    return app.run()


def test_the_app_runs_without_exceptions(bundle, monkeypatch):
    """The guard that matters. A Streamlit script that raises still serves HTTP 200 — the
    error only appears once the client connects and the script actually executes."""
    app = _run(bundle, monkeypatch)
    assert not app.exception, [str(e) for e in app.exception]


def test_the_headline_numbers_are_rendered(bundle, monkeypatch):
    app = _run(bundle, monkeypatch)
    labels = [m.label for m in app.metric]
    assert "Gameweek" in labels
    assert "Expected points" in labels
    assert "Captain" in labels


def test_every_tab_is_present(bundle, monkeypatch):
    app = _run(bundle, monkeypatch)
    rendered = " ".join(str(t) for t in app.tabs) if app.tabs else ""
    for name in ("Squad", "Players", "Fixtures", "Brief"):
        assert name in rendered or any(name in str(m.value) for m in app.markdown)


def test_a_missing_bundle_explains_itself_rather_than_crashing(tmp_path, monkeypatch):
    """A front end whose data has not been published must say so and name the command. An
    empty page would be indistinguishable from a model that recommends nothing."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    app = _run(empty, monkeypatch)

    assert not app.exception
    text = " ".join(str(e.value) for e in app.error) + " ".join(
        str(c.value) for c in app.code
    )
    assert "publish" in text


@pytest.fixture
def two_variant_bundle(tmp_path):
    """A bundle carrying both the standard and minutes-budget views."""
    from fpl_expert.serving import write_bundle

    players = _players()
    squad = players.head(15)
    other = players.tail(15).reset_index(drop=True)
    write_bundle(
        tmp_path, gw=1, span=6, players=players, solution=_Solution(squad, squad.head(11)),
        brief="# Brief", variants={"minutes budget": (players, _Solution(other, other.head(11)))},
    )
    return tmp_path


def test_both_views_are_stored_and_separable(two_variant_bundle):
    from fpl_expert.serving import read_bundle

    loaded = read_bundle(two_variant_bundle)
    assert set(loaded["players"]["variant"]) == {"standard", "minutes budget"}
    assert set(loaded["manifest"]["variants"]) == {"standard", "minutes budget"}
    # each view keeps its OWN squad, or the toggle would show one squad under two labels
    squads = {
        name: set(g[g["in_squad"]]["player_id"])
        for name, g in loaded["players"].groupby("variant")
    }
    assert squads["standard"] != squads["minutes budget"]


def test_the_app_renders_a_two_variant_bundle(two_variant_bundle, monkeypatch):
    app = _run(two_variant_bundle, monkeypatch)
    assert not app.exception, [str(e) for e in app.exception]
    assert app.radio, "no variant switch rendered"
    assert set(app.radio[0].options) == {"standard", "minutes budget"}


def test_a_single_variant_bundle_shows_no_switch(bundle, monkeypatch):
    """One view is not a choice. A radio with a single option is noise."""
    app = _run(bundle, monkeypatch)
    assert not app.exception
    assert not app.radio


def test_the_bundle_carries_no_personal_data(bundle):
    """The bundle is committed and deployed, so it must describe the GAME and not the user.
    Anything entry-specific belongs in `fpl myteam --brief`, which stays local."""
    from fpl_expert.serving import read_bundle

    loaded = read_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    forbidden = {"entry", "entry_id", "manager", "selling_price", "bank", "free_transfers"}

    assert not forbidden & set(loaded["players"].columns)
    assert not forbidden & set(manifest)
