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


def _by_gameweek(players, gws=(1, 2, 3)):
    """Per-gameweek forecasts, deliberately differing week to week so a navigator can be seen
    to actually change what is on screen."""
    return {
        gw: players.assign(expected_points=players["expected_points"] + gw)
        for gw in gws
    }


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
        by_gameweek=_by_gameweek(players),
    )
    return tmp_path


@pytest.fixture
def single_gw_bundle(tmp_path):
    """A bundle published before `forecasts.parquet` existed — no navigator, no crash."""
    from fpl_expert.serving import write_bundle

    players = _players()
    squad = players.head(15)
    write_bundle(
        tmp_path, gw=1, span=6, players=players, solution=_Solution(squad, squad.head(11)),
        brief="# Brief",
    )
    return tmp_path


def _run(bundle_dir, monkeypatch):
    """Point the app at a test bundle by patching the module-level BUNDLE path.

    `FPL_ENTRY_CACHE` is redirected for the same reason the bundle is: the My Team tab
    remembers an entry id on disk, and a test must neither read the developer's real one nor
    overwrite it.
    """
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.session_state["_test_bundle"] = str(bundle_dir)
    monkeypatch.setenv("FPL_SERVING_DIR", str(bundle_dir))
    monkeypatch.setenv("FPL_ENTRY_CACHE", str(Path(bundle_dir) / "entry.json"))
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
    for name in ("Squad", "Players", "Fixtures", "Brief", "My Team"):
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


def _arrow(app, glyph):
    """The navigator buttons by label — position would break the moment a tab adds one."""
    return next(b for b in app.button if b.label == glyph)


def test_the_gameweek_navigator_steps_forward_and_back(bundle, monkeypatch):
    """The arrows must change the forecasts on screen, not just the label.

    The fixture makes each gameweek's expected points differ by a known amount, so a
    navigator that moved the caption while still rendering week one would fail here.
    """
    app = _run(bundle, monkeypatch)
    published = next(m for m in app.metric if m.label == "Gameweek")
    start = float(next(m for m in app.metric if m.label == "Expected points").value)
    assert published.value == "1"

    _arrow(app, "▶").click().run()
    assert next(m for m in app.metric if m.label == "Gameweek").value == "2"
    stepped = float(next(m for m in app.metric if m.label == "Expected points").value)
    assert stepped > start                            # GW2 adds +1 per player in the fixture

    _arrow(app, "◀").click().run()
    assert next(m for m in app.metric if m.label == "Gameweek").value == "1"
    assert float(
        next(m for m in app.metric if m.label == "Expected points").value
    ) == pytest.approx(start)


def test_the_navigator_stops_at_both_ends(bundle, monkeypatch):
    """Nothing outside the published window can be reached — there are no forecasts there."""
    app = _run(bundle, monkeypatch)

    assert _arrow(app, "◀").disabled                  # already at the first gameweek
    assert not _arrow(app, "▶").disabled

    _arrow(app, "▶").click().run()
    _arrow(app, "▶").click().run()                    # now at the last served gameweek

    assert app.session_state["view_gw"] == 3
    assert _arrow(app, "▶").disabled


def test_stepping_forward_says_the_squad_is_still_the_published_one(bundle, monkeypatch):
    """The fifteen were solved once, on the whole horizon. Showing next week's forecasts over
    the same squad must not read as 'this is what to pick next week'."""
    app = _run(bundle, monkeypatch)
    _arrow(app, "▶").click().run()

    warned = " ".join(str(w.value) for w in app.warning)
    assert "GW1 decision" in warned or "chosen once" in warned


def test_a_bundle_without_forecasts_shows_no_navigator(single_gw_bundle, monkeypatch):
    """Old bundles predate `forecasts.parquet`. They must still render, minus the arrows."""
    app = _run(single_gw_bundle, monkeypatch)

    assert not app.exception
    assert next(m for m in app.metric if m.label == "Gameweek").value == "1"
    assert not [b for b in app.button if b.label in ("◀", "▶")]


def test_my_team_explains_itself_when_the_model_cannot_run_here(bundle, monkeypatch, tmp_path):
    """The deployed app is the bundle and nothing else, so this tab cannot work there.

    Reported from Streamlit Cloud as a raw `FileNotFoundError: no table at
    data/interim/teams/season=2026-27` traceback. The packages install fine there — it is the
    DATA that is gitignored — so an ImportError guard never fired. The tab must say so before
    offering a button that spins for twenty seconds and then fails.
    """
    import app as app_module

    monkeypatch.setattr(app_module, "MODEL_INPUTS", {"the archive": "data/interim"})
    monkeypatch.chdir(tmp_path)
    assert app_module.missing_model_inputs() == []      # the real checkout HAS these

    monkeypatch.setattr(app_module, "MODEL_INPUTS", {"the archive": "definitely/not/here"})
    assert app_module.missing_model_inputs() == ["the archive"]


def test_an_empty_directory_counts_as_a_missing_input(monkeypatch, tmp_path):
    """A shallow checkout can leave the directory behind with nothing in it, and `config.path`
    creates each layer on read — so existence alone is not evidence of data."""
    import app as app_module

    (tmp_path / "hollow").mkdir()
    assert not app_module._populated(tmp_path / "hollow")
    (tmp_path / "hollow" / "something.parquet").write_text("x", encoding="utf-8")
    assert app_module._populated(tmp_path / "hollow")


def test_my_team_prompts_rather_than_running_the_model_on_load(bundle, monkeypatch):
    """The expensive guarantee: opening the page must not fire a 20-second pipeline run.

    Every other tab reads the bundle; this one forecasts the whole horizon. Asserted through
    behaviour rather than by patching `analyse`, because `AppTest` execs the script into its
    own namespace and a patch on the imported module would never reach it — the test would
    pass whether or not the guard existed.

    The tab must therefore be sitting at its prompt, with no analysis rendered. A run on load
    would also have to reach the network without an entry id, which `test_the_app_runs_
    without_exceptions` would catch.
    """
    app = _run(bundle, monkeypatch)
    text = " ".join(str(i.value) for i in app.info) + " ".join(
        str(c.value) for c in app.caption
    )

    assert "entry id" in text.lower()
    assert not app.exception
    assert "myteam_entry" not in app.session_state      # nothing was queued for analysis


def test_the_entry_id_is_remembered_across_restarts(bundle, monkeypatch, tmp_path):
    """'Cache it until overridden' has to mean the next session too, not just the next rerun.

    Session state dies with the browser tab, so the id is persisted to a gitignored file.
    """
    import importlib

    cache = tmp_path / "entry.json"
    monkeypatch.setenv("FPL_ENTRY_CACHE", str(cache))
    import app as app_module

    importlib.reload(app_module)

    assert app_module.remembered_entry() is None       # nothing remembered yet
    app_module.remember_entry(3468852)
    assert app_module.remembered_entry() == 3468852

    cache.write_text("not json at all", encoding="utf-8")
    assert app_module.remembered_entry() is None       # damaged file reads as 'none yet'


def test_the_bundle_carries_no_personal_data(bundle):
    """The bundle is committed and deployed, so it must describe the GAME and not the user.
    Anything entry-specific belongs in `fpl myteam --brief`, which stays local."""
    from fpl_expert.serving import read_bundle

    loaded = read_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    forbidden = {"entry", "entry_id", "manager", "selling_price", "bank", "free_transfers"}

    assert not forbidden & set(loaded["players"].columns)
    assert not forbidden & set(manifest)
