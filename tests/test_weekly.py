"""The composite weekly refresh: ordering, isolation, and step selection."""

from __future__ import annotations

from typer.testing import CliRunner

from fpl_expert.cli import WEEKLY_STEPS, app

runner = CliRunner()


def _stub(monkeypatch, calls, failing=(), benign=()):
    """Replace every step with a recorder so ordering and isolation can be asserted."""
    from fpl_expert import cli

    def make(name):
        def step(**_kwargs):
            calls.append(name)
            if name in benign:
                raise RuntimeError("no settled gameweeks yet - nothing to ingest")
            if name in failing:
                raise RuntimeError(f"{name} blew up")
        return step

    for name in WEEKLY_STEPS:
        monkeypatch.setattr(cli, name, make(name), raising=True)


def test_every_step_runs_in_the_order_the_pipeline_requires(monkeypatch):
    """`results` before `minutes` or the retrain misses the gameweek just played;
    `snapshot` before `publish` or the strict point-in-time accessor has nothing to read."""
    calls: list[str] = []
    _stub(monkeypatch, calls)
    result = runner.invoke(app, ["weekly"])

    assert result.exit_code == 0, result.output
    assert calls == list(WEEKLY_STEPS)
    assert calls.index("results") < calls.index("minutes")
    assert calls.index("snapshot") < calls.index("publish")


def test_one_failing_step_does_not_abort_the_rest(monkeypatch):
    """Losing eleven minutes of ingestion because the odds feed was down is the failure mode
    this exists to prevent."""
    calls: list[str] = []
    _stub(monkeypatch, calls, failing={"odds"})
    result = runner.invoke(app, ["weekly"])

    assert calls == list(WEEKLY_STEPS), "later steps must still be attempted"
    assert result.exit_code == 1, "but the run must report failure"
    assert "FAILED" in result.output
    assert "odds" in result.output


def test_nothing_to_ingest_is_reported_as_skipped_not_failed(monkeypatch):
    """Running mid-week, before the gameweek is checked, is normal."""
    calls: list[str] = []
    _stub(monkeypatch, calls, benign={"results"})
    result = runner.invoke(app, ["weekly"])

    assert result.exit_code == 0, result.output
    assert "skipped" in result.output
    assert "FAILED" not in result.output


def test_only_and_skip_select_steps_and_keep_the_order(monkeypatch):
    calls: list[str] = []
    _stub(monkeypatch, calls)
    runner.invoke(app, ["weekly", "--only", "publish", "--only", "snapshot"])
    assert calls == ["snapshot", "publish"], "order follows the pipeline, not the flags"

    calls.clear()
    runner.invoke(app, ["weekly", "--skip", "results", "--skip", "minutes"])
    assert calls == ["odds", "update", "snapshot", "publish"]


def test_an_unknown_step_is_rejected_rather_than_silently_ignored(monkeypatch):
    calls: list[str] = []
    _stub(monkeypatch, calls)
    result = runner.invoke(app, ["weekly", "--only", "nonsense"])

    assert result.exit_code != 0
    assert calls == []
    assert "nonsense" in result.output


def test_odds_is_given_the_current_season_explicitly(monkeypatch):
    """`ingest_odds` defaults to `history_seasons`, which lists COMPLETED seasons, so a bare
    `fpl odds` reports success and fetches nothing. That left Dixon-Coles fitting the promoted
    clubs on zero current-season matches, pinned at their parameter bounds."""
    from fpl_expert import cli

    seen: dict = {}
    for name in WEEKLY_STEPS:
        monkeypatch.setattr(cli, name, lambda **kw: None, raising=True)
    monkeypatch.setattr(cli, "odds", lambda **kw: seen.update(kw), raising=True)

    runner.invoke(app, ["weekly", "--only", "odds"])

    assert seen.get("seasons"), "the season must be passed, not left to the config default"
    assert "-" in seen["seasons"][0], f"expected a season like 2026-27, got {seen['seasons']}"
