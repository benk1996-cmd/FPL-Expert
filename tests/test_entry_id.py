"""Resolving the FPL entry id from `.env`, and the precedence that governs it."""

from __future__ import annotations

import pytest

from fpl_expert.config import ENTRY_VAR, entry_id


@pytest.fixture(autouse=True)
def _no_ambient_entry(monkeypatch):
    """Tests must not read the developer's real id, nor the repo's real `.env`."""
    monkeypatch.delenv(ENTRY_VAR, raising=False)


def test_an_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv(ENTRY_VAR, "111")
    assert entry_id(999) == 999


def test_the_environment_is_used_when_no_argument_is_given(monkeypatch):
    monkeypatch.setenv(ENTRY_VAR, "3468852")
    assert entry_id() == 3468852


def test_a_real_environment_variable_beats_the_dotenv_file(monkeypatch, tmp_path):
    """Standard precedence, and what lets a deployment or a one-off shell override the file."""
    monkeypatch.setenv(ENTRY_VAR, "222")
    assert entry_id() == 222


def test_a_missing_id_explains_where_to_put_one(monkeypatch, tmp_path):
    """An empty `--entry` must not become a confusing API 404 three steps later."""
    monkeypatch.setattr("fpl_expert.config.load_env", lambda *a, **k: None)
    with pytest.raises(ValueError, match=r"\.env"):
        entry_id()


def test_a_non_numeric_id_is_rejected_at_the_boundary(monkeypatch):
    monkeypatch.setenv(ENTRY_VAR, "beanchodeFC")
    monkeypatch.setattr("fpl_expert.config.load_env", lambda *a, **k: None)
    with pytest.raises(ValueError, match="not a plain number"):
        entry_id()


def test_whitespace_around_the_value_is_tolerated(monkeypatch):
    monkeypatch.setenv(ENTRY_VAR, "  3468852  ")
    assert entry_id() == 3468852


def test_the_app_and_the_cli_read_the_same_variable(monkeypatch):
    """Two entry points, one id. They parse it differently — the front end cannot depend on
    dotenv — so this pins them to the same variable and the same answer."""
    import app as app_module

    monkeypatch.setenv(ENTRY_VAR, "3468852")
    assert app_module._entry_from_env() == entry_id()


def test_the_committed_example_is_a_placeholder_not_a_real_id():
    """`.env.example` is committed; `.env` is not. The template must not carry anyone's id."""
    from pathlib import Path

    from fpl_expert.config import project_root

    text = Path(project_root() / ".env.example").read_text(encoding="utf-8")
    assert "FPL_ENTRY=1234567" in text
    assert "3468852" not in text
