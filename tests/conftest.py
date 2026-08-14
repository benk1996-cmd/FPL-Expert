"""Shared fixtures. Tests are offline by design — no test hits the FPL API."""

from __future__ import annotations

import pytest

from fpl_expert.config import Config


@pytest.fixture
def tmp_config(tmp_path, monkeypatch):
    """A Config whose data layers point into tmp_path, patched into the storage module."""
    cfg = Config(
        project={"season": "2026/27"},
        paths={
            "raw": str(tmp_path / "raw"),
            "interim": str(tmp_path / "interim"),
            "processed": str(tmp_path / "processed"),
            "external": str(tmp_path / "external"),
        },
    )
    monkeypatch.setattr("fpl_expert.data.storage.load_config", lambda: cfg)
    return cfg


@pytest.fixture
def bootstrap():
    """A minimal bootstrap-static payload with the shapes the parsers rely on."""
    return {
        "total_players": 3_530_329,
        "elements": [
            {
                "id": 1, "web_name": "Raya", "element_type": 1, "team": 1, "now_cost": 55,
                "selected_by_percent": "12.3", "form": "4.5", "expected_goals": "0.0",
                "scout_risks": [], "news": "",
            },
            {
                "id": 2, "web_name": "Saka", "element_type": 3, "team": 1, "now_cost": 100,
                "selected_by_percent": "41.2", "form": "6.1", "expected_goals": "0.42",
                "scout_risks": [{"type": "rotation"}], "news": "",
            },
        ],
        "teams": [{"id": 1, "name": "Arsenal", "strength": 5}],
        "events": [
            {"id": 1, "deadline_time": "2026-08-21T17:30:00Z", "finished": False,
             "is_current": False, "is_next": True, "overrides": {"rules": {}}, "chip_plays": []},
            {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": False,
             "is_current": False, "is_next": False, "overrides": {"rules": {}}, "chip_plays": []},
        ],
    }
