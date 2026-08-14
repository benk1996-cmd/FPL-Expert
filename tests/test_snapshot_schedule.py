"""The checkpoint ladder that decides when a scheduled job captures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from fpl_expert.data import snapshot as snap

CHECKPOINTS = (48.0, 24.0, 6.0, 2.0, 0.5)


@pytest.fixture
def clock(monkeypatch):
    """Pin the deadline and let each test choose how far away 'now' is."""
    deadline = datetime(2026, 8, 21, 17, 30, tzinfo=UTC)

    def _set(hours_to_deadline, held=()):
        monkeypatch.setattr(snap, "FplApi", lambda: _FakeApi())
        monkeypatch.setattr(snap, "next_gameweek", lambda _b: 1)
        monkeypatch.setattr(snap, "_deadline_for", lambda _b, _gw: deadline)
        monkeypatch.setattr(
            snap, "datetime",
            _FrozenDatetime(deadline - timedelta(hours=hours_to_deadline)),
        )
        rows = [
            {"target_gw": 1, "taken_before_deadline": True, "hours_to_deadline": h} for h in held
        ]
        monkeypatch.setattr(snap, "list_snapshots", lambda: pd.DataFrame(rows))

    return _set


class _FakeApi:
    def bootstrap_static(self, use_cache=True):
        return {"events": []}


class _FrozenDatetime:
    def __init__(self, now):
        self._now = now

    def now(self, tz=None):
        return self._now


def test_not_due_far_from_the_deadline(clock):
    clock(200)
    due, why = snap.snapshot_due(CHECKPOINTS)
    assert not due and "earliest checkpoint" in why


def test_due_at_the_first_checkpoint(clock):
    clock(40)                                   # inside 48h, nothing held
    due, _ = snap.snapshot_due(CHECKPOINTS)
    assert due


def test_not_due_twice_within_the_same_checkpoint(clock):
    """Hourly scheduling must not spam captures — one per checkpoint is enough."""
    clock(40, held=[44.0])                      # already captured inside the 48h band
    due, why = snap.snapshot_due(CHECKPOINTS)
    assert not due and "already hold" in why


def test_due_again_at_a_tighter_checkpoint(clock):
    """The escalation that matters: a 44h-old capture misses Friday team news, so a
    fresh one is taken as the deadline nears. PointInTime uses the latest."""
    clock(5, held=[44.0])                       # inside 6h, but only hold a 44h capture
    due, why = snap.snapshot_due(CHECKPOINTS)
    assert due and "6.0h checkpoint" in why


def test_not_due_after_the_deadline(clock):
    clock(-1)
    due, why = snap.snapshot_due(CHECKPOINTS)
    assert not due and "has passed" in why


def test_ladder_escalates_all_the_way_down(clock):
    """Walk the ladder: each band triggers exactly one capture."""
    held = []
    for hours in (47.0, 23.0, 5.0, 1.5, 0.2):
        clock(hours, held=held)
        due, _ = snap.snapshot_due(CHECKPOINTS)
        assert due, f"expected a capture at {hours}h to deadline"
        held.append(hours)
    assert len(held) == 5

    clock(0.1, held=held)                       # everything captured
    assert not snap.snapshot_due(CHECKPOINTS)[0]
