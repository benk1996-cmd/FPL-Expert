"""Point-in-time guarantees: the machinery that keeps the backtest honest."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from fpl_expert.data.odds import split_admissible
from fpl_expert.data.snapshot import (
    LookaheadError,
    MissingSnapshotError,
    PointInTime,
    assert_no_post_deadline_columns,
    list_snapshots,
    parse_stamp,
)
from fpl_expert.data.storage import write_raw, write_table


def _manifest(stamp, gw, before_deadline, reason="test"):
    return {
        "stamp": stamp, "target_gw": gw, "taken_before_deadline": before_deadline,
        "reason": reason, "deadline": "2026-08-21T17:30:00+00:00", "hours_to_deadline": 1.0,
    }


# --- the lookahead guard --------------------------------------------------


def test_guard_rejects_closing_odds():
    df = pd.DataFrame({"home_team": ["Arsenal"], "p_home_open": [0.5], "p_home_close": [0.55]})
    with pytest.raises(LookaheadError, match="p_home_close"):
        assert_no_post_deadline_columns(df)


def test_guard_rejects_match_results():
    """Final scores postdate the deadline just as closing odds do."""
    df = pd.DataFrame({"home_team": ["Arsenal"], "home_goals": [2], "away_goals": [1]})
    with pytest.raises(LookaheadError, match="home_goals"):
        assert_no_post_deadline_columns(df)


def test_guard_passes_clean_feature_frame():
    df = pd.DataFrame({"home_team": ["Arsenal"], "p_home_open": [0.5], "overround_open": [1.03]})
    assert_no_post_deadline_columns(df)  # must not raise


def test_guard_catches_the_realistic_failure_a_join():
    """The likely accident: joining odds for team names and dragging closing columns along."""
    tidy = pd.DataFrame({
        "season": ["2025-26"], "date": [pd.Timestamp("2026-05-24")],
        "home_team": ["Spurs"], "away_team": ["Everton"],
        "p_home_open": [0.50], "p_home_close": [0.52], "home_goals": [1], "away_goals": [0],
    })
    features, evaluation = split_admissible(tidy)
    joined = features.merge(evaluation, on=["season", "date", "home_team", "away_team"])
    with pytest.raises(LookaheadError):
        assert_no_post_deadline_columns(joined, context="naive join")


# --- the physical split ---------------------------------------------------


def test_split_puts_closing_odds_and_results_out_of_reach():
    tidy = pd.DataFrame({
        "season": ["2025-26"], "date": [pd.Timestamp("2026-05-24")],
        "home_team": ["Spurs"], "away_team": ["Everton"],
        "p_home_open": [0.50], "overround_open": [1.03],
        "p_home_close": [0.52], "overround_close": [1.05],
        "home_goals": [1], "away_goals": [0],
    })
    features, evaluation = split_admissible(tidy)

    assert_no_post_deadline_columns(features)          # the whole point
    assert "p_home_open" in features
    assert "p_home_close" not in features
    assert {"p_home_close", "home_goals", "away_goals"} <= set(evaluation.columns)
    # Identifiers are duplicated so the halves can be rejoined for evaluation.
    assert "home_team" in features and "home_team" in evaluation


# --- resolving a gameweek to a snapshot -----------------------------------


def test_for_gameweek_takes_the_latest_pre_deadline_snapshot(tmp_config):
    """Closest to the deadline is best — it knows the most while still being legitimate."""
    for stamp, before in [
        ("20260819T120000Z", True),   # two days out
        ("20260821T170000Z", True),   # 30 min out — the one we want
        ("20260821T180000Z", False),  # after the deadline
    ]:
        write_raw(_manifest(stamp, 1, before), "snapshot", "manifest", stamp=stamp)
    write_raw({"elements": [], "events": [], "teams": []}, "snapshot", "bootstrap",
              stamp="20260821T170000Z")

    pit = PointInTime.for_gameweek(1)
    assert pit.stamp == "20260821T170000Z"
    assert pit.as_of == datetime(2026, 8, 21, 17, 0, tzinfo=UTC)


def test_for_gameweek_refuses_to_fall_back_to_a_post_deadline_snapshot(tmp_config):
    """Silently using post-deadline state is the exact failure this module prevents."""
    write_raw(_manifest("20260821T180000Z", 1, False), "snapshot", "manifest",
              stamp="20260821T180000Z")
    with pytest.raises(MissingSnapshotError, match="unrecoverable"):
        PointInTime.for_gameweek(1)


def test_for_gameweek_with_no_snapshots_at_all(tmp_config):
    with pytest.raises(MissingSnapshotError, match="no snapshots"):
        PointInTime.for_gameweek(1)


def test_snapshots_of_other_gameweeks_do_not_satisfy_a_gameweek(tmp_config):
    write_raw(_manifest("20260821T170000Z", 2, True), "snapshot", "manifest",
              stamp="20260821T170000Z")
    with pytest.raises(MissingSnapshotError):
        PointInTime.for_gameweek(1)


def test_list_snapshots_reads_every_manifest(tmp_config):
    write_raw(_manifest("20260819T120000Z", 1, True), "snapshot", "manifest",
              stamp="20260819T120000Z")
    write_raw(_manifest("20260821T170000Z", 1, True), "snapshot", "manifest",
              stamp="20260821T170000Z")
    df = list_snapshots()
    assert len(df) == 2
    assert df["stamp"].tolist() == ["20260819T120000Z", "20260821T170000Z"]  # oldest first


# --- reading through PointInTime ------------------------------------------


def test_point_in_time_odds_never_exposes_closing_columns(tmp_config):
    """Fallback path: no odds captured in the snapshot, so it reads odds_features."""
    write_raw(_manifest("20260821T170000Z", 1, True), "snapshot", "manifest",
              stamp="20260821T170000Z")
    write_table(
        pd.DataFrame({"home_team": ["Spurs"], "p_home_open": [0.5]}),
        "external", "odds_features",
    )
    odds = PointInTime.for_gameweek(1).odds()
    assert "p_home_open" in odds.columns
    assert not [c for c in odds.columns if c.endswith("_close")]


def test_point_in_time_odds_raises_if_the_features_table_is_contaminated(tmp_config):
    """Defence in depth: even if the upstream split regresses, reads still fail loudly."""
    write_raw(_manifest("20260821T170000Z", 1, True), "snapshot", "manifest",
              stamp="20260821T170000Z")
    write_table(
        pd.DataFrame({"home_team": ["Spurs"], "p_home_open": [0.5], "p_home_close": [0.6]}),
        "external", "odds_features",
    )
    with pytest.raises(LookaheadError):
        PointInTime.for_gameweek(1).odds()


def test_parse_stamp_roundtrip():
    assert parse_stamp("20260821T173000Z") == datetime(2026, 8, 21, 17, 30, tzinfo=UTC)


def test_planning_falls_back_to_the_latest_snapshot_for_a_future_gameweek(tmp_config):
    """`for_gameweek` answers a BACKTEST question — what did we know before this deadline —
    and a future gameweek has no such snapshot, so valuing a squad over a six-week horizon
    failed outright. Planning asks a different question and gets a separate constructor, so
    nothing evaluating the past can reach the fallback by accident.
    """
    from fpl_expert.data.snapshot import MissingSnapshotError, PointInTime, take_snapshot

    take_snapshot(reason="test")
    listed = list_snapshots()
    covered = int(listed.loc[listed["taken_before_deadline"], "target_gw"].dropna().iloc[0])

    with pytest.raises(MissingSnapshotError):
        PointInTime.for_gameweek(covered + 5)

    planning = PointInTime.for_planning(covered + 5)
    assert planning.target_gw == covered + 5
    assert planning.stamp == listed.sort_values("stamp")["stamp"].iloc[-1]


def test_planning_still_prefers_a_real_pre_deadline_snapshot(tmp_config):
    from fpl_expert.data.snapshot import PointInTime, take_snapshot

    take_snapshot(reason="test")
    listed = list_snapshots()
    covered = int(listed.loc[listed["taken_before_deadline"], "target_gw"].dropna().iloc[0])
    assert PointInTime.for_planning(covered).stamp == PointInTime.for_gameweek(covered).stamp
