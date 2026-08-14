"""Minutes model: bucketing, point-in-time features, and the availability gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fpl_expert.models.minutes import (
    TransitionBaseline,
    apply_availability_gate,
    assemble_predictions,
    bucket_minutes,
    build_features,
    clean_history,
)


def _history(rows):
    base = {"season": "2025-26", "position": "MID", "value": 60, "name": "Player One"}
    return pd.DataFrame([{**base, **r} for r in rows])


# --- bucketing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("minutes", "expected"), [(0, 0), (1, 1), (59, 1), (60, 2), (90, 2), (None, 0)]
)
def test_bucket_boundaries_match_the_scoring_rules(minutes, expected):
    """60 is the threshold for the second appearance point AND clean-sheet eligibility."""
    assert bucket_minutes(pd.Series([minutes])).iloc[0] == expected


# --- cleaning -------------------------------------------------------------


def test_clean_history_drops_assistant_manager_rows():
    """'AM' rows are the defunct Assistant Manager entries — permanently zero minutes,
    and they would drag every base rate downward if left in."""
    df = _history([
        {"GW": 1, "minutes": 90, "position": "MID"},
        {"GW": 1, "minutes": 0, "position": "AM"},
    ])
    assert clean_history(df)["position"].tolist() == ["MID"]


def test_clean_history_harmonises_goalkeeper_labels():
    """The archive uses GK in some seasons and GKP in others."""
    df = _history([{"GW": 1, "minutes": 90, "position": "GKP"}])
    assert clean_history(df)["position"].iloc[0] == "GK"


# --- point-in-time features ----------------------------------------------


def test_features_never_see_the_current_gameweek():
    """The shift(1) is the point-in-time guarantee at this layer. A player who blanks for
    four weeks then plays 90 must show lag features reflecting the blanks, not the 90."""
    df = _history([{"GW": gw, "minutes": 0} for gw in range(1, 5)]
                  + [{"GW": 5, "minutes": 90}])
    out = build_features(df).sort_values("GW")

    last = out[out["GW"] == 5].iloc[0]
    assert last["minutes_lag1"] == 0
    assert last["played_ewm"] == 0.0        # knows only the four blanks
    assert last["target"] == 2              # but the target IS the 90


def test_first_appearance_has_no_history():
    out = build_features(_history([{"GW": 1, "minutes": 90}]))
    assert pd.isna(out["minutes_lag1"].iloc[0])
    assert out["career_games"].iloc[0] == 0


def test_features_key_on_name_so_history_survives_a_new_season():
    """FPL reassigns element ids every season. Keying on the id would make every player a
    debutant each August and throw away most of the signal."""
    rows = [{"GW": gw, "minutes": 90, "season": "2024-25", "element": 111} for gw in range(1, 39)]
    rows += [{"GW": 1, "minutes": 90, "season": "2025-26", "element": 999}]   # id changed
    out = build_features(_history(rows))

    new_season = out[(out["season"] == "2025-26") & (out["GW"] == 1)].iloc[0]
    assert new_season["career_games"] == 38          # carried across the season boundary
    assert new_season["played_ewm"] == pytest.approx(1.0)


def test_appearances_excludes_the_current_row():
    df = _history([{"GW": gw, "minutes": 90} for gw in range(1, 4)])
    out = build_features(df).sort_values("GW")
    assert out["appearances"].tolist() == [0, 1, 2]


# --- assembling predictions ----------------------------------------------


def test_assemble_predictions_computes_appearance_points():
    proba = np.array([[0.2, 0.3, 0.5]])
    out = assemble_predictions(proba)
    assert out["p_appear"].iloc[0] == pytest.approx(0.8)
    assert out["expected_appearance_points"].iloc[0] == pytest.approx(0.3 * 1 + 0.5 * 2)
    assert out["expected_minutes"].iloc[0] == pytest.approx(0.3 * 22 + 0.5 * 85)


# --- the availability gate ------------------------------------------------


def test_gate_removes_all_playing_mass_at_zero_percent():
    preds = assemble_predictions(np.array([[0.1, 0.2, 0.7]]))
    out = apply_availability_gate(preds, pd.Series([0]))
    assert out["p_zero"].iloc[0] == pytest.approx(1.0)
    assert out["expected_appearance_points"].iloc[0] == pytest.approx(0.0)


def test_gate_halves_playing_mass_at_fifty_percent():
    preds = assemble_predictions(np.array([[0.1, 0.2, 0.7]]))
    out = apply_availability_gate(preds, pd.Series([50]))
    assert out["p_short"].iloc[0] == pytest.approx(0.1)
    assert out["p_long"].iloc[0] == pytest.approx(0.35)
    assert out["p_zero"].iloc[0] == pytest.approx(0.55)


def test_gate_preserves_the_short_long_ratio():
    """A doubtful player who would have started still starts IF fit, so the shape of the
    playing distribution should not change — only its total mass."""
    preds = assemble_predictions(np.array([[0.1, 0.2, 0.7]]))
    out = apply_availability_gate(preds, pd.Series([25]))
    assert out["p_long"].iloc[0] / out["p_short"].iloc[0] == pytest.approx(0.7 / 0.2)


def test_gate_is_a_no_op_when_there_is_no_news():
    """`chance_of_playing_next_round` is null when FPL has published nothing, which means
    available — not unknown, and certainly not zero."""
    preds = assemble_predictions(np.array([[0.1, 0.2, 0.7]]))
    out = apply_availability_gate(preds, pd.Series([np.nan]))
    assert out["p_long"].iloc[0] == pytest.approx(0.7)


def test_status_overrides_a_stale_percentage():
    """A suspension is certain regardless of what the percentage field still says."""
    preds = assemble_predictions(np.array([[0.1, 0.2, 0.7]]))
    out = apply_availability_gate(preds, pd.Series([100]), status=pd.Series(["s"]))
    assert out["p_zero"].iloc[0] == pytest.approx(1.0)


# --- baseline -------------------------------------------------------------


def test_transition_baseline_learns_the_persistence_of_minutes():
    df = pd.DataFrame({
        "minutes_lag1": [90] * 80 + [0] * 80,
        "target": [2] * 70 + [0] * 10 + [0] * 75 + [2] * 5,
    })
    baseline = TransitionBaseline().fit(df)
    proba = baseline.predict_proba(pd.DataFrame({"minutes_lag1": [90, 0]}))

    assert proba[0, 2] == pytest.approx(0.875)     # played last week -> likely again
    assert proba[1, 2] == pytest.approx(0.0625)    # didn't -> unlikely


def _two_seasons(first_minutes, second_minutes, name="Player One"):
    """One player across a season boundary, so the season-level features have something to
    summarise on either side of it."""
    rows = [
        {"season": "2023-24", "GW": i + 1, "minutes": m, "name": name}
        for i, m in enumerate(first_minutes)
    ]
    rows += [
        {"season": "2024-25", "GW": i + 1, "minutes": m, "name": name}
        for i, m in enumerate(second_minutes)
    ]
    return _history(rows)


def test_prev_season_rate_summarises_only_the_season_before():
    """A season's own outcome leaking into its own rows would be the most direct lookahead
    available here, so the summary is shifted a whole season forward."""
    features = build_features(_two_seasons([90, 90, 90, 0], [0, 0, 0, 0]))

    first = features[features["season"] == "2023-24"]
    assert first["prev_season_started_rate"].isna().all()      # nothing precedes it

    second = features[features["season"] == "2024-25"]
    # Three of four at 60+ last season, and it must NOT reflect this season's four blanks.
    assert np.allclose(second["prev_season_started_rate"], 0.75)
    assert (second["prev_season_games"] == 4).all()


def test_season_games_is_zero_at_an_opener():
    """The cue that tells the model there is no within-season evidence yet."""
    features = build_features(_two_seasons([90, 90], [90, 90]))
    openers = features[features["GW"] == 1]
    assert len(openers) == 2
    assert (openers["season_games"] == 0).all()
    assert openers["season_started_rate"].isna().all()


def test_season_started_rate_excludes_the_current_row():
    features = build_features(_two_seasons([90, 90], [90, 0, 90]))
    features = features[features["season"] == "2024-25"]
    ordered = features.sort_values("GW")
    # By gameweek 3 the rate is the mean of gameweeks 1-2 (one start of two), not of 1-3.
    assert ordered["season_started_rate"].iloc[2] == pytest.approx(0.5)
