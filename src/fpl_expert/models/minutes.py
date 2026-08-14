"""Minutes model — the single largest driver of FPL points variance.

Whether a player scores 0, 2 or 12 in a gameweek is decided first by whether he plays at
all. 58% of player-gameweeks are zero minutes; get that wrong and no amount of attacking
modelling recovers it. So minutes are predicted first and everything else is conditioned
on them.

The target is three buckets rather than a minutes regression, because the FPL scoring
thresholds are what matter and they are discontinuous:

    0 minutes   no appearance points, no clean sheet
    1-59        1 appearance point, no clean sheet
    60+         2 appearance points, and clean-sheet eligibility

Predicting 45 minutes when the truth is bimodal (bench cameo or full match) is worse than
useless — the mean is the least likely outcome. A distribution over buckets is what the
points assembler (Item 10) actually needs.

**Availability data is deliberately NOT a training feature.** FPL's
`chance_of_playing_next_round` and `status` are the strongest live signals we have, but the
historical archive does not contain them — they are only captured going forward, by our own
snapshots (Item 3). Training on features that will not exist for past gameweeks, or serving
with features the model never saw, is train/serve skew either way. Instead
`apply_availability_gate` uses `chance_of_playing_next_round` at inference as an explicit
multiplicative prior on playing at all. It is already a probability, so it needs no fitting,
and keeping it outside the model makes its effect visible and overridable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Bucket labels. Ordered, and the order is used by `expected_minutes`.
BUCKET_ZERO, BUCKET_SHORT, BUCKET_LONG = 0, 1, 2
BUCKET_NAMES = {BUCKET_ZERO: "p_zero", BUCKET_SHORT: "p_short", BUCKET_LONG: "p_long"}

# Typical minutes within each bucket, for converting a distribution to an expectation.
# The short bucket is deliberately not 30: substitute appearances cluster late in matches.
BUCKET_MINUTES = {BUCKET_ZERO: 0.0, BUCKET_SHORT: 22.0, BUCKET_LONG: 85.0}

# Eleven players for ninety minutes. A hard constraint the per-player model cannot see, since
# it scores each player independently — see `balance_team_minutes`.
TEAM_MINUTES_BUDGET = 11 * 90.0

# Positions that are not real players. 'AM' rows are the defunct Assistant Manager entries
# from 2024-25, which carry zero minutes forever and would bias every base rate downward.
NON_PLAYER_POSITIONS = frozenset({"AM"})
POSITION_ALIASES = {"GKP": "GK"}

FEATURE_COLUMNS = [
    "played_ewm", "started_ewm", "minutes_ewm",
    "minutes_lag1", "minutes_lag2", "started_lag1",
    "appearances", "career_games", "gw", "value", "rest_days",
    "position_code",
    # Season-level context. Worth a small, consistent gain — walk-forward log loss 0.4847 ->
    # 0.4828 overall and 0.7902 -> 0.7876 on season openers, better in every test season.
    #
    # They were added to fix something that turned out not to be broken. The EWM features
    # above have a five-gameweek half-life, so at a season opener they are made entirely of
    # LAST season's closing weeks, and that looked like a defect: Haaland went into 2026-27
    # GW1 on p_long 0.609 against Fernandes on 0.845, despite 60+ rates of 87% and 89% across
    # the season just finished. Two investigations said otherwise.
    #
    # Resetting the EWM at the season boundary — so an opener could not see last May at all —
    # made openers WORSE in all four test seasons (0.788 -> 0.802). And the empirical base
    # rate settles it: across five season boundaries, players with a strong prior season
    # (>=0.8) but weak closing form (EWM <=0.7) started the next opener only **52.9%** of the
    # time (n=34), against 77.2% for those who finished strongly. `started_ewm` correlates
    # with actually starting an opener at 0.535, ahead of the prior-season rate at 0.510.
    #
    # Late-season minutes carry real information about the next campaign rather than noise.
    # 0.609 for a player rested twice in his final six is not an error, it is slightly
    # generous. These features stay because they help; the story that motivated them was wrong.
    "season_games", "season_started_rate",
    "prev_season_started_rate", "prev_season_games",
]


def bucket_minutes(minutes: pd.Series) -> pd.Series:
    """Map raw minutes onto the three scoring-relevant buckets."""
    return pd.Series(
        np.select(
            [minutes.fillna(0) <= 0, minutes.fillna(0) < 60],
            [BUCKET_ZERO, BUCKET_SHORT],
            default=BUCKET_LONG,
        ),
        index=minutes.index,
        dtype=int,
    )


def clean_history(history: pd.DataFrame) -> pd.DataFrame:
    """Drop non-player rows and harmonise position labels across seasons."""
    df = history.copy()
    df["position"] = df["position"].replace(POSITION_ALIASES)
    before = len(df)
    df = df[~df["position"].isin(NON_PLAYER_POSITIONS)]
    df = df[df["minutes"].notna()]
    if (dropped := before - len(df)) > 0:
        log.info("dropped %d non-player/incomplete rows", dropped)
    return df


def build_features(history: pd.DataFrame, half_life: float = 5.0) -> pd.DataFrame:
    """Point-in-time features for every player-gameweek.

    Every feature is built from `shift(1)` within a player's own timeline, so a row never
    sees its own result. That shift is the whole point-in-time guarantee at this layer.

    Players are keyed by name rather than `element`, because FPL reassigns element ids each
    season — keying on the id would make every player a debutant every August and throw away
    the cross-season history that is most of the signal.
    """
    df = clean_history(history).copy()
    df["player_key"] = df["name"]
    df["_season_rank"] = df["season"].rank(method="dense").astype(int)
    df["_idx"] = df["_season_rank"] * 38 + df["GW"]
    df = df.sort_values(["player_key", "_idx"]).reset_index(drop=True)

    key = df["player_key"]
    played = (df["minutes"] > 0).astype(float)
    started = (df["minutes"] >= 60).astype(float)

    for name, series in (("played", played), ("started", started), ("minutes", df["minutes"])):
        prev = series.groupby(key).shift(1)
        df[f"_{name}_prev"] = prev
        df[f"{name}_ewm"] = (
            prev.groupby(key).ewm(halflife=half_life, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    df["minutes_lag1"] = df["_minutes_prev"]
    df["minutes_lag2"] = df["minutes"].groupby(key).shift(2)
    df["started_lag1"] = df["_started_prev"]

    # --- season-level context, so an opener has something other than last May to go on.
    within = [key, df["season"]]
    # Expanding mean of "started 60+", excluding the current row. At gameweek 1 there is no
    # evidence yet and this is NaN, which LightGBM handles natively as its own branch.
    df["season_games"] = df.groupby(within).cumcount()
    df["season_started_rate"] = (
        started.groupby(within).cumsum() - started
    ) / df["season_games"].replace(0, np.nan)

    # The PREVIOUS season's complete record. Fully known before a new season starts, so this
    # is not lookahead — it is the single most relevant thing available in August. Built by
    # summarising each player-season, then shifting that summary forward one season, so a
    # season's own outcome can never leak into its own rows.
    per_season = df.groupby(["player_key", "season"], as_index=False).agg(
        _games=("minutes", "size"), _started=("minutes", lambda m: (m >= 60).sum())
    )
    per_season["_rate"] = per_season["_started"] / per_season["_games"]
    per_season = per_season.sort_values(["player_key", "season"])
    per_season["prev_season_started_rate"] = per_season.groupby("player_key")["_rate"].shift(1)
    per_season["prev_season_games"] = per_season.groupby("player_key")["_games"].shift(1)

    # Mapped through a MultiIndex rather than merged: a merge rebuilds the index, and the
    # cumulative features below are assigned from Series carrying the ORIGINAL one. They would
    # still align today by luck of row order, and stop aligning the moment anything upstream
    # reorders.
    keys = pd.MultiIndex.from_arrays([df["player_key"], df["season"]])
    lookup = per_season.set_index(["player_key", "season"])
    df["prev_season_started_rate"] = keys.map(lookup["prev_season_started_rate"])
    df["prev_season_games"] = keys.map(lookup["prev_season_games"])
    # Cumulative counts, shifted so the current row is excluded.
    df["career_games"] = df.groupby("player_key").cumcount()
    df["appearances"] = played.groupby(key).cumsum() - played

    if "kickoff_time" in df.columns:
        kickoff = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
        df["rest_days"] = (
            kickoff.groupby(key).diff().dt.total_seconds().div(86400).clip(upper=14).fillna(14)
        )
    else:
        df["rest_days"] = 14.0

    df["position_code"] = df["position"].map({"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}).fillna(-1)
    df["gw"] = df["GW"]
    if "value" not in df.columns:
        df["value"] = np.nan
    df["target"] = bucket_minutes(df["minutes"])
    return df


# --- baseline -------------------------------------------------------------


@dataclass
class TransitionBaseline:
    """Predict this gameweek's bucket from last gameweek's. Deliberately hard to beat.

    Minutes are strongly autocorrelated — P(60+ | 60+ last week) is around 0.78 against
    0.10 otherwise — so any model that cannot beat this is adding nothing, and saying so
    early is cheaper than discovering it in Item 14.
    """

    table: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> TransitionBaseline:
        prev = bucket_minutes(df["minutes_lag1"].fillna(0))
        self.table = (
            pd.crosstab(prev, df["target"], normalize="index")
            .reindex(index=[0, 1, 2], columns=[0, 1, 2])
            .fillna(1 / 3)
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        prev = bucket_minutes(df["minutes_lag1"].fillna(0))
        return self.table.reindex(prev).to_numpy()


# --- model ----------------------------------------------------------------


@dataclass
class MinutesModel:
    """Gradient-boosted multiclass model over the three minutes buckets."""

    params: dict = field(default_factory=dict)
    model: object | None = None
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def _default_params(self) -> dict:
        return {
            "objective": "multiclass",
            "num_class": 3,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 100,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
            "seed": 7,
            **self.params,
        }

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            num_boost_round: int = 600) -> MinutesModel:
        import lightgbm as lgb

        dtrain = lgb.Dataset(train[self.features], label=train["target"])
        callbacks, valid_sets = [], None
        if valid is not None and not valid.empty:
            valid_sets = [lgb.Dataset(valid[self.features], label=valid["target"])]
            callbacks = [lgb.early_stopping(50, verbose=False)]
        self.model = lgb.train(
            self._default_params(), dtrain, num_boost_round=num_boost_round,
            valid_sets=valid_sets, callbacks=callbacks,
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        return self.model.predict(df[self.features])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bucket probabilities, expected minutes, and expected appearance points."""
        proba = self.predict_proba(df)
        return assemble_predictions(proba, index=df.index)

    def save(self, path) -> None:
        """Persist as LightGBM text plus a feature manifest.

        The feature list is stored alongside the booster because a silent reordering of
        columns at predict time produces plausible-looking nonsense rather than an error.
        """
        import json
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        path.with_suffix(".features.json").write_text(
            json.dumps({"features": self.features}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> MinutesModel:
        import json
        from pathlib import Path

        import lightgbm as lgb

        path = Path(path)
        manifest = json.loads(path.with_suffix(".features.json").read_text(encoding="utf-8"))
        instance = cls(features=manifest["features"])
        instance.model = lgb.Booster(model_file=str(path))
        return instance


def assemble_predictions(proba: np.ndarray, index=None) -> pd.DataFrame:
    """Turn bucket probabilities into the quantities downstream models need."""
    out = pd.DataFrame(proba, columns=[BUCKET_NAMES[i] for i in (0, 1, 2)], index=index)
    out["expected_minutes"] = sum(
        out[BUCKET_NAMES[b]] * BUCKET_MINUTES[b] for b in (0, 1, 2)
    )
    # Appearance points: 1 for a short outing, 2 for 60+.
    out["expected_appearance_points"] = out["p_short"] * 1 + out["p_long"] * 2
    out["p_appear"] = out["p_short"] + out["p_long"]
    return out


def balance_team_minutes(
    predictions: pd.DataFrame,
    team: pd.Series,
    *,
    budget: float = TEAM_MINUTES_BUDGET,
    max_iter: int = 25,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    """Rescale playing probabilities so each team's expected minutes sum to eleven players.

    A team plays exactly 11 x 90 minutes. The minutes model does not know that — it scores
    each player independently — so nothing stops a squad's expectations summing to anything at
    all. Measured on the live GW1 frame: Hull 399 minutes, Chelsea 1236, mean 912 against a
    budget of 990, with 70% of clubs short.

    The error is driven by SQUAD SIZE rather than by injuries. A club with 36 registered
    players accumulates 36 small probabilities where only eleven can play, and a thin promoted
    squad accumulates too few. Both distort every downstream quantity, because expected minutes
    multiply the attacking rates and the bucket probabilities gate appearance, clean-sheet and
    defensive-contribution points.

    `p_short` and `p_long` are scaled by a common per-team factor and `p_zero` takes the
    remainder, so the SHAPE of a player's minutes distribution is preserved and only his
    likelihood of featuring moves. Iterated because the probabilities are capped at one: a
    single nailed starter cannot absorb more, so the excess redistributes to team-mates who
    can, which is water-filling rather than a closed-form rescale.

    Players already gated to zero stay at zero — an unavailable player is not a candidate for
    the minutes his team-mates give up.
    """
    out = predictions.copy()
    short = np.array(out["p_short"], dtype=float)
    long_ = np.array(out["p_long"], dtype=float)
    groups = pd.Series(team).reset_index(drop=True)

    for index in groups.groupby(groups, sort=False).groups.values():
        rows = np.asarray(index, dtype=int)
        for _ in range(max_iter):
            total = (short[rows] * BUCKET_MINUTES[BUCKET_SHORT]
                     + long_[rows] * BUCKET_MINUTES[BUCKET_LONG]).sum()
            if total <= 0 or abs(total - budget) < tolerance:
                break
            scaled = np.clip(short[rows] * (budget / total), 0.0, 1.0)
            scaled_long = np.clip(long_[rows] * (budget / total), 0.0, 1.0)
            # Renormalise any player pushed past certainty, keeping his short/long ratio.
            over = scaled + scaled_long > 1.0
            if over.any():
                mass = (scaled + scaled_long)[over]
                scaled[over] /= mass
                scaled_long[over] /= mass
            if np.allclose(scaled, short[rows]) and np.allclose(scaled_long, long_[rows]):
                break                      # every player is saturated; nothing left to give
            short[rows], long_[rows] = scaled, scaled_long

    out["p_short"], out["p_long"] = short, long_
    out["p_zero"] = np.clip(1.0 - short - long_, 0.0, 1.0)
    out["expected_minutes"] = (
        short * BUCKET_MINUTES[BUCKET_SHORT] + long_ * BUCKET_MINUTES[BUCKET_LONG]
    )
    out["expected_appearance_points"] = short * 1 + long_ * 2
    out["p_appear"] = short + long_
    return out


def apply_availability_gate(
    predictions: pd.DataFrame, chance_of_playing: pd.Series, *, status: pd.Series | None = None
) -> pd.DataFrame:
    """Fold FPL's published availability into the prediction, at inference only.

    `chance_of_playing_next_round` is null when there is no news (treat as fully available)
    and otherwise 0/25/50/75/100. It is already a probability, so it is applied directly as
    a multiplicative gate on playing rather than learned: the archive has no history of this
    field, so a learned version would be train/serve skew.

    Mass removed from the playing buckets goes to `p_zero`, and the two playing buckets keep
    their relative shape — a doubtful player who would have started still starts *if* he is
    fit, so the short/long split should not change.
    """
    gate = pd.to_numeric(chance_of_playing, errors="coerce").fillna(100.0).clip(0, 100) / 100.0
    gate = gate.reindex(predictions.index).fillna(1.0)
    if status is not None:
        # Suspended, injured or otherwise unavailable overrides any stale percentage.
        unavailable = status.reindex(predictions.index).isin(["i", "s", "u", "n"])
        gate = gate.where(~unavailable.fillna(False), 0.0)

    out = predictions.copy()
    for col in ("p_short", "p_long"):
        out[col] = out[col] * gate
    out["p_zero"] = 1.0 - out["p_short"] - out["p_long"]
    out["expected_minutes"] = out["p_short"] * BUCKET_MINUTES[BUCKET_SHORT] + out["p_long"] * BUCKET_MINUTES[BUCKET_LONG]
    out["expected_appearance_points"] = out["p_short"] * 1 + out["p_long"] * 2
    out["p_appear"] = out["p_short"] + out["p_long"]
    return out
