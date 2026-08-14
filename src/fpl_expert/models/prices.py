"""Price changes: who rises, who falls, and what that is worth.

Team value compounds. A manager who holds risers all season carries a squad worth several
million more than one who does not, and that budget buys points nobody attributes to price.
Nothing in this project modelled it until now.

**What drives it.** FPL moves a player's price on net transfers relative to how many people
already own him, not on performance directly — performance only matters through the transfers
it provokes. The archive carries both sides of that (`transfers_in`, `transfers_out`,
`selected`), so the driver is directly observable rather than inferred. Measured over 2024-25:

    net transfers / owners     n        mean delta    P(rise)   P(fall)
    heavy sells (< -15%)      1,444       -0.254       0.006     0.258
    sells (-15% to -5%)       3,827       -0.140       0.005     0.146
    flat (-5% to +5%)        17,685       -0.019       0.009     0.028
    buys (+5% to +15%)        1,872       +0.085       0.085     0.004
    heavy buys (> +15%)       1,598       +0.142       0.135     0.001

Cleanly ordered and monotone, which is what makes an ordered model the right shape here rather
than two independent ones: a player cannot be simultaneously likely to rise and likely to fall,
and fitting those as separate problems allows exactly that contradiction.

**Relative flow is not enough on its own.** FPL's thresholds are absolute, so a net fraction
badly misleads at the bottom of the ownership distribution: 21 net transfers on a player owned
by 32 managers is a fraction of 0.66 and moves nothing whatsoever. A first version fitted on
relative flow alone duly nominated players owned by a few dozen people as the week's most
likely risers. Absolute volume is in the feature set for that reason and is worth more than
everything else combined.

**Why ordered logit rather than a classifier.** Prices move in single increments of £0.1m —
falls outnumber rises roughly three to one, and 92.5% of player-gameweeks see no change at
all. The three outcomes are ordered (fall < same < rise) along one underlying pressure, so a
single linear predictor with two cutpoints expresses the whole thing and cannot produce the
incoherent combination a multiclass model can.

**A caution this project has earned.** A real signal is not automatically an exploitable one.
Ownership bias was measured, consistent and statistically clear, and calibrating on it made
decisions worse. Expected price movement is worth about £0.1m per player per gameweek at the
extremes, against forecast differences of whole points — so the prior should be that this is
a reporting column, not an objective term, until a paired measurement says otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

log = logging.getLogger(__name__)

# Price is stored in tenths of a million; one increment is £0.1m.
PRICE_STEP = 0.1
FALL, SAME, RISE = 0, 1, 2


def build_features(history: pd.DataFrame) -> pd.DataFrame:
    """Per player-gameweek drivers of the NEXT gameweek's price change.

    The target is shifted forward by one gameweek within each player and season, so a row only
    ever contains transfer flow that preceded the move it is predicting. Getting that backwards
    would produce a model that explains price changes from the transfers they caused.
    """
    frame = history.groupby(["season", "GW", "name"], as_index=False).agg(
        value=("value", "max"),
        selected=("selected", "max"),
        transfers_balance=("transfers_balance", "sum"),
        transfers_in=("transfers_in", "sum"),
        transfers_out=("transfers_out", "sum"),
    ).sort_values(["season", "name", "GW"])

    grouped = frame.groupby(["season", "name"])
    frame["next_value"] = grouped["value"].shift(-1)
    frame["delta"] = frame["next_value"] - frame["value"]

    owners = frame["selected"].clip(lower=1)
    frame["net_fraction"] = frame["transfers_balance"] / owners
    frame["churn_fraction"] = (
        frame["transfers_in"] + frame["transfers_out"]
    ) / owners
    # Ownership scale matters on its own: the same net fraction moves a widely-held player
    # more slowly, because FPL's thresholds are absolute rather than proportional.
    frame["log_owners"] = np.log1p(frame["selected"])

    # ABSOLUTE volume, not just relative. A net fraction alone is badly misleading at the
    # bottom of the ownership distribution — 21 net transfers on a player owned by 32 managers
    # is a fraction of 0.66 and moves nothing at all. Measured:
    #
    #     owners        n        P(price changes)
    #     < 1k        15,353         0.0096
    #     1k - 20k    69,667         0.0413
    #     20k - 200k  55,011         0.0696
    #     > 200k      33,179         0.2215
    #
    # Adding these two took walk-forward log loss from 0.2770 to 0.2342 against a base rate
    # of 0.3085 — from 10% better than knowing nothing to 24% better.
    managers = frame.groupby(["season", "GW"])["selected"].transform("sum") / 15.0
    frame["net_per_manager"] = frame["transfers_balance"] / managers.clip(lower=1)
    frame["log_net"] = np.sign(frame["transfers_balance"]) * np.log1p(
        frame["transfers_balance"].abs()
    )

    frame = frame.dropna(subset=["delta"])
    frame["target"] = np.where(frame["delta"] > 0, RISE, np.where(frame["delta"] < 0, FALL, SAME))
    return frame.reset_index(drop=True)


FEATURES = (
    "net_fraction", "churn_fraction", "log_owners", "net_per_manager", "log_net",
)


@dataclass
class PriceModel:
    """Ordered logit over {fall, same, rise}.

    One linear predictor `z` of transfer pressure and two cutpoints. `P(fall) = sigma(c0 - z)`
    and `P(rise) = 1 - sigma(c1 - z)`, which by construction keeps the three probabilities
    ordered and summing to one — a player under buying pressure cannot come out likely to fall.
    """

    coefficients: np.ndarray | None = None
    cutpoints: tuple[float, float] = (-3.0, 3.0)
    features: tuple[str, ...] = FEATURES
    fitted_on: tuple[str, ...] = ()
    means: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES)))
    scales: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURES)))

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[list(self.features)].to_numpy(dtype=float)
        return (raw - self.means) / self.scales

    def fit(self, frame: pd.DataFrame) -> PriceModel:
        design_raw = frame[list(self.features)].to_numpy(dtype=float)
        # Standardised so the two cutpoints and the coefficients are on comparable scales,
        # which is what keeps the unconstrained optimiser well behaved.
        self.means = design_raw.mean(axis=0)
        self.scales = np.where(design_raw.std(axis=0) > 0, design_raw.std(axis=0), 1.0)
        design = (design_raw - self.means) / self.scales
        target = frame["target"].to_numpy(dtype=int)

        def negative_log_likelihood(theta):
            beta = theta[:-2]
            # The gap is exponentiated so the cutpoints can never cross, which would invert
            # the ordering and make the middle category vanish.
            low, gap = theta[-2], np.exp(theta[-1])
            z = design @ beta
            p_fall = expit(low - z)
            p_not_rise = expit(low + gap - z)
            probabilities = np.stack(
                [p_fall, np.clip(p_not_rise - p_fall, 1e-12, None), 1.0 - p_not_rise]
            )
            return -np.log(np.clip(probabilities[target, np.arange(len(target))], 1e-12, None)).sum()

        start = np.concatenate([np.zeros(design.shape[1]), [-2.0, np.log(4.0)]])
        result = minimize(negative_log_likelihood, start, method="L-BFGS-B")
        self.coefficients = result.x[:-2]
        self.cutpoints = (float(result.x[-2]), float(result.x[-2] + np.exp(result.x[-1])))
        if "season" in frame.columns:
            self.fitted_on = tuple(sorted(frame["season"].unique()))
        log.info(
            "price model fitted on %s: coefficients %s cutpoints %.2f/%.2f",
            list(self.fitted_on) or "unknown",
            np.round(self.coefficients, 3), *self.cutpoints,
        )
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """(n, 3) probabilities over fall / same / rise."""
        if self.coefficients is None:
            raise RuntimeError("model is not fitted")
        z = self._design(frame) @ self.coefficients
        p_fall = expit(self.cutpoints[0] - z)
        p_not_rise = expit(self.cutpoints[1] - z)
        return np.column_stack([p_fall, np.clip(p_not_rise - p_fall, 0.0, None), 1.0 - p_not_rise])

    def expected_change(self, frame: pd.DataFrame) -> np.ndarray:
        """Expected price movement in £m for the coming gameweek."""
        probabilities = self.predict_proba(frame)
        return PRICE_STEP * (probabilities[:, RISE] - probabilities[:, FALL])


class NoChangeBaseline:
    """Predicts the base rates and nothing else — the bar any price model must clear.

    92.5% of player-gameweeks see no change, so a model can look accurate while knowing
    nothing. This is what makes accuracy the wrong metric here and log loss the right one.
    """

    def __init__(self) -> None:
        self.rates = np.array([0.05, 0.90, 0.05])

    def fit(self, frame: pd.DataFrame) -> NoChangeBaseline:
        counts = np.bincount(frame["target"].to_numpy(dtype=int), minlength=3)
        self.rates = counts / counts.sum()
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(self.rates, (len(frame), 1))


def team_value_path(
    squad_ids, forecasts_by_gw: dict[int, pd.DataFrame], model: PriceModel
) -> pd.DataFrame:
    """Expected value of a held squad over a planning horizon.

    Reported rather than optimised. What it answers is whether a squad is quietly gaining or
    bleeding value, which is a slow effect a single gameweek cannot show — and which no other
    column in the system exposes at all.
    """
    rows = []
    running = 0.0
    for gw, frame in sorted(forecasts_by_gw.items()):
        held = frame[frame["player_id"].isin(squad_ids)]
        if held.empty:
            continue
        change = float(model.expected_change(held).sum())
        running += change
        rows.append({"gw": gw, "expected_change": change, "cumulative": running})
    return pd.DataFrame(rows)
