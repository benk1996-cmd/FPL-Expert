"""Match model: team strengths, scoreline distributions, and clean-sheet probabilities.

Clean sheets and goals conceded are properties of the *match*, not of the player. A
defender's points depend on whether his team keeps a clean sheet, which depends on the
opponent's attack and his own team's defence. So the match is modelled first and player
outcomes are attributed from it, rather than trying to learn "clean sheet probability" per
player from thin, heavily-confounded data.

Two independent estimates of the same quantity are produced and blended:

1. **Dixon-Coles.** Goals are near-Poisson with team attack and defence strengths plus a
   home advantage, fitted by weighted maximum likelihood with exponential time decay.
   Independent Poissons famously misprice low-scoring draws, so the Dixon-Coles `tau`
   correction adjusts the 0-0, 1-0, 0-1 and 1-1 cells — exactly the scorelines that decide
   clean sheets, which makes it load-bearing here rather than cosmetic.

2. **The market.** Opening odds are inverted back into the goal expectations that produced
   them, by finding the (lambda_home, lambda_away) whose scoreline grid best reproduces the
   quoted 1X2 and over/under 2.5 probabilities. The bookmaker line already aggregates
   information we cannot see, so this is usually the stronger signal on its own.

Only *opening* odds are used. Closing odds postdate the FPL deadline and are unreachable
from the feature path by construction (see `data/snapshot.py`).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

log = logging.getLogger(__name__)

MAX_GOALS = 10          # P(11+ goals) is ~1e-7 at Premier League scoring rates
DEFAULT_HALF_LIFE_DAYS = 240.0
MIN_TAU = 1e-6


def dixon_coles_tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray,
                    rho: float) -> np.ndarray:
    """Low-score dependence correction.

    Independent Poissons under-predict 0-0 and 1-1 and over-predict 1-0 and 0-1. Those are
    precisely the clean-sheet scorelines, so leaving this out biases the single quantity
    defenders and goalkeepers are scored on.
    """
    tau = np.ones_like(lh, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1.0 - lh * la * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lh * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + la * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
    return np.clip(tau, MIN_TAU, None)


def score_grid(lambda_home: float, lambda_away: float, rho: float = 0.0,
               max_goals: int = MAX_GOALS) -> np.ndarray:
    """Joint distribution over scorelines. `grid[x, y]` is P(home x, away y)."""
    home = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    grid = np.outer(home, away)

    if rho:
        xs, ys = np.meshgrid(np.arange(max_goals + 1), np.arange(max_goals + 1), indexing="ij")
        grid = grid * dixon_coles_tau(
            xs, ys, np.full(xs.shape, lambda_home), np.full(xs.shape, lambda_away), rho
        )
    return grid / grid.sum()


def outcome_probabilities(grid: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win) from a scoreline grid."""
    home = float(np.tril(grid, -1).sum())   # home goals > away goals
    draw = float(np.trace(grid))
    return home, draw, 1.0 - home - draw


def clean_sheet_probabilities(grid: np.ndarray) -> tuple[float, float]:
    """(home keeps a clean sheet, away keeps a clean sheet).

    A team keeps a clean sheet when the OPPONENT fails to score, so the home clean sheet is
    the column where away goals are zero.
    """
    return float(grid[:, 0].sum()), float(grid[0, :].sum())


def goals_conceded_distribution(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(home conceded pmf, away conceded pmf) — needed for the -1 per 2 conceded rule."""
    return grid.sum(axis=0), grid.sum(axis=1)


def probability_over(grid: np.ndarray, line: float = 2.5) -> float:
    """P(total goals > line)."""
    xs, ys = np.meshgrid(np.arange(grid.shape[0]), np.arange(grid.shape[1]), indexing="ij")
    return float(grid[(xs + ys) > line].sum())


# --- Dixon-Coles ----------------------------------------------------------


@dataclass
class DixonColes:
    """Weighted-MLE team strength model with a home advantage and low-score correction."""

    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    home_advantage: float = 0.0
    rho: float = 0.0

    def _weights(self, dates: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
        age_days = (as_of - pd.to_datetime(dates)).dt.total_seconds().to_numpy() / 86400
        return 0.5 ** (np.clip(age_days, 0, None) / self.half_life_days)

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> DixonColes:
        """Fit on matches played strictly before `as_of`.

        Past results are legitimate training data — they were known before the deadline we
        are predicting. What must never appear is the result of the match being predicted,
        which the `as_of` cut enforces.
        """
        df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
        df["date"] = pd.to_datetime(df["date"])
        as_of = pd.Timestamp(as_of) if as_of is not None else df["date"].max() + pd.Timedelta(days=1)
        df = df[df["date"] < as_of]
        if df.empty:
            raise ValueError("no matches before as_of")

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        index = {team: i for i, team in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home_team"].map(index).to_numpy()
        ai = df["away_team"].map(index).to_numpy()
        x = df["home_goals"].to_numpy(float)
        y = df["away_goals"].to_numpy(float)
        w = self._weights(df["date"], as_of)

        def unpack(params):
            att = np.append(params[: n - 1], -params[: n - 1].sum())   # sum-to-zero
            dfn = np.append(params[n - 1 : 2 * n - 2], -params[n - 1 : 2 * n - 2].sum())
            return att, dfn, params[-3], params[-2], params[-1]

        def negative_log_likelihood(params):
            att, dfn, intercept, home_adv, rho = unpack(params)
            lh = np.exp(intercept + att[hi] - dfn[ai] + home_adv)
            la = np.exp(intercept + att[ai] - dfn[hi])
            tau = dixon_coles_tau(x, y, lh, la, rho)
            ll = x * np.log(lh) - lh + y * np.log(la) - la + np.log(tau)
            return -float((w * ll).sum())

        start = np.concatenate([
            np.zeros(n - 1), np.zeros(n - 1), [np.log(1.4)], [0.25], [-0.03]
        ])
        bounds = [(-3, 3)] * (2 * n - 2) + [(-2, 2), (-1, 1), (-0.15, 0.15)]
        result = minimize(negative_log_likelihood, start, method="L-BFGS-B", bounds=bounds)

        att, dfn, self.intercept, self.home_advantage, self.rho = unpack(result.x)
        self.attack = dict(zip(self.teams, att, strict=True))
        self.defence = dict(zip(self.teams, dfn, strict=True))
        log.info(
            "fitted on %d matches: home_adv=%.3f rho=%.3f converged=%s",
            len(df), self.home_advantage, self.rho, result.success,
        )
        return self

    def lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Expected goals for (home, away).

        A team absent from the fit — newly promoted, so with no Premier League record — is
        treated as exactly league-average (attack and defence of zero) rather than dropped.
        Same principle as the cold-start prior in the rate model: an uninformative estimate
        beats no estimate, and the blend with the market corrects it where the market knows
        better.
        """
        ah, dh = self.attack.get(home_team, 0.0), self.defence.get(home_team, 0.0)
        aa, da = self.attack.get(away_team, 0.0), self.defence.get(away_team, 0.0)
        return (
            float(np.exp(self.intercept + ah - da + self.home_advantage)),
            float(np.exp(self.intercept + aa - dh)),
        )

    def predict_grid(self, home_team: str, away_team: str) -> np.ndarray:
        lh, la = self.lambdas(home_team, away_team)
        return score_grid(lh, la, self.rho)


# --- inverting the market -------------------------------------------------


def lambdas_from_odds(
    p_home: float, p_draw: float, p_away: float,
    p_over25: float | None = None, rho: float = 0.0,
) -> tuple[float, float]:
    """Recover the goal expectations implied by a set of de-vigged odds.

    1X2 alone pins the balance between the teams but not the total — "tight game between two
    good attacks" and "tight game between two poor ones" imply the same 1X2 and very
    different clean-sheet probabilities. The over/under 2.5 line resolves that, which is why
    it is worth carrying.
    """
    targets = [p_home, p_draw, p_away]

    def loss(log_lambdas):
        lh, la = np.exp(log_lambdas)
        grid = score_grid(lh, la, rho)
        predicted = list(outcome_probabilities(grid))
        error = sum((p - t) ** 2 for p, t in zip(predicted, targets, strict=True))
        if p_over25 is not None and not np.isnan(p_over25):
            error += (probability_over(grid, 2.5) - p_over25) ** 2
        return error

    result = minimize(
        loss, np.log([1.5, 1.2]), method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500},
    )
    return tuple(float(v) for v in np.exp(result.x))


def blend_lambdas(
    model: tuple[float, float], market: tuple[float, float], market_weight: float = 0.7
) -> tuple[float, float]:
    """Geometric blend of two goal-expectation estimates.

    Blending in log space keeps the result positive and treats the estimates
    multiplicatively, which matches how attack and defence strengths combine.
    """
    w = float(np.clip(market_weight, 0.0, 1.0))
    return tuple(
        float(np.exp(w * np.log(mk) + (1 - w) * np.log(md)))
        for md, mk in zip(model, market, strict=True)
    )


def match_features(
    grid: np.ndarray, home_team: str, away_team: str, lambda_home: float, lambda_away: float
) -> list[dict]:
    """Flatten a scoreline grid into the two team-level rows the player models consume."""
    cs_home, cs_away = clean_sheet_probabilities(grid)
    conceded_home, conceded_away = goals_conceded_distribution(grid)
    p_home, p_draw, p_away = outcome_probabilities(grid)
    return [
        {
            "team": home_team, "opponent": away_team, "is_home": True,
            "expected_goals_for": lambda_home, "expected_goals_against": lambda_away,
            "p_clean_sheet": cs_home, "p_win": p_home, "p_draw": p_draw, "p_lose": p_away,
            "p_concede_0": conceded_home[0], "p_concede_1": conceded_home[1],
            "p_concede_2plus": float(conceded_home[2:].sum()),
            "expected_conceded_penalty": _conceded_penalty(conceded_home),
        },
        {
            "team": away_team, "opponent": home_team, "is_home": False,
            "expected_goals_for": lambda_away, "expected_goals_against": lambda_home,
            "p_clean_sheet": cs_away, "p_win": p_away, "p_draw": p_draw, "p_lose": p_home,
            "p_concede_0": conceded_away[0], "p_concede_1": conceded_away[1],
            "p_concede_2plus": float(conceded_away[2:].sum()),
            "expected_conceded_penalty": _conceded_penalty(conceded_away),
        },
    ]


def _conceded_penalty(conceded_pmf: np.ndarray) -> float:
    """Expected goals-conceded points for a GK or defender: -1 per 2 conceded, rounded down."""
    goals = np.arange(len(conceded_pmf))
    return float(-(goals // 2 * conceded_pmf).sum())


def build_match_forecasts(
    fixtures: pd.DataFrame,
    model: DixonColes,
    *,
    market_weight: float = 0.8,
) -> pd.DataFrame:
    """Team-level forecasts for a set of fixtures — the input the player models consume.

    `fixtures` needs `home_team`/`away_team` as FPL short names, plus optionally the
    `p_*_open` columns. Where odds are absent (early season, or a fixture the book has not
    priced) the Dixon-Coles estimate is used alone, which is why the blend is kept rather
    than deferring entirely to the market.
    """
    rows = []
    for fixture in fixtures.itertuples():
        home, away = fixture.home_team, fixture.away_team
        lambdas = model.lambdas(home, away)
        source = "model"

        p_home = getattr(fixture, "p_home_open", np.nan)
        if not (p_home is None or (isinstance(p_home, float) and np.isnan(p_home))):
            market = lambdas_from_odds(
                p_home, fixture.p_draw_open, fixture.p_away_open,
                getattr(fixture, "p_over25_open", None), rho=model.rho,
            )
            lambdas = blend_lambdas(lambdas, market, market_weight)
            source = "blend"

        grid = score_grid(*lambdas, model.rho)
        for team_row in match_features(grid, home, away, *lambdas):
            team_row.update({
                "gw": getattr(fixture, "event", getattr(fixture, "gw", None)),
                "kickoff_time": getattr(fixture, "kickoff_time", None),
                "source": source,
                # Carried so that consumers whose player rows are already per-FIXTURE can
                # join on it. Joining on team alone is a cross product in a double gameweek.
                "fixture": getattr(fixture, "fixture", None),
            })
            rows.append(team_row)
    return pd.DataFrame(rows)


# --- rolling-origin evaluation --------------------------------------------


def rolling_lambdas(
    matches: pd.DataFrame,
    *,
    refit_days: int = 14,
    min_train_days: int = 540,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """Out-of-sample goal expectations for every match, from both sources.

    The model is refitted every `refit_days` on matches strictly before the block, which is
    how it would run in production. Returns raw lambdas rather than blended probabilities so
    a blend weight can be swept afterwards without refitting.
    """
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    start = df["date"].min() + pd.Timedelta(days=min_train_days)
    blocks = pd.date_range(start, df["date"].max() + pd.Timedelta(days=refit_days),
                           freq=f"{refit_days}D")

    rows = []
    for block_start, block_end in itertools.pairwise(blocks):
        block = df[(df["date"] >= block_start) & (df["date"] < block_end)]
        if block.empty:
            continue
        model = DixonColes(half_life_days=half_life_days).fit(df, as_of=block_start)

        for match in block.itertuples():
            dc_home, dc_away = model.lambdas(match.home_team, match.away_team)
            market = (np.nan, np.nan)
            if not np.isnan(getattr(match, "p_home_open", np.nan)):
                market = lambdas_from_odds(
                    match.p_home_open, match.p_draw_open, match.p_away_open,
                    getattr(match, "p_over25_open", None), rho=model.rho,
                )
            rows.append({
                "date": match.date, "season": match.season,
                "home_team": match.home_team, "away_team": match.away_team,
                "home_goals": match.home_goals, "away_goals": match.away_goals,
                "dc_home": dc_home, "dc_away": dc_away,
                "mkt_home": market[0], "mkt_away": market[1],
                "rho": model.rho,
            })
    return pd.DataFrame(rows)


def score_blend(predictions: pd.DataFrame, market_weight: float) -> pd.DataFrame:
    """Apply a blend weight and derive outcome and clean-sheet probabilities."""
    out = []
    for row in predictions.itertuples():
        if np.isnan(row.mkt_home):
            lh, la = row.dc_home, row.dc_away
        else:
            lh, la = blend_lambdas(
                (row.dc_home, row.dc_away), (row.mkt_home, row.mkt_away), market_weight
            )
        grid = score_grid(lh, la, row.rho)
        p_home, p_draw, p_away = outcome_probabilities(grid)
        cs_home, cs_away = clean_sheet_probabilities(grid)
        out.append({
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "cs_home": cs_home, "cs_away": cs_away,
            "lambda_home": lh, "lambda_away": la,
        })
    return pd.concat([predictions.reset_index(drop=True), pd.DataFrame(out)], axis=1)
