"""Season simulator: replay a whole season, making real decisions against real outcomes.

This is the only honest measure of the system. Component-level metrics (log loss on minutes,
calibration on clean sheets) say whether a model is well fitted; they do not say whether the
squad it produces scores points. Those can diverge badly, because FPL decisions are made
under constraints — a forecast that is 5% better on average but wrong about which £12m
midfielder to captain can be worth less than one that is worse everywhere and right there.

Two things it is built to answer:

1. **How good is the whole system?** Season points against benchmarks, not against a
   held-out loss.
2. **Which components deserve more work?** `ablate` degrades one component at a time and
   remeasures. A component whose removal costs nothing is not worth deepening, however
   crude it looks; one whose removal is expensive is where effort belongs. This is why the
   simulator was built before the sprint components were deepened.

Transfers are chosen by the real MILP (`optimise/transfers.py`), valued on a discounted
multi-gameweek horizon INCLUDING the captaincy premium. `transfer_policy="greedy"` restores
the simpler policy it replaced.

**Both claims this docstring used to make for that horizon are withdrawn (2026-08-12).** It
said wiring in the MILP horizon was worth +213 season points, and that horizon and captaincy
together were worth +111/+137/+49/+207 — "positive in every season, which no other change
tested here has managed". Both were measured while `horizon_valuations` assembled its window
from forecasts built AFTER the decision. On point-in-time forecasts the same comparison is:

    season      myopic     horizon 6      difference
    2023-24      2229        2340            +111
    2024-25      2287        2113            -174
    2025-26      2055        2070             +15
                                        mean   -16

The horizon is not currently distinguishable from not having one, and the consistency that
made the old claim persuasive was an artefact of every season's horizon being assembled the
same wrong way. It stays at `DEFAULT_HORIZON = 6` because nothing shows it is worse either —
an unproven default, not a result. `horizon=0` restores the myopic policy. See DECISIONS.

Earlier rank figures are withdrawn separately: they were computed against a simulated field
the double-gameweek fixes of 2026-08-10 revealed to be far too weak to rank against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import load_scoring_rules
from ..optimise.chips import (
    bench_boost_value,
    chip_windows,
    free_hit_value,
    should_play,
    triple_captain_value,
    wildcard_value,
)
from ..optimise.risk import effective_points
from ..optimise.squad import select_squad
from ..optimise.transfers import horizon_points, recommend_transfers

log = logging.getLogger(__name__)

# Bench Boost and Triple Captain. RE-DERIVED TWICE, and Free Hit lost its place the second
# time — the honest sequence is worth keeping because each instrument overturned the last.
#
# 1. "+16.2 [+14.5, +17.9] over 600 draws, 76% of them" came from `repeat_sim`, which
#    resamples outcomes from the model's OWN pmf. `rescore` is linear in the draws, so the
#    expected difference is just the model's forecast of which chip set it prefers — a
#    quantity the planner chose its decisions to maximise. Monte-Carlo error on a tautology.
#
# 2. `bootstrap_realised` grades on what actually happened, resampling gameweeks in blocks of
#    four so fixture runs and double gameweeks survive. On that instrument Free Hit came out
#    at +7.2 [+5.6, +8.9] with a 0.52 win rate — "survives, on half the evidence claimed".
#    But those decisions were still made against a horizon reading the future, and chip
#    TIMING is exactly what a frozen forward view distorts.
#
# 3. Same instrument, point-in-time forecasts (2026-08-12). Free Hit reverses:
#
#     variant vs BB+TC    pooled diff    95% CI            wins   realised mean (n=3)
#     BB+TC                    0.0                          —           2179
#     BB+TC+free hit          -5.5    [-6.9, -4.0]         41%          2174
#     all chips              -27.7    [-33.4, -22.1]       40%          2150
#     BB+TC+wildcard         -28.2    [-33.5, -22.8]       41%          2150
#     no chips               -47.1    [-48.8, -45.5]        0%          2137
#
# **Playing chips is settled; WHICH chips is not.** `no chips` loses in all three seasons at a
# 0.00 win rate, so the category earns its place beyond doubt. But Free Hit is -11.4 / -16.3 /
# +11.3 per season — the sign flips, exactly as it flips for wildcard (-117.8 / +58.5 / -25.2,
# a 176-point swing between seasons). No chip set is resolvable at n=3.
#
# So the default reverts to the SIMPLER set. The positive evidence that put Free Hit here is
# gone and the pooled difference now runs against it; with the per-season signs unresolved,
# fewer moving parts breaks the tie. This is a retreat to the defensible, not a new finding.
DEFAULT_CHIPS = ("bench_boost", "triple_captain")

# How far ahead a transfer is valued. Measured against a myopic policy, not assumed.
DEFAULT_HORIZON = 6

# How many gameweeks a decision week needs forecast from its own vantage point, when the
# horizon is built point-in-time. `horizon_valuations` covers `gw .. gw + horizon - 1`; the
# chip planner asks one further, up to `gw + horizon` inclusive. Tied to the horizon rather
# than written as a literal, because a window one gameweek short fails silently: the planner
# would simply see no future opportunity and play its chips early.
LOOKAHEAD_SPAN = DEFAULT_HORIZON + 1


@dataclass
class SeasonResult:
    gameweeks: pd.DataFrame
    total_points: float
    label: str = ""
    meta: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)

    def summary(self) -> pd.Series:
        gw = self.gameweeks
        return pd.Series({
            "label": self.label,
            "total_points": round(self.total_points, 1),
            "points_per_gw": round(self.total_points / max(len(gw), 1), 2),
            "transfers": int(gw["transfers_made"].sum()),
            "hits_taken": int(gw["hit_cost"].sum() / 4),
            "captain_points": round(gw["captain_points"].sum(), 1),
        })


def _legal_squad(
    forecasts: pd.DataFrame, rules: dict, budget: float, points_col: str,
    flexibility_weight: float = 0.0,
):
    squad_rules = rules["squad"]
    return select_squad(
        forecasts,
        budget=budget,
        squad_quota=squad_rules["positions"],
        formation=squad_rules["formation"],
        max_per_club=squad_rules["max_per_club"],
        points_col=points_col,
        flexibility_weight=flexibility_weight,
        # A horizon column already prices the armband through `captaincy_uplift`; doubling
        # the captain on top of it counts the premium twice.
        double_captain=points_col == "expected_points",
    )


def pick_xi(held: pd.DataFrame, rules: dict, points_col: str) -> tuple:
    """Best legal XI from whoever is actually available, plus the ordered squad.

    Deliberately NOT `select_squad`: that enforces a full 2/5/5/3 fifteen, and in a BLANK
    gameweek some of your players have no fixture and therefore no row at all. Demanding a
    legal fifteen there is infeasible — but you still field a team, so the XI is filled to
    the formation minimums and topped up by forecast.
    """
    squad_rules = rules["squad"]
    formation = squad_rules["formation"]
    held = held.sort_values(points_col, ascending=False)

    starters: list = []
    counts = dict.fromkeys(formation, 0)
    for position, limits in formation.items():
        take = held[held["position"] == position].head(limits["min"])
        starters.extend(take.index.tolist())
        counts[position] = len(take)

    for row in held.drop(index=starters).itertuples():
        if len(starters) >= squad_rules["starting_xi"]:
            break
        limits = formation.get(row.position, {})
        if counts.get(row.position, 0) < limits.get("max", 0):
            starters.append(row.Index)
            counts[row.position] += 1

    return held, held.loc[starters]


def score_gameweek(
    squad_ids: set,
    captain_id,
    gw_frame: pd.DataFrame,
    rules: dict,
    points_col: str = "expected_points",
    chip: str | None = None,
) -> dict:
    """Score a held squad against realised points, picking the XI by forecast.

    The XI is chosen on FORECAST, not on outcome — choosing it with hindsight would inflate
    every result and is the easiest way to build a backtest that cannot be reproduced live.
    """
    held = gw_frame[gw_frame["player_id"].isin(squad_ids)].copy()
    if held.empty:
        return {"points": 0.0, "captain_points": 0.0, "starters": 0}

    held, xi = pick_xi(held, rules, points_col)
    captain_row = xi[xi["player_id"] == captain_id]
    if captain_row.empty:
        # `pick_xi` returns the XI in FORMATION order, so `head(1)` was the goalkeeper rather
        # than the best starter. The test covering this asserted only that the armband scored
        # something, so it passed while handing the captaincy to a keeper.
        captain_row = xi.nlargest(1, points_col)

    # The vice-captain. FPL passes the armband on when the captain does not play at all, and
    # the simulator used to double his zero — a second unlisted conservative bias alongside
    # the missing auto-substitutions.
    captain_played = (
        "actual_minutes" not in xi.columns
        or captain_row.empty
        or float(captain_row["actual_minutes"].fillna(0).iloc[0]) > 0
    )
    if not captain_played:
        vice = xi[
            (xi["player_id"] != captain_row["player_id"].iloc[0])
            & (xi["actual_minutes"].fillna(0) > 0)
        ]
        if not vice.empty:
            captain_row = vice.nlargest(1, points_col)

    captain_points = float(captain_row["actual_points"].fillna(0).iloc[0])

    # Bench Boost makes the whole 15 score, not just the eleven.
    scoring = held if chip == "bench_boost" else xi
    base = float(scoring["actual_points"].fillna(0).sum())
    # The armband already grants a second copy; Triple Captain adds a third.
    multiplier = 2 if chip == "triple_captain" else 1

    return {
        "points": base + multiplier * captain_points,
        "captain_points": multiplier * captain_points,
        "starters": len(xi),
        "chip": chip,
        # Everything needed to rescore this gameweek against a DIFFERENT set of outcomes
        # without re-deciding anything. See `rescore`.
        "scoring_ids": scoring["player_id"].tolist(),
        "captain_id": captain_row["player_id"].iloc[0],
        "captain_multiplier": multiplier,
    }


def _choose_captain(selector, gw, held, points_col, points_so_far):
    """The armband: highest expected points, unless a selector overrides it.

    A selector is handed the running season total as well as the week, because a rank-aware
    choice depends on whether you are protecting a lead or chasing — the one thing a fixed
    ownership discount could never express.
    """
    if held.empty:
        return None
    if selector is None:
        return held.nlargest(1, points_col)["player_id"].iloc[0]
    return selector(gw, held, points_col, points_so_far)


def horizon_valuations(
    forecasts: pd.DataFrame,
    *,
    horizon: int = DEFAULT_HORIZON,
    decay: float = 0.84,
    captaincy_weight: float = 0.0,
    points_col: str = "expected_points",
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
    smoothing: float = 0.0,
) -> dict[int, pd.DataFrame]:
    """Forward-looking player value for a decision made at each gameweek.

    A transfer is a durable change, so valuing it on the single gameweek in front of you
    over-trades. Each entry looks `horizon` gameweeks ahead from that decision point.

    **Without `lookahead` the window is not point-in-time.** `forecasts` holds one row per
    player-gameweek, and GW15's row was built with rates, form and a team model as of GW15 —
    none of which existed when the GW10 transfer was made. No realised outcome is read, so
    solve-once/rescore-many stays valid, but a forecast built from post-decision inputs is
    still future information and the numbers it produces are not reproducible live.

    `lookahead[decision_gw][target_gw]` supplies the honest alternative: GW15's fixtures
    forecast from GW10's state of the league, built by `historical_forecast.forecast_horizon`.
    Decision weeks it does not cover fall back to the flat frame, which is why it is checked
    per gameweek rather than once.

    `smoothing` damps how fast the valuation is allowed to move between decision weeks — see
    `_smooth_across_decisions`. Off by default; it is a belief about revisions, not a fact.
    """
    gameweeks = sorted(forecasts["gw"].unique())
    by_gw = {gw: forecasts[forecasts["gw"] == gw] for gw in gameweeks}

    valuations = {}
    for gw in gameweeks:
        forward = (lookahead or {}).get(gw)
        if forward:
            window = {g: f for g, f in forward.items() if gw <= g < gw + horizon}
        else:
            window = {g: by_gw[g] for g in gameweeks if gw <= g < gw + horizon}
        valuations[gw] = horizon_points(
            window, decay=decay, points_col=points_col, captaincy_weight=captaincy_weight
        )
    return _smooth_across_decisions(valuations, smoothing) if smoothing else valuations


def _smooth_across_decisions(
    valuations: dict[int, pd.DataFrame], halflife: float
) -> dict[int, pd.DataFrame]:
    """Average each player's forward valuation with what it was in earlier decision weeks.

    A point-in-time horizon MOVES from week to week. That movement is what makes it honest —
    and it is also what a re-optimiser chases. Measured on 2024-25, the leaky horizon's
    week-to-week rank stability was 0.9932, effectively frozen; the honest one is 0.9624,
    and transfers rose from 49 to 64 with hits from 12 to 27.

    Smoothing damps the revisions without adding information: the exponential mean at GW10
    is built only from GW1-10, so it is as available live as the raw value. This is a
    statement about how much to believe one week's revision, not about the future.

    Deliberately applied to the VALUATION rather than to the forecast. The weekly forecast
    is what the XI and armband are chosen on and should track the latest news exactly; it is
    only the durable multi-week decision that benefits from a longer memory.
    """
    frames = []
    for as_of, table in sorted(valuations.items()):
        block = table.copy()
        block["_as_of"] = as_of
        frames.append(block)

    combined = pd.concat(frames, ignore_index=True).sort_values(["player_id", "_as_of"])
    combined["horizon_points"] = (
        combined.groupby("player_id")["horizon_points"]
        .ewm(halflife=halflife, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    return {
        int(as_of): block.drop(columns="_as_of").reset_index(drop=True)
        for as_of, block in combined.groupby("_as_of", sort=True)
    }


def simulate_season(
    forecasts: pd.DataFrame,
    *,
    label: str = "model",
    points_col: str = "expected_points",
    transfer_threshold: float = 0.6,
    flexibility_weight: float = 0.0,
    horizon: int = DEFAULT_HORIZON,
    decay: float = 0.84,
    captaincy_weight: float = 1.0,
    lambda_rank: float = 0.0,
    transfer_policy: str = "milp",
    max_transfers: int = 2,
    use_chips: bool = False,
    allowed_chips: tuple[str, ...] | None = DEFAULT_CHIPS,
    captain_selector=None,
    rules: dict | None = None,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
    smoothing: float = 0.0,
    hit_bar: float | None = None,
) -> SeasonResult:
    """Replay a season: pick an opening squad, then one transfer a gameweek when worth it.

    `forecasts` carries every gameweek's per-player forecast AND realised points, with the
    forecast columns built point-in-time upstream.

    With `horizon > 0`, transfer and squad decisions are made on forward-looking value
    (optionally including the captaincy premium) instead of the gameweek in front of them.
    Scoring is always against realised points, whichever valuation drives the decisions.

    `lookahead` makes that forward valuation point-in-time — see `horizon_valuations`. It is
    optional because the ablation and the unit tests operate on a flat forecast table, but a
    headline season total quoted without it is not reproducible live.

    `hit_bar` is the forecast gain a transfer must clear per -4 hit INSIDE the optimiser, as
    distinct from the -4 actually deducted when scoring, which never changes. They are the
    same number only if the forecast margin is unbiased, and it is not: across 111 transfer
    decisions the realised return regresses on the forecast gain with slope 0.436, stable in
    every season. A move forecast to gain 14 gains about 9.4. Default None keeps the nominal 4.
    """
    rules = rules or load_scoring_rules()
    budget = rules["squad"]["budget"]
    gameweeks = sorted(forecasts["gw"].unique())

    decision_col = points_col
    valuations: dict[int, pd.DataFrame] = {}
    if horizon:
        valuations = horizon_valuations(
            forecasts, horizon=horizon, decay=decay,
            captaincy_weight=captaincy_weight, points_col=points_col,
            lookahead=lookahead, smoothing=smoothing,
        )
        decision_col = "horizon_points"

    def frame_for(gw: int, as_of: int | None = None) -> pd.DataFrame:
        """Gameweek `gw`'s decision frame, optionally as it looked at an EARLIER week.

        `as_of` is what the chip planner needs when it looks forward: judging at GW12 whether
        to hold Bench Boost for GW17 must use GW12's view of GW17, not GW17's own. Returned
        without a horizon column, because a chip is valued on the single week it is played in.
        """
        if as_of is not None and gw != as_of:
            forward = (lookahead or {}).get(as_of, {}).get(gw)
            if forward is not None:
                return forward
        frame = forecasts[forecasts["gw"] == gw]
        if not valuations:
            return frame
        merged = frame.merge(valuations[gw], on="player_id", how="left")
        merged["horizon_points"] = merged["horizon_points"].fillna(merged[points_col])
        if lambda_rank and "eo_pct" in merged.columns:
            # Score against the FIELD, not in absolute terms: points from a player the
            # field already owns barely move your rank, because you rise together.
            merged["horizon_points"] = effective_points(
                merged["horizon_points"], merged["eo_pct"], lambda_rank
            )
        return merged

    first = frame_for(gameweeks[0])
    solution = _legal_squad(first, rules, budget, decision_col, flexibility_weight)
    squad = set(solution.squad["player_id"])
    bank = budget - solution.total_cost
    # What each held player was bought for, so selling price can follow FPL's rule rather
    # than assuming the market price is recoverable.
    purchase = dict(zip(solution.squad["player_id"], solution.squad["price"], strict=True))
    free_transfers = 1
    transfer_rules = rules.get("transfers", {})
    max_banked = transfer_rules.get("max_banked", 5)
    hit_cost = transfer_rules.get("hit_cost", -4)

    planner = (
        ChipPlanner(rules, gameweeks, horizon or 6, allowed_chips) if use_chips else None
    )

    rows, trace = [], []
    for gw in gameweeks:
        frame = frame_for(gw)
        transfers, hits = 0, 0
        moves = None
        if gw != gameweeks[0]:
            before = set(squad)
            if transfer_policy == "milp":
                squad, bank, transfers, hits, plan = _milp_transfer(
                    squad, bank, frame, rules, decision_col, free_transfers, max_transfers,
                    purchase, hit_bar,
                )
                if plan is not None:
                    moves = {
                        "gw": gw,
                        "in": plan.transfers_in["player_id"].tolist(),
                        "out": plan.transfers_out["player_id"].tolist(),
                        "forecast_gain": plan.gain,
                        "hits": plan.hits,
                    }
            else:
                squad, bank, transfers = _maybe_transfer(
                    squad, bank, frame, rules, decision_col, transfer_threshold, purchase
                )
            purchase = _update_purchases(purchase, before, squad, frame)
            # One free transfer earned per gameweek, banked up to the configured maximum.
            free_transfers = min(free_transfers - transfers + 1, max_banked)
            free_transfers = max(free_transfers, 1)
        hit = hits * abs(hit_cost) if hits else 0

        held = frame[frame["player_id"].isin(squad)]
        # The armband is decided on THIS gameweek's forecast, not the horizon: you captain
        # for the week in front of you, whatever the transfer was justified by.
        #
        # For a TOTAL POINTS objective the argmax below is not a heuristic, it is optimal:
        # the armband adds a second copy of the captain's score, so the objective is linear
        # and the highest mean wins whatever the shape of the distribution. `captain_selector`
        # exists for the rank objective, where that stops being true — see
        # `optimise/rank_objective.py`.
        captain_id = _choose_captain(
            captain_selector, gw, held, points_col, float(sum(r["points"] for r in rows))
        )

        chip = None
        if planner is not None:
            chip, reason = planner.decide(gw, squad, bank, frame_for, rules, points_col)
            if chip:
                log.info("GW%d: playing %s (%s)", gw, chip, reason)
            if chip in {"wildcard", "free_hit"}:
                # Both grant unlimited transfers, funded by the whole squad's value. The
                # wildcard's squad persists; the free hit's is handed back afterwards.
                previous, previous_bank, previous_purchase = squad, bank, purchase
                # Funded by SELLING the squad, so at selling prices — a rebuild does not get
                # to spend paper gains it could not realise.
                budget_now = bank + float(selling_prices(held, purchase).sum())
                try:
                    rebuilt = _legal_squad(
                        frame, rules, budget_now,
                        points_col if chip == "free_hit" else decision_col,
                        flexibility_weight,
                    )
                except RuntimeError:
                    # A gameweek blank enough that no legal fifteen exists is exactly when a
                    # free hit gets played, and an unguarded solve crashed the whole season
                    # rather than declining the chip.
                    log.warning("GW%d: %s declined — no legal squad available", gw, chip)
                    chip, rebuilt = None, None
                if rebuilt is None:
                    scored = score_gameweek(squad, captain_id, frame, rules, points_col)
                    rows.append({
                        "gw": gw, "points": scored["points"] - hit,
                        "captain_points": scored["captain_points"],
                        "transfers_made": transfers, "hit_cost": hit,
                        "squad_size": len(squad), "free_transfers": free_transfers, "chip": "",
                    })
                    trace.append({
                        "gw": gw, "hit": hit, "chip": "",
                        "scoring_ids": scored.get("scoring_ids", []),
                        "captain_id": scored.get("captain_id"),
                        "captain_multiplier": scored.get("captain_multiplier", 1),
                        "squad_ids": sorted(squad),
                        "moves": moves,
                    })
                    continue
                squad = set(rebuilt.squad["player_id"])
                bank = budget_now - rebuilt.total_cost
                # A wildcard squad is genuinely bought, so every player's basis resets to
                # what was paid for him now.
                purchase = dict(
                    zip(rebuilt.squad["player_id"], rebuilt.squad["price"], strict=True)
                )
                held = frame[frame["player_id"].isin(squad)]
                # Re-picked from the rebuilt squad on expected points, NOT through the
                # selector: a rank-aware selector carries per-gameweek state (the field's
                # running total) and calling it twice in one week would advance that twice.
                captain_id = (
                    held.nlargest(1, points_col)["player_id"].iloc[0]
                    if not held.empty else None
                )
                scored = score_gameweek(squad, captain_id, frame, rules, points_col)
                if chip == "free_hit":
                    # Handed straight back, basis and all — a free hit is a loan, not a sale.
                    squad, bank, purchase = previous, previous_bank, previous_purchase
            else:
                scored = score_gameweek(squad, captain_id, frame, rules, points_col, chip)
        else:
            scored = score_gameweek(squad, captain_id, frame, rules, points_col)

        rows.append({
            "gw": gw, "points": scored["points"] - hit,
            "captain_points": scored["captain_points"],
            "transfers_made": transfers, "hit_cost": hit, "squad_size": len(squad),
            "free_transfers": free_transfers,
            "chip": chip or "",
        })
        trace.append({
            "gw": gw, "hit": hit, "chip": chip or "",
            "scoring_ids": scored.get("scoring_ids", []),
            "captain_id": scored.get("captain_id"),
            "captain_multiplier": scored.get("captain_multiplier", 1),
            # The full fifteen, not just who scored. `rescore` never reads it, but without it
            # nothing outside the simulator can tell a player who was BOUGHT early and
            # benched from one who was not bought at all — which is exactly the difference a
            # forecast the decision could not have seen produces.
            "squad_ids": sorted(squad),
            # The transfers made this week and the forecast margin that justified them.
            # Season totals cannot tell a policy that traded WELL from one that traded often
            # and got lucky; ~190 individual decisions a season can.
            "moves": moves,
        })

    table = pd.DataFrame(rows)
    return SeasonResult(
        table, float(table["points"].sum()), label,
        meta={"chips": planner.played if planner else {}},
        trace=trace,
    )


def rescore(trace: list[dict], outcomes: dict) -> float:
    """Season total for an already-decided season, against a different set of outcomes.

    Every decision the simulator makes — opening squad, transfers, XI, armband, chip timing —
    reads only forecast columns. Realised points enter at one place: scoring. So a season's
    decisions can be solved ONCE and then scored against as many resampled outcome draws as
    wanted, which is what makes repeated simulation affordable rather than an overnight job.

    `outcomes` maps `(gw, player_id)` to points. Missing entries score zero, matching the
    simulator's own treatment of a player with no row that week.
    """
    total = 0.0
    for week in trace:
        gw = week["gw"]
        total += sum(outcomes.get((gw, pid), 0.0) for pid in week["scoring_ids"])
        if week["captain_id"] is not None:
            total += week["captain_multiplier"] * outcomes.get(
                (gw, week["captain_id"]), 0.0
            )
        total -= week["hit"]
    return total


class ChipPlanner:
    """Decides which chip, if any, to play each gameweek.

    Holds the optimal-stopping state: which chips remain, when each expires, and a running
    estimate of what a typical week is worth so that opportunities beyond the forecast
    horizon are not treated as if they do not exist.
    """

    def __init__(self, rules: dict, gameweeks: list[int], horizon: int,
                 allowed: tuple[str, ...] | None = None) -> None:
        self.horizon = horizon
        self.gameweeks = gameweeks
        self.played: dict[str, int] = {}
        # One use per WINDOW, not per chip: there are two of each, one per half of the
        # season, and they do not carry over. Keying availability on the chip name alone
        # silently throws away the second-half copy.
        self.slots = [
            [chip, start, stop, False]      # the flag is "already used"
            for chip, windows in chip_windows(rules).items()
            for start, stop in windows
            if allowed is None or chip in allowed
        ]
        self.history: dict[str, list[float]] = {}
        self._best_xi: dict[tuple, float | None] = {}

    def _best_available(self, gw, frame, rules, budget, points_col, as_of=None) -> float | None:
        """Best XI reachable from scratch this gameweek — the thing a rebuild is worth.

        Memoised on gameweek and budget, because valuing a wildcard means solving a full
        15-player MILP and the planner asks for one at every gameweek in its lookahead
        window: 38 gameweeks x 7 weeks ahead x 2 rebuild chips is over 500 solves a season,
        and it dominated the runtime of everything that plays them.

        The budget is rounded to £0.5m before it becomes part of the key. Two budgets that
        close produce the same squad in all but pathological cases, and the alternative is a
        cache that never hits — the whole quantity is a rough comparison against the current
        XI, not a number anything is settled on.

        `as_of` belongs in the key. Once forecasts are point-in-time, GW17 seen from GW12 is
        a different frame from GW17 seen from GW16, and caching on the gameweek alone would
        serve one week's answer to another's question.
        """
        key = (gw, round(budget * 2), as_of)
        if key not in self._best_xi:
            try:
                solution = _legal_squad(frame, rules, budget, points_col)
                self._best_xi[key] = float(
                    solution.starting_xi[points_col].fillna(0).sum()
                )
            except RuntimeError:
                self._best_xi[key] = None
        return self._best_xi[key]

    def _value(self, chip, gw, squad, bank, frame_for, rules, points_col, as_of=None) -> float:
        frame = frame_for(gw, as_of)
        held = frame[frame["player_id"].isin(squad)]
        if held.empty:
            return 0.0

        _, xi = pick_xi(held, rules, points_col)
        if chip == "bench_boost":
            bench_ids = set(held["player_id"]) - set(xi["player_id"])
            return bench_boost_value(held, bench_ids, points_col)
        if chip == "triple_captain":
            captain_id = xi.nlargest(1, points_col)["player_id"].iloc[0] if not xi.empty else None
            return triple_captain_value(xi, captain_id, points_col)

        budget = bank + float(held["price"].fillna(0).sum())
        current = float(xi[points_col].fillna(0).sum())
        best = self._best_available(gw, frame, rules, budget, points_col, as_of)
        if best is None:
            return 0.0
        return free_hit_value(current, best) if chip == "free_hit" else wildcard_value(
            current, best
        )

    def decide(self, gw, squad, bank, frame_for, rules, points_col):
        """Play at most one chip — FPL allows only one per gameweek."""
        best_slot, best_reason, best_value = None, "", 0.0

        for slot in self.slots:
            chip, start, stop, used = slot
            if used or not (start <= gw <= stop):
                continue
            value = self._value(chip, gw, squad, bank, frame_for, rules, points_col, gw)
            self.history.setdefault(chip, []).append(value)

            # Every forward valuation is anchored at THIS gameweek. Holding a chip is an
            # optimal-stopping problem, and one solved with next month's forecasts would
            # stop far too well.
            future = [
                self._value(chip, g, squad, bank, frame_for, rules, points_col, gw)
                for g in self.gameweeks
                if gw < g <= min(gw + self.horizon, stop)
            ]
            unknown = max(0, stop - (gw + self.horizon))
            typical = float(np.median(self.history[chip])) if self.history[chip] else 0.0

            play, reason = should_play(
                chip, value, future, unknown_weeks=unknown, unknown_estimate=typical,
                weeks_to_expiry=stop - gw,
            )
            if play and value > best_value:
                best_slot, best_reason, best_value = slot, reason, value

        if best_slot is None:
            return None, ""
        best_slot[3] = True
        chip = best_slot[0]
        self.played[f"{chip}_{best_slot[1]}-{best_slot[2]}"] = gw
        return chip, best_reason


def _update_purchases(purchase: dict, before: set, after: set, frame: pd.DataFrame) -> dict:
    """Keep the purchase-price ledger in step with the squad.

    Players sold are dropped and players bought are recorded at the price they were bought
    for, which is what they will later be sold against. A player who leaves and returns
    later is re-recorded at the new price, which is correct — FPL forgets your old basis.
    """
    prices = frame.drop_duplicates("player_id").set_index("player_id")["price"]
    updated = {pid: cost for pid, cost in purchase.items() if pid in after}
    for pid in after - before:
        if pid in prices.index:
            updated[pid] = float(prices.loc[pid])
    return updated


def selling_prices(held: pd.DataFrame, purchase: dict) -> pd.Series:
    """What each held player would actually fetch, given what he was bought for.

    Reuses the live path's rule (`data/my_team.selling_price_tenths`) rather than restating
    it: you keep half of any profit, rounded DOWN to 0.1, and bear losses in full. Bought at
    7.0 and now worth 7.5, you receive 7.2.

    Previously the simulator took selling price to be market price, which quietly handed it
    the full rise on every player who had gone up and let the optimiser propose transfers a
    real manager could not fund.
    """
    from ..data.my_team import PRICE_DIVISOR, selling_price_tenths

    current = held["price"].fillna(0.0)
    bought = held["player_id"].map(purchase).fillna(current)
    return pd.Series(
        [
            selling_price_tenths(round(b * PRICE_DIVISOR), round(c * PRICE_DIVISOR))
            / PRICE_DIVISOR
            for b, c in zip(bought, current, strict=True)
        ],
        index=held.index,
    )


def _milp_transfer(squad, bank, frame, rules, points_col, free_transfers, max_transfers,
                   purchase=None, hit_bar=None):
    """Transfers chosen by the real optimiser rather than a single greedy swap.

    The greedy policy examines exactly one candidate move per gameweek and cannot express
    the two things that matter most: a multi-transfer chain (sell a mid-priced player to
    fund an upgrade elsewhere) and a hit that pays for itself. It also has to check the
    club limit after the fact, which is what caused it to abandon 43% of gameweeks.

    Selling price follows FPL's real rule when `purchase` is supplied — half of any profit,
    rounded down — rather than assuming a player can be sold for his market price.
    """
    held = frame[frame["player_id"].isin(squad)].copy()
    if held.empty:
        return squad, bank, 0, 0, None

    held["selling_price"] = (
        selling_prices(held, purchase) if purchase else held["price"]
    )
    candidates = frame[~frame["player_id"].isin(squad)]
    if candidates.empty:
        return squad, bank, 0, 0, None

    try:
        plan = recommend_transfers(
            held, frame, bank=bank, free_transfers=free_transfers,
            max_per_club=rules["squad"]["max_per_club"],
            squad_quota=rules["squad"]["positions"],
            max_transfers=max_transfers, points_col=points_col,
            **({} if hit_bar is None else {"hit_cost": hit_bar}),
        )
    except (RuntimeError, ValueError) as exc:
        log.debug("transfer solve skipped: %s", exc)
        return squad, bank, 0, 0, None

    if plan.n_transfers == 0 or plan.net_gain <= 0:
        return squad, bank, 0, 0, None

    new_squad = (squad - set(plan.transfers_out["player_id"])) | set(
        plan.transfers_in["player_id"]
    )
    # The plan travels back with the squad so the trace can record WHY each move was made.
    # A season total cannot distinguish a policy that traded well from one that traded often
    # and got lucky; the forecast margin behind each move can, and there are ~190 of them a
    # season against three season totals.
    return new_squad, plan.bank_after, plan.n_transfers, plan.hits, plan


def _maybe_transfer(squad, bank, frame, rules, points_col, threshold, purchase=None):
    """Swap the worst held player for the best affordable replacement, if it is worth it."""
    held = frame[frame["player_id"].isin(squad)].copy()
    if held.empty:
        return squad, bank, 0

    held["selling_price"] = selling_prices(held, purchase) if purchase else held["price"]
    worst = held.nsmallest(1, points_col).iloc[0]
    candidates = frame[
        (~frame["player_id"].isin(squad))
        & (frame["position"] == worst["position"])
        & (frame["price"] <= worst["selling_price"] + bank)
    ]
    if candidates.empty:
        return squad, bank, 0

    # Walk candidates best-first until one is actually legal. Taking ONLY the single best
    # and abandoning the transfer when it breaks the 3-per-club limit blocked 43% of
    # gameweeks in 2025-26 — 16 of 37 — despite gains of 10+ horizon points being available.
    # The next-best candidate is almost always legal and nearly as good.
    limit = rules["squad"]["max_per_club"]
    for best in candidates.nlargest(20, points_col).itertuples():
        gain = getattr(best, points_col) - worst[points_col]
        if gain < threshold:
            break                       # sorted by value: nothing later can clear it either
        new_squad = (squad - {worst["player_id"]}) | {best.player_id}
        clubs = frame[frame["player_id"].isin(new_squad)]["team"].value_counts()
        if clubs.max() <= limit:
            return new_squad, bank + worst["selling_price"] - best.price, 1

    return squad, bank, 0


# --- benchmarks and ablation ---------------------------------------------


# How many past gameweeks a form baseline averages over, and how hard it is shrunk toward a
# price prior when a player has fewer than that.
FORM_WINDOW = 6
FORM_PRIOR_WEIGHT = 2.0
# Rough points per gameweek per £m. FIXED rather than fitted, so the baseline carries no
# information from the season it is being scored on — a fitted scale would be a small leak
# and this is a yardstick, not a model.
FORM_PRICE_SCALE = 0.25


def form_scores(
    forecasts: pd.DataFrame,
    *,
    window: int = FORM_WINDOW,
    prior_weight: float = FORM_PRIOR_WEIGHT,
) -> pd.Series:
    """Recent points per gameweek, as known BEFORE each gameweek kicks off.

    Exists because the price-only baseline turned out to be unusable. Maximising a
    price-derived objective under a budget that binds at exactly £100m is close to degenerate:
    two solutions 0.8% apart in objective shared 5 of 15 players and scored 90 against 36 in
    the same gameweek. There is no single number to quote, so beating it means nothing.

    Form is not collinear with the budget, so its optimum is a real one. It is also what an
    ordinary manager actually uses, which makes it the honest bar for a forecasting system:
    the model has to beat "buy whoever has been scoring".

    `shift(1)` before rolling is what keeps it point-in-time — a player's own result must not
    inform the squad that was picked to include him. Early gameweeks have little or no history,
    so the mean is shrunk toward a price prior with a weight of `prior_weight` pseudo-weeks;
    at GW1 that leaves the baseline picking on price alone, which is exactly the information a
    real manager has in August.
    """
    frame = forecasts[["gw", "player_id", "actual_points", "price"]].sort_values(
        ["player_id", "gw"]
    )
    key = frame["player_id"]
    previous = frame["actual_points"].fillna(0.0).groupby(key).shift(1)

    rolled = (
        previous.groupby(key).rolling(window, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    seen = (
        previous.groupby(key).rolling(window, min_periods=1).count()
        .reset_index(level=0, drop=True)
    )
    prior = frame["price"].fillna(0.0) * FORM_PRICE_SCALE
    shrunk = (rolled.fillna(0.0) * seen + prior * prior_weight) / (seen + prior_weight)
    return shrunk.reindex(forecasts.index)


def attach_form(
    forecasts: pd.DataFrame,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict[int, dict[int, pd.DataFrame]] | None]:
    """Add `form_score` to a season's frames, keeping the horizon views point-in-time.

    The horizon frames get the DECISION week's form, not the target week's. A GW10 manager
    judging GW15 knows how his players have played up to GW10 and nothing more; giving those
    frames GW15's own form would rebuild, on the baseline, precisely the lookahead this
    project spent a session removing.
    """
    scored = forecasts.assign(form_score=form_scores(forecasts, **kwargs))
    if lookahead is None:
        return scored, None

    by_gw = {
        gw: block.set_index("player_id")["form_score"]
        for gw, block in scored.groupby("gw")
    }
    out = {}
    for as_of, window in lookahead.items():
        known = by_gw.get(as_of)
        out[as_of] = {
            gw: frame.assign(
                form_score=(
                    frame["player_id"].map(known).fillna(
                        frame["price"].fillna(0.0) * FORM_PRICE_SCALE
                    )
                    if known is not None
                    else frame["price"].fillna(0.0) * FORM_PRICE_SCALE
                )
            )
            for gw, frame in window.items()
        }
    return scored, out


def benchmark_form(
    forecasts: pd.DataFrame,
    rules: dict | None = None,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
    **kwargs,
):
    """Pick on recent scoring form — the bar a forecasting system has to clear."""
    scored, forward = attach_form(forecasts, lookahead)
    return simulate_season(
        scored, label="form", points_col="form_score", lookahead=forward, **kwargs
    )


def benchmark_random(forecasts: pd.DataFrame, seed: int = 0, rules: dict | None = None):
    """A legal squad chosen at random — the floor any model must clear comfortably.

    Benchmarks take no `lookahead`: they score on a column the horizon frames do not carry,
    and any lookahead they retain flatters THEM, not us.
    """
    rng = np.random.default_rng(seed)
    noisy = forecasts.copy()
    noisy["random_score"] = rng.random(len(noisy))
    return simulate_season(noisy, label="random", points_col="random_score", rules=rules)


def benchmark_price(forecasts: pd.DataFrame, rules: dict | None = None):
    """Pick purely on price. FPL prices encode a great deal of expert judgement, so this is
    a genuinely hard baseline and a model that cannot beat it has earned nothing."""
    priced = forecasts.copy()
    priced["price_score"] = priced["price"]
    return simulate_season(priced, label="price-only", points_col="price_score", rules=rules)


ABLATIONS = {
    "no_bonus": ["pts_bonus"],
    "no_defcon": ["pts_defcon"],
    "no_saves": ["pts_saves"],
    "no_cards": ["pts_cards"],
    "no_clean_sheet": ["pts_clean_sheet"],
    "no_attack": ["pts_goals", "pts_assists"],
    "no_appearance": ["pts_appearance"],
}


def _drop_components(
    lookahead: dict[int, dict[int, pd.DataFrame]] | None, columns: list[str]
) -> dict[int, dict[int, pd.DataFrame]] | None:
    """Remove the same point components from every horizon frame.

    Without this an ablation only removes a component from the gameweek in front of the
    manager, leaving it intact in the forward valuation that actually drives transfers —
    which would measure roughly a sixth of the component's true decision weight.
    """
    if not lookahead:
        return None
    out = {}
    for as_of, window in lookahead.items():
        frames = {}
        for gw, frame in window.items():
            present = [c for c in columns if c in frame.columns]
            if not present:
                frames[gw] = frame
                continue
            trimmed = frame.copy()
            trimmed["expected_points"] = (
                trimmed["expected_points"] - trimmed[present].sum(axis=1)
            )
            frames[gw] = trimmed
        out[as_of] = frames
    return out


def ablate(
    forecasts: pd.DataFrame,
    rules: dict | None = None,
    lookahead: dict[int, dict[int, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Season points with each component zeroed, to rank where effort is worth spending.

    Interpretation is the point. A large drop means the component carries real decision
    weight and deserves a better model. A drop near zero means it is either already
    irrelevant or so crude it is contributing nothing — and the two are distinguishable by
    whether removing it *helps*.
    """
    baseline = simulate_season(forecasts, label="full", rules=rules, lookahead=lookahead)
    rows = [{"variant": "full", "total_points": baseline.total_points, "delta": 0.0}]

    for name, columns in ABLATIONS.items():
        present = [c for c in columns if c in forecasts.columns]
        if not present:
            continue
        variant = forecasts.copy()
        variant["expected_points"] = variant["expected_points"] - variant[present].sum(axis=1)
        result = simulate_season(
            variant, label=name, rules=rules,
            lookahead=_drop_components(lookahead, present),
        )
        rows.append({
            "variant": name,
            "total_points": result.total_points,
            "delta": result.total_points - baseline.total_points,
        })
    return pd.DataFrame(rows).sort_values("delta")
