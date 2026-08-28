"""Advice for a squad someone actually owns, independent of how it is displayed.

`fpl myteam` and the Streamlit "My Team" view ask the identical question and must not answer
it differently. Before this module existed only the CLI could ask it, and the front end's
choice would have been to duplicate thirty lines of orchestration — the same drift that
already happened once between `report` and `myteam`, where `report` silently rendered no chip
or price section for months because it passed none of `build_report`'s optional arguments.

**This is entry-specific and therefore local.** The serving bundle is committed and deployed,
so it describes the GAME and never the user (see `test_the_bundle_carries_no_personal_data`).
Nothing here may be written into `data/serving/`.

**It is also slow**: `forecast_gameweek` loads the 185,000-row archive, rebuilds minutes
features over all of it and refits Dixon-Coles, once per horizon week. Roughly 20 seconds for
a six-week horizon. Any caller rendering a page must cache it rather than calling per view.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EntryAdvice:
    """Everything the CLI and the front end both need, computed once."""

    entry: int
    manager_name: str
    team_name: str
    gameweek: int
    span: int
    squad: pd.DataFrame          # your 15, with purchase and selling prices
    held: pd.DataFrame           # your 15 joined to this week's forecasts
    latest: pd.DataFrame         # every player, this gameweek plus horizon valuation
    plan: Any                    # TransferPlan
    bank: float
    free_transfers: int
    profile: dict = field(default_factory=dict)
    per_gw: dict = field(default_factory=dict)   # gameweek -> that week's forecasts

    @property
    def gw1_points(self) -> int | None:
        return self.profile.get("summary_event_points")

    @property
    def overall_rank(self) -> int | None:
        return self.profile.get("summary_overall_rank")


def analyse_entry(
    entry: int,
    *,
    gw: int | None = None,
    span: int | None = None,
    max_transfers: int = 2,
    bench_aware: bool = False,
    defer_aware: bool = False,
) -> EntryAdvice:
    """Pull a real squad and work out what to do with it for the coming gameweek.

    `bench_aware` judges each plan on the XI plus autosub-weighted bench rather than the sum of
    fifteen. It was ON here for two days and is now OFF: the ensemble measured it at
    **-126 / -65 / +98** across three seasons, pooled -31 [-70, +8], losing 10 of 10 paths in
    2023-24. Off matches `simulate_season`, so the live policy is again the one the +399
    headline was measured under. Pass `--bench-aware` to reproduce the variant.

    `defer_aware` judges a hit on the CURRENT gameweek's gain, because the same move can
    usually be made next week for a free transfer. Measured at **-41 / +197 / +24**, pooled
    **+60 [+21, +100]** — the strongest decision-layer result this project has produced, and
    still not adoptable: the sign flips, so ground rule 2 refuses it as a default. Exposed as
    `--defer-aware` so the judgement can be made per decision rather than by the default alone.

    Raises `MissingSnapshotError` when no pre-deadline snapshot exists for `gw` — the target
    gameweek is read through the strict point-in-time accessor deliberately, so the fix is to
    run `fpl snapshot` before the deadline rather than to loosen the guard. Future gameweeks
    in the horizon are PLANNING and use the latest known state, because no snapshot for them
    can exist yet.
    """
    from .config import load_config, load_scoring_rules
    from .data.fpl_api import FplApi, next_gameweek
    from .data.my_team import bank, current_squad, fetch_entry, free_transfers
    from .data.snapshot import PointInTime
    from .models.points import aggregate_gameweek
    from .optimise.transfers import horizon_points, recommend_transfers
    from .pipeline import forecast_gameweek

    cfg, rules = load_config(), load_scoring_rules()
    api = FplApi()
    target = gw if gw is not None else next_gameweek(api.bootstrap_static())
    span = span if span is not None else cfg.optimise.horizon_gws

    profile = fetch_entry(api, entry)
    players = PointInTime.for_gameweek(target).players()
    squad = current_squad(api, entry, target, players)
    available = free_transfers(api, entry, target, rules["transfers"]["max_banked"])
    in_bank = bank(profile)

    per_gw = {
        g: aggregate_gameweek(forecast_gameweek(g, planning=g != target))
        for g in range(target, target + span)
    }
    horizon_table = horizon_points(per_gw, decay=cfg.optimise.future_decay)

    latest = per_gw[target].merge(horizon_table, on="player_id", how="left")
    latest["horizon_points"] = latest["horizon_points"].fillna(0.0)
    held = latest[latest["player_id"].isin(squad["id"])].merge(
        squad[["id", "selling_price"]].rename(columns={"id": "player_id"}), on="player_id"
    )

    plan = recommend_transfers(
        held, latest, bank=in_bank, free_transfers=available,
        max_per_club=rules["squad"]["max_per_club"], max_transfers=max_transfers,
        rules=rules, bench_aware=bench_aware, defer_aware=defer_aware,
        per_gw=per_gw if defer_aware else None, decay=cfg.optimise.future_decay,
    )

    return EntryAdvice(
        entry=entry,
        manager_name=" ".join(
            p for p in (profile.get("player_first_name"), profile.get("player_last_name")) if p
        ),
        team_name=profile.get("name") or "?",
        gameweek=target,
        span=span,
        squad=squad,
        held=held,
        latest=latest,
        plan=plan,
        bank=in_bank,
        free_transfers=available,
        profile=profile,
        per_gw=per_gw,
    )
