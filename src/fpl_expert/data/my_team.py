"""Read your actual squad from the FPL API, including exact selling prices.

Everything here uses public endpoints. `my-team/{id}/` would give the live squad and selling
prices directly, but it returns 403 without a login, and storing FPL credentials is not worth
it: purchase prices can be reconstructed exactly from the public transfer history, which is
all the selling-price rule needs.

    entry/{id}/                    bank, team value, transfers made
    entry/{id}/event/{gw}/picks/   the 15 you fielded that gameweek
    entry/{id}/transfers/          every transfer, with the price paid and received

**Timing.** Picks for the UPCOMING gameweek 404 until its deadline passes, so "your current
squad" is read from the last completed gameweek and then rolled forward by any transfers
already registered for the upcoming one. Without that roll-forward the optimiser would
recommend a transfer you have already made.
"""

from __future__ import annotations

import logging

import pandas as pd

from .fpl_api import FplApi

log = logging.getLogger(__name__)

PRICE_DIVISOR = 10.0


def selling_price_tenths(purchase: int, current: int) -> int:
    """FPL's selling-price rule, computed in tenths to avoid float rounding drift.

    You keep only HALF of any profit, rounded DOWN to the nearest 0.1; losses are borne in
    full. Getting this wrong silently inflates your budget and makes the optimiser propose
    transfers you cannot afford — bought at 7.0 and now worth 7.5, you receive 7.2, not 7.5.
    """
    profit = current - purchase
    if profit <= 0:
        return current
    return purchase + profit // 2


def fetch_entry(api: FplApi, entry_id: int) -> dict:
    entry = api.entry(entry_id, use_cache=False)
    if entry is None:
        raise ValueError(f"entry {entry_id} not found — check the manager id")
    return entry


def fetch_transfers(api: FplApi, entry_id: int) -> pd.DataFrame:
    """Full transfer history: element in/out and the prices at the time."""
    payload = api.client.get_json(
        f"{api.base}/entry/{entry_id}/transfers/", use_cache=False, allow_404=True
    )
    if not payload:
        return pd.DataFrame(
            columns=["element_in", "element_out", "element_in_cost", "element_out_cost", "event"]
        )
    return pd.DataFrame(payload)


def purchase_prices(
    initial_picks: pd.DataFrame, transfers: pd.DataFrame, current_squad: set[int]
) -> dict[int, int]:
    """What you actually paid for each player currently held, in tenths.

    Players from the opening squad were bought at their GW1 price; anyone transferred in
    later was bought at `element_in_cost`. Replaying the transfer log in order handles a
    player bought, sold and bought again at a different price.
    """
    paid: dict[int, int] = {}
    for row in initial_picks.itertuples():
        paid[int(row.element)] = int(getattr(row, "purchase_price", 0) or 0)

    if not transfers.empty:
        for row in transfers.sort_values("event").itertuples():
            paid[int(row.element_in)] = int(row.element_in_cost)
            paid.pop(int(row.element_out), None)

    return {pid: price for pid, price in paid.items() if pid in current_squad}


def current_squad(
    api: FplApi, entry_id: int, upcoming_gw: int, players: pd.DataFrame
) -> pd.DataFrame:
    """Your squad going into `upcoming_gw`, with purchase and selling prices attached.

    `players` is the live player table (from a `PointInTime` snapshot), supplying current
    prices and names.
    """
    last_completed = upcoming_gw - 1
    picks_payload = None
    while last_completed >= 1 and picks_payload is None:
        picks_payload = api.entry_picks(entry_id, last_completed, use_cache=False)
        if picks_payload is None:
            last_completed -= 1
    if picks_payload is None:
        raise ValueError(
            f"no completed gameweek picks for entry {entry_id}. Before GW1 there is no squad "
            f"to optimise from — use `fpl squad` to build an opening 15 instead."
        )

    picks = pd.DataFrame(picks_payload["picks"])
    held = set(picks["element"].astype(int))

    # Roll forward any transfers already registered for the upcoming gameweek.
    transfers = fetch_transfers(api, entry_id)
    pending = transfers[transfers["event"] == upcoming_gw] if not transfers.empty else transfers
    for row in pending.itertuples():
        held.discard(int(row.element_out))
        held.add(int(row.element_in))
    if len(pending):
        log.info("applied %d transfer(s) already made for GW%d", len(pending), upcoming_gw)

    first_picks = api.entry_picks(entry_id, 1, use_cache=True)
    initial = pd.DataFrame(first_picks["picks"]) if first_picks else pd.DataFrame(columns=["element"])
    paid = purchase_prices(initial, transfers, held)

    squad = players[players["id"].isin(held)].copy()
    squad["purchase_price_tenths"] = squad["id"].map(paid).fillna(squad["now_cost"]).astype(int)
    squad["selling_price_tenths"] = [
        selling_price_tenths(int(p), int(c))
        for p, c in zip(squad["purchase_price_tenths"], squad["now_cost"], strict=True)
    ]
    squad["selling_price"] = squad["selling_price_tenths"] / PRICE_DIVISOR
    squad["price"] = squad["now_cost"] / PRICE_DIVISOR
    squad["is_captain"] = squad["id"].isin(
        picks.loc[picks.get("is_captain", False), "element"] if "is_captain" in picks else []
    )
    return squad


def free_transfers(api: FplApi, entry_id: int, upcoming_gw: int, max_banked: int = 5) -> int:
    """Free transfers available, reconstructed from the transfer log.

    One is earned per gameweek and they accumulate up to `max_banked`. Derived rather than
    read because the count is only exposed on the authenticated endpoint.
    """
    transfers = fetch_transfers(api, entry_id)
    available = 1
    for gw in range(2, upcoming_gw + 1):
        used = int((transfers["event"] == gw).sum()) if not transfers.empty else 0
        if gw < upcoming_gw:
            available = min(available - used + 1, max_banked)
            available = max(available, 1)
    return max(1, min(available, max_banked))


def bank(entry: dict) -> float:
    """Money in the bank, in millions. Null before the season starts."""
    value = entry.get("last_deadline_bank")
    return 0.0 if value is None else value / PRICE_DIVISOR
