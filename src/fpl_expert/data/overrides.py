"""Hand-entered availability corrections for the LIVE path, applied at inference only.

Why this exists
---------------

The model reads no press conferences. `status` and `chance_of_playing_next_round` are the only
availability signals it has, and they are FPL's — which means the model knows a player has left
or is injured exactly when the game says so, and not a moment sooner. Between a departure being
reported and FPL flagging it, the forecast carries an asset that will not play, and every
decision built on it is wrong in a way no amount of modelling can detect.

This is the narrow, checkable patch for that gap: a file of corrections keyed to a player, each
carrying a reason and an expiry, applied to the player table before the availability gate runs.

What it is NOT
--------------

* **Not a model parameter.** It corrects an INPUT that is stale, in the same units FPL
  publishes. Nothing here is tuned, fitted, or measured against outcomes.
* **Not reachable from the backtest.** The archive has no availability data at all, so no
  historical row can be overridden and no measured result can be moved by editing this file.
  See the permanent limitation in NEXT_SESSION.md.
* **Not written to the snapshot.** `data/raw/` is the evidence trail and stays exactly as
  captured. Overrides are applied to the frame in memory, after the snapshot is read.

Every applied override logs a WARNING, every expired one logs that it was ignored, and every
entry matching no player logs that it matched nothing. A correction that silently stops applying
— because a name changed, or a date passed — is the failure this project has been bitten by
before: a fallback that changes the model must say so.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from ..config import _read_yaml, project_root

log = logging.getLogger(__name__)

# FPL's own vocabulary, reused rather than invented so an override says the same thing the feed
# would have said. `apply_availability_gate` zeroes i/s/u/n and reads `d` through the percentage.
VALID_STATUS = {"a", "d", "i", "s", "u", "n"}


def load_overrides(path: Path | None = None) -> list[dict]:
    """Read the corrections file. Absent or empty is the normal case, not an error."""
    path = path or project_root() / "config" / "overrides.yaml"
    if not Path(path).exists():
        return []
    return list((_read_yaml(Path(path)) or {}).get("availability") or [])


def _expired(entry: dict, today: date) -> bool:
    until = entry.get("until")
    if until is None:
        return False
    if isinstance(until, str):
        until = date.fromisoformat(until)
    return until < today


def apply_availability_overrides(
    players: pd.DataFrame, overrides: list[dict] | None = None, *, today: date | None = None
) -> pd.DataFrame:
    """Return `players` with any live corrections applied to status and playing chance.

    Matches on `id` when given, otherwise on an exact `web_name`. Never partial-matches: two
    players sharing a surname is common and guessing between them would be worse than missing.
    """
    overrides = load_overrides() if overrides is None else overrides
    if not overrides:
        return players
    today = today or datetime.now(UTC).date()

    out = players.copy()
    for entry in overrides:
        who = entry.get("id", entry.get("web_name"))
        if _expired(entry, today):
            log.warning(
                "availability override for %s EXPIRED on %s and was not applied — delete it "
                "or extend `until`", who, entry.get("until"),
            )
            continue

        if "id" in entry:
            mask = out["id"] == entry["id"]
        elif "web_name" in entry:
            mask = out["web_name"] == entry["web_name"]
        else:
            log.warning("availability override needs an `id` or `web_name`: %r", entry)
            continue

        if not mask.any():
            log.warning("availability override matched no player: %r", entry)
            continue
        if mask.sum() > 1:
            log.warning(
                "availability override for %r matches %d players — use `id` instead of "
                "`web_name`; skipped", who, int(mask.sum()),
            )
            continue

        status = entry.get("status")
        if status is not None:
            if status not in VALID_STATUS:
                log.warning("unknown status %r for %s; expected one of %s — skipped",
                            status, who, sorted(VALID_STATUS))
                continue
            out.loc[mask, "status"] = status
        if "chance_of_playing_next_round" in entry:
            out.loc[mask, "chance_of_playing_next_round"] = entry[
                "chance_of_playing_next_round"
            ]

        name = out.loc[mask, "web_name"].iloc[0]
        log.warning(
            "OVERRIDE APPLIED: %s status=%s chance=%s until=%s (%s). This is hand-entered and "
            "is NOT what FPL published.",
            name, status, entry.get("chance_of_playing_next_round"), entry.get("until"),
            entry.get("reason", "no reason given"),
        )
    return out
