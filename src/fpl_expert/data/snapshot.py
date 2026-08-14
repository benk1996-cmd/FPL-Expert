"""Point-in-time snapshots, and the guard rails that make backtests honest.

Most FPL data is recoverable after the fact: results, player stats, final points all get
published. A few fields are not, because they are *overwritten in place* as the world
changes and nobody archives the old value:

    news, chance_of_playing_next_round, status   injury/availability, rewritten constantly
    now_cost                                      prices drift daily
    selected_by_percent                           ownership moves continuously
    kickoff_time                                  shifts with TV scheduling
    bookmaker odds at the deadline                only open and close are ever published

Those are exactly the fields a pre-deadline decision depends on. If we do not capture them
before each deadline, that state is gone permanently and no amount of later work
reconstructs it. Hence this module, and hence the fact that `data/raw/` is append-only.

The second job here is enforcement. `PointInTime` is the only sanctioned way to read data
for a gameweek being predicted: it resolves to the last snapshot taken *before* that
gameweek's deadline, and it has no method that returns post-deadline market data. See
`assert_no_post_deadline_columns` for the belt-and-braces check on assembled frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from ..config import load_config
from .fpl_api import FplApi, next_gameweek, parse_fixtures, parse_players
from .storage import TS_FORMAT, raw_snapshots, read_raw, read_table, utc_stamp, write_raw

log = logging.getLogger(__name__)

SOURCE = "snapshot"

# Columns carrying information that did not exist at the FPL deadline. Never features.
POST_DEADLINE_SUFFIXES = ("_close",)
POST_DEADLINE_COLUMNS = frozenset({"home_goals", "away_goals"})


class LookaheadError(ValueError):
    """Raised when a frame intended for features contains post-deadline information."""


class MissingSnapshotError(FileNotFoundError):
    """Raised when no pre-deadline snapshot exists for a gameweek. Unrecoverable."""


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, TS_FORMAT).replace(tzinfo=UTC)


def assert_no_post_deadline_columns(df: pd.DataFrame, *, context: str = "feature frame") -> None:
    """Fail loudly if a feature frame has picked up post-deadline data.

    A join is the usual culprit: pulling in the odds table for team names and silently
    dragging `p_home_close` along with it. That column would be the strongest predictor in
    the matrix and would quietly invalidate every downstream result, so this raises rather
    than warns.
    """
    offending = [
        c for c in df.columns
        if c in POST_DEADLINE_COLUMNS or c.endswith(POST_DEADLINE_SUFFIXES)
    ]
    if offending:
        raise LookaheadError(
            f"{context} contains post-deadline columns {sorted(offending)}. These did not "
            f"exist at the FPL deadline and cannot be features. Use the odds_features "
            f"table, not odds_eval."
        )


# --- taking snapshots -----------------------------------------------------


def take_snapshot(reason: str = "manual") -> dict:
    """Capture the full mutable state of the game right now, plus a manifest.

    Deliberately bypasses the HTTP cache: a cached bootstrap would record stale
    availability and prices under a fresh timestamp, which is worse than no snapshot
    because it looks valid.
    """
    api = FplApi()
    taken_at = datetime.now(UTC)
    stamp = utc_stamp(taken_at)

    bootstrap = api.bootstrap_static(use_cache=False)
    fixtures = api.fixtures(use_cache=False)
    write_raw(bootstrap, SOURCE, "bootstrap", stamp=stamp)
    write_raw(fixtures, SOURCE, "fixtures", stamp=stamp)

    target_gw = next_gameweek(bootstrap)
    deadline = _deadline_for(bootstrap, target_gw)

    odds_rows = 0
    try:  # best effort — a missing odds feed must not cost us the FPL snapshot
        from .http import HttpClient
        from .odds import fetch_upcoming, normalise

        upcoming = normalise(fetch_upcoming(HttpClient(min_interval=1.0)), _season())
        write_raw(upcoming.to_dict("records"), SOURCE, "odds", stamp=stamp)
        odds_rows = len(upcoming)
    except Exception as exc:  # noqa: BLE001 - never let odds failure abort the snapshot
        log.warning("odds capture failed for snapshot %s: %s", stamp, exc)

    manifest = {
        "stamp": stamp,
        "taken_at": taken_at.isoformat(),
        "reason": reason,
        "season": _season(),
        "target_gw": target_gw,
        "deadline": deadline.isoformat() if deadline is not None else None,
        # The single most important field here. A snapshot taken after the deadline is a
        # perfectly good record of history but must never be used as a feature source.
        "taken_before_deadline": bool(deadline is not None and taken_at < deadline),
        "hours_to_deadline": (
            round((deadline - taken_at).total_seconds() / 3600, 2) if deadline is not None else None
        ),
        "players": len(bootstrap.get("elements", [])),
        "fixtures": len(fixtures),
        "odds_rows": odds_rows,
    }
    write_raw(manifest, SOURCE, "manifest", stamp=stamp)
    log.info(
        "snapshot %s: GW%s, %.2fh to deadline, before_deadline=%s",
        stamp, target_gw, manifest["hours_to_deadline"] or float("nan"),
        manifest["taken_before_deadline"],
    )
    return manifest


def _season() -> str:
    return str(load_config().project.get("season", "unknown"))


def _deadline_for(bootstrap: dict, gw: int | None) -> datetime | None:
    if gw is None:
        return None
    for event in bootstrap["events"]:
        if event["id"] == gw:
            return pd.Timestamp(event["deadline_time"]).to_pydatetime().astimezone(UTC)
    return None


def list_snapshots() -> pd.DataFrame:
    """Every snapshot manifest, oldest first."""
    rows = []
    for path in raw_snapshots(SOURCE, "manifest"):
        rows.append(read_raw(SOURCE, "manifest", stamp=path.parent.name.split("=", 1)[1]))
    return pd.DataFrame(rows)


# Hours before a deadline at which we want a snapshot. Escalating rather than one-and-done
# for two reasons. Freshness: a capture 24h out misses the Friday press conferences, and
# `PointInTime` always uses the LATEST pre-deadline snapshot, so a later one is strictly
# better. Robustness: a job on a personal machine only fires while that machine is awake,
# so several chances to catch a deadline beats one. Each capture is a few hundred KB.
DEFAULT_CHECKPOINTS: tuple[float, ...] = (48.0, 24.0, 6.0, 2.0, 0.5)


def snapshot_due(
    checkpoints: tuple[float, ...] = DEFAULT_CHECKPOINTS,
) -> tuple[bool, str]:
    """Whether a snapshot should be taken now, for use by a scheduled job.

    Walks a ladder of checkpoints towards the deadline. Due when we have passed a
    checkpoint but hold no snapshot taken that close to the deadline. Safe to run hourly:
    it captures at most once per checkpoint and is otherwise a no-op.
    """
    api = FplApi()
    bootstrap = api.bootstrap_static(use_cache=True)
    gw = next_gameweek(bootstrap)
    deadline = _deadline_for(bootstrap, gw)
    if deadline is None:
        return False, "no upcoming gameweek"

    hours = (deadline - datetime.now(UTC)).total_seconds() / 3600
    if hours < 0:
        return False, f"GW{gw} deadline has passed"

    passed = [c for c in checkpoints if hours <= c]
    if not passed:
        return False, f"GW{gw} deadline is {hours:.1f}h away (earliest checkpoint {max(checkpoints)}h)"
    target = min(passed)

    existing = list_snapshots()
    if not existing.empty:
        held = existing[(existing["target_gw"] == gw) & existing["taken_before_deadline"]]
        if not held.empty and (held["hours_to_deadline"] <= target).any():
            return False, f"GW{gw}: already hold a snapshot inside the {target}h checkpoint"
    return True, f"GW{gw} deadline in {hours:.1f}h — capturing for the {target}h checkpoint"


# --- reading, with lookahead made structurally impossible -----------------


@dataclass(frozen=True)
class PointInTime:
    """Read-only view of the world as it stood at `as_of`.

    Every feature for a gameweek must be built through one of these. There is deliberately
    no accessor for closing odds or match results — the guarantee is enforced by what this
    class does not expose, not by remembering to avoid a column.
    """

    as_of: datetime
    stamp: str
    target_gw: int | None = None

    @classmethod
    def at(cls, as_of: datetime, target_gw: int | None = None) -> PointInTime:
        return cls(as_of=as_of, stamp=utc_stamp(as_of), target_gw=target_gw)

    @classmethod
    def for_gameweek(cls, gw: int) -> PointInTime:
        """Resolve to the latest snapshot taken before gameweek `gw`'s deadline.

        Raises rather than falling back to a later snapshot: silently using post-deadline
        state is the exact failure this module exists to prevent, and a loud error at
        build time is far cheaper than a backtest nobody can trust.
        """
        manifests = list_snapshots()
        if manifests.empty:
            raise MissingSnapshotError("no snapshots have been taken — run `fpl snapshot`")
        eligible = manifests[
            (manifests["target_gw"] == gw) & manifests["taken_before_deadline"]
        ].sort_values("stamp")
        if eligible.empty:
            raise MissingSnapshotError(
                f"no pre-deadline snapshot for GW{gw}. That state is unrecoverable — the "
                f"gameweek cannot be used for point-in-time evaluation."
            )
        row = eligible.iloc[-1]
        return cls(as_of=parse_stamp(row["stamp"]), stamp=row["stamp"], target_gw=gw)

    @classmethod
    def for_planning(cls, gw: int) -> PointInTime:
        """State to plan gameweek `gw` with — the pre-deadline snapshot, or the newest one.

        `for_gameweek` answers a BACKTEST question: what did we know before this deadline? For
        a gameweek that has not happened yet there is no such snapshot and there never will be
        until it arrives, so planning three weeks ahead with it fails outright.

        Planning asks a different question — what do we know NOW about a future gameweek — and
        the honest answer is today's prices, availability and fixtures projected forward. That
        is not a lookahead violation, because a later snapshot cannot exist yet; the distinction
        is kept in a separate constructor precisely so nothing evaluating the past can reach it
        by accident.
        """
        try:
            return cls.for_gameweek(gw)
        except MissingSnapshotError:
            manifests = list_snapshots()
            if manifests.empty:
                raise
            row = manifests.sort_values("stamp").iloc[-1]
            log.info(
                "no pre-deadline snapshot for GW%d — planning from the latest state (%s)",
                gw, row["stamp"],
            )
            return cls(as_of=parse_stamp(row["stamp"]), stamp=row["stamp"], target_gw=gw)

    # -- feeds -------------------------------------------------------------

    def bootstrap(self) -> dict:
        return read_raw(SOURCE, "bootstrap", stamp=self.stamp)

    def players(self) -> pd.DataFrame:
        """Player state as known at the deadline, including availability and price."""
        return parse_players(self.bootstrap())

    def fixtures(self) -> pd.DataFrame:
        return parse_fixtures(read_raw(SOURCE, "fixtures", stamp=self.stamp))

    def odds(self) -> pd.DataFrame:
        """Odds as captured at the snapshot, else pre-deadline odds from the feature table.

        Never closing odds. `odds_eval` is not reachable from here by design.
        """
        try:
            captured = pd.DataFrame(read_raw(SOURCE, "odds", stamp=self.stamp))
            if not captured.empty:
                assert_no_post_deadline_columns(captured, context="snapshot odds")
                return captured
        except FileNotFoundError:
            pass
        df = read_table("external", "odds_features")
        assert_no_post_deadline_columns(df, context="odds_features")
        return df
