"""Layered storage: immutable raw dumps, then typed parquet.

`raw/` is append-only and partitioned by pull timestamp. Never overwrite it — it is the
evidence trail that makes the point-in-time backtest (Item 3) defensible. Anything
derived belongs in `interim/` or `processed/` and can be rebuilt at will.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_config

log = logging.getLogger(__name__)

TS_FORMAT = "%Y%m%dT%H%M%SZ"


def utc_stamp(when: datetime | None = None) -> str:
    """Filename-safe UTC timestamp, e.g. `20260809T213000Z`."""
    return (when or datetime.now(UTC)).strftime(TS_FORMAT)


def write_raw(payload: Any, source: str, name: str, *, stamp: str | None = None) -> Path:
    """Write an immutable, gzipped JSON dump to `raw/{source}/{name}/pulled_at=.../data.json.gz`."""
    stamp = stamp or utc_stamp()
    out = load_config().path("raw") / source / name / f"pulled_at={stamp}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "data.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return path


def raw_snapshots(source: str, name: str) -> list[Path]:
    """All dumps for a feed, oldest first (partition names sort chronologically)."""
    base = load_config().path("raw") / source / name
    if not base.exists():
        return []
    return sorted(base.glob("pulled_at=*/data.json.gz"))


def read_raw(source: str, name: str, *, stamp: str | None = None) -> Any:
    """Read one dump — the newest by default, or the last one at/before `stamp`.

    The `stamp` form is what point-in-time feature building uses: it answers "what did we
    know before this deadline?" rather than "what do we know now?".
    """
    snapshots = raw_snapshots(source, name)
    if not snapshots:
        raise FileNotFoundError(f"no raw snapshots for {source}/{name}")
    if stamp is not None:
        eligible = [p for p in snapshots if p.parent.name.split("=", 1)[1] <= stamp]
        if not eligible:
            raise FileNotFoundError(f"no {source}/{name} snapshot at or before {stamp}")
        snapshots = eligible
    with gzip.open(snapshots[-1], "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode dict/list-valued columns so Arrow can write them.

    Several FPL feeds nest structures that Arrow cannot infer a schema for (empty structs
    in `events.overrides`, ragged `fixtures.stats`). We keep them as JSON text rather than
    dropping them; the lossless original is always in `raw/` anyway.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object and out[col].map(lambda v: isinstance(v, (dict, list))).any():
            out[col] = out[col].map(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
            )
    return out


def write_table(df: pd.DataFrame, layer: str, name: str, **partitions: str | int) -> Path:
    """Write a dataframe to `{layer}/{name}/{k=v}/.../{name}.parquet`, overwriting.

    A stamp of when the partition was written goes alongside it. Partitions are refreshed one
    at a time, so a table can silently hold two generations of a model at once — and reading
    the lot back concatenates them without complaint. That happened: a stale 2022-23 partition
    survived a model change and made a re-measured calibration look 12% worse than it was.
    `partition_report` turns that into something visible.
    """
    out = load_config().path(layer) / name
    for key, value in partitions.items():
        out = out / f"{key}={value}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.parquet"
    _parquet_safe(df).to_parquet(path, index=False)
    (out / "_written_at").write_text(
        datetime.now(UTC).isoformat(timespec="seconds"), encoding="utf-8"
    )
    return path


def partition_report(layer: str, name: str) -> pd.DataFrame:
    """When each partition of a table was last written, newest first.

    Use it before trusting any measurement that spans partitions. A spread of write times is
    not automatically wrong — seasons genuinely are regenerated separately — but it means the
    table mixes generations, and whether that matters depends on what changed in between.
    """
    base = load_config().path(layer) / name
    if not base.exists():
        raise FileNotFoundError(f"no table at {base}")

    import pyarrow.parquet as pq

    rows = []
    for part in sorted(base.rglob("*.parquet")):
        stamp = part.parent / "_written_at"
        rows.append({
            "partition": str(part.parent.relative_to(base)),
            "written_at": stamp.read_text(encoding="utf-8").strip() if stamp.exists() else "",
            # From the file's own metadata rather than by loading it — this is a health
            # check and should stay cheap enough to run before every measurement.
            "rows": pq.ParquetFile(part).metadata.num_rows,
        })
    report = pd.DataFrame(rows).sort_values("written_at", ascending=False)
    if report["written_at"].nunique() > 1:
        log.warning(
            "%s/%s spans %d write times — it holds more than one generation of the model",
            layer, name, report["written_at"].nunique(),
        )
    return report.reset_index(drop=True)


def read_table(layer: str, name: str, **partitions: str | int) -> pd.DataFrame:
    """Read one or many parquet parts, concatenating across partitions if unspecified."""
    base = load_config().path(layer) / name
    for key, value in partitions.items():
        base = base / f"{key}={value}"
    if not base.exists():
        raise FileNotFoundError(f"no table at {base}")
    parts = sorted(base.rglob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet files under {base}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
