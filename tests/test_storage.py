"""Storage layer: parquet safety and, critically, point-in-time read semantics."""

from __future__ import annotations

import pandas as pd
import pytest

from fpl_expert.data.storage import read_raw, read_table, write_raw, write_table


def test_write_table_handles_nested_columns(tmp_config):
    """Arrow cannot infer a schema for empty structs or ragged lists; we JSON-encode them."""
    df = pd.DataFrame({
        "id": [1, 2],
        "overrides": [{"rules": {}}, {"rules": {}}],   # empty struct — Arrow rejects natively
        "stats": [[], [{"identifier": "goals_scored"}]],
        "price": [5.5, 10.0],
    })
    write_table(df, "interim", "widgets", season="2026-27")
    out = read_table("interim", "widgets", season="2026-27")

    assert len(out) == 2
    assert out["price"].tolist() == [5.5, 10.0]
    assert out["overrides"].iloc[0] == '{"rules": {}}'
    assert out["stats"].iloc[0] == "[]"


def test_raw_roundtrip_preserves_unicode(tmp_config):
    """Player names are not latin-1; a cp1252 default would corrupt them silently."""
    write_raw({"name": "Højlund", "x": 1}, "fpl_api", "bootstrap", stamp="20260809T120000Z")
    assert read_raw("fpl_api", "bootstrap")["name"] == "Højlund"


def test_read_raw_respects_as_of_stamp(tmp_config):
    """The point-in-time guarantee: reading 'as of' a stamp must not see later snapshots.

    This is the mechanism the whole backtest rests on (Item 3) — if it leaks, every
    downstream evaluation is optimistic.
    """
    write_raw({"price": 5.0}, "fpl_api", "bootstrap", stamp="20260810T000000Z")
    write_raw({"price": 5.1}, "fpl_api", "bootstrap", stamp="20260811T000000Z")
    write_raw({"price": 5.2}, "fpl_api", "bootstrap", stamp="20260812T000000Z")

    assert read_raw("fpl_api", "bootstrap")["price"] == 5.2                          # latest
    assert read_raw("fpl_api", "bootstrap", stamp="20260811T120000Z")["price"] == 5.1
    assert read_raw("fpl_api", "bootstrap", stamp="20260810T000000Z")["price"] == 5.0  # inclusive

    with pytest.raises(FileNotFoundError):
        read_raw("fpl_api", "bootstrap", stamp="20260809T000000Z")  # before any snapshot


def test_missing_table_raises(tmp_config):
    with pytest.raises(FileNotFoundError):
        read_table("interim", "does_not_exist")


def test_partition_report_reveals_a_table_holding_two_generations(tmp_config):
    """The trap this exists for: partitions are refreshed one at a time, a stale one survives
    a model change, and reading the table back concatenates two generations silently. That
    happened — a stale 2022-23 partition made a re-measured calibration look 12% worse than
    it was, and sent me hunting a bug in the model instead of in the data.
    """
    import time

    from fpl_expert.data.storage import partition_report, read_table, write_table

    write_table(pd.DataFrame({"x": [1, 2]}), "processed", "demo", season="2023-24")
    time.sleep(1.05)                       # stamps are second-resolution
    write_table(pd.DataFrame({"x": [3]}), "processed", "demo", season="2024-25")

    report = partition_report("processed", "demo")
    assert len(report) == 2
    assert report["written_at"].nunique() == 2          # visibly two generations
    assert report["rows"].sum() == len(read_table("processed", "demo"))
    # Newest first, so the freshly written partition leads.
    assert "2024-25" in report["partition"].iloc[0]


def test_partition_report_on_a_single_write_shows_one_generation(tmp_config):
    from fpl_expert.data.storage import partition_report, write_table

    write_table(pd.DataFrame({"x": [1]}), "processed", "solo", season="2025-26")
    assert partition_report("processed", "solo")["written_at"].nunique() == 1
