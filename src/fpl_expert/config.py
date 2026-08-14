"""Typed loading of `config/config.yaml` and `config/scoring_rules.yaml`.

Scoring rules are configuration, never constants in model code — FPL changes them most
seasons (see the 2026/27 BPS retune), and a backtest over multiple seasons needs to score
each season under the rules that were actually in force at the time.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


def project_root() -> Path:
    """Repo root, resolved from this file rather than the cwd."""
    return Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    # Tolerate keys we have not modelled yet, so adding config never breaks loading.
    model_config = ConfigDict(extra="allow")


class Paths(_Base):
    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    external: str = "data/external"


class DataCfg(_Base):
    fpl_api_base: str = "https://fantasy.premierleague.com/api"
    history_seasons: list[str] = []
    request_delay_seconds: float = 1.0
    cache_ttl_hours: float = 6.0


class ObjectiveCfg(_Base):
    lambda_rank: float = 0.0
    target_rank: int | None = None
    field_size: int = 11_000_000


class OptimiseCfg(_Base):
    horizon_gws: int = 6
    future_decay: float = 0.84
    transfer_hit_cost: int = 4
    solver: str = "HiGHS"
    objective: ObjectiveCfg = ObjectiveCfg()


class OwnershipCfg(_Base):
    overall_league_id: int = 314
    top_n_managers: int = 10_000
    sample_size: int = 1_000
    request_delay_seconds: float = 0.5
    min_gw_for_top10k: int = 6


class Config(_Base):
    project: dict = {}
    paths: Paths = Paths()
    data: DataCfg = DataCfg()
    model: dict = {}
    optimise: OptimiseCfg = OptimiseCfg()
    ownership: OwnershipCfg = OwnershipCfg()
    backtest: dict = {}

    def path(self, layer: str) -> Path:
        """Absolute path for a data layer ('raw', 'interim', 'processed', 'external')."""
        rel = getattr(self.paths, layer)
        p = project_root() / rel
        p.mkdir(parents=True, exist_ok=True)
        return p


def _read_yaml(path: Path) -> dict:
    # Always explicit about encoding: the default on Windows is cp1252 and the FPL feed
    # contains non-latin-1 player names.
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    return Config(**_read_yaml(path or project_root() / "config" / "config.yaml"))


@functools.lru_cache(maxsize=1)
def load_scoring_rules(path: Path | None = None) -> dict:
    """Raw dict — the scoring table is data, and Item 10 consumes it as such."""
    return _read_yaml(path or project_root() / "config" / "scoring_rules.yaml")
