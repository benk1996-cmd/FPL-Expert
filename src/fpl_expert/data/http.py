"""Polite HTTP client: rate limiting, retry with backoff, and an on-disk cache.

The FPL API is free and unauthenticated, which makes it easy to abuse by accident — the
ownership job (Item 2b) issues ~1,200 requests per run. Every caller goes through here so
that pacing and retry behaviour are defined in exactly one place.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

USER_AGENT = "fpl-expert/0.1 (personal research project)"
RETRY_STATUS = {429, 500, 502, 503, 504}


class RateLimiter:
    """Guarantees a minimum interval between successive requests."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if (remaining := self.min_interval - elapsed) > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


class HttpClient:
    """JSON-over-HTTP client with caching and backoff.

    Args:
        cache_dir: where cached responses live. `None` disables caching.
        cache_ttl_hours: entries older than this are refetched.
        min_interval: seconds between requests.
        max_retries: attempts per URL before raising.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 6.0,
        min_interval: float = 1.0,
        max_retries: int = 4,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl_hours * 3600
        self.limiter = RateLimiter(min_interval)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()}.json.gz"

    def _read_cache(self, url: str) -> Any | None:
        path = self._cache_path(url)
        if path is None or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.cache_ttl:
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_cache(self, url: str, payload: Any) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def get_json(self, url: str, *, use_cache: bool = True, allow_404: bool = False) -> Any | None:
        """Fetch and parse JSON.

        Returns `None` for a 404 when `allow_404` is set — several FPL endpoints
        legitimately 404 rather than returning empty (picks before a deadline, for one).
        """
        if use_cache and (hit := self._read_cache(url)) is not None:
            return hit

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException as exc:  # transport-level failure
                last_error = exc
                log.warning("request failed (%s), attempt %d: %s", url, attempt + 1, exc)
                time.sleep(2**attempt)
                continue

            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code in RETRY_STATUS:
                # Honour Retry-After when the server sends one, else exponential backoff.
                delay = float(resp.headers.get("Retry-After", 2**attempt))
                log.warning("HTTP %d for %s, backing off %.1fs", resp.status_code, url, delay)
                time.sleep(delay)
                continue

            resp.raise_for_status()
            payload = resp.json()
            if use_cache:
                self._write_cache(url, payload)
            return payload

        raise RuntimeError(f"giving up on {url} after {self.max_retries} attempts") from last_error

    def get_text(self, url: str) -> str:
        """Fetch raw text (CSV downloads from the historical archive)."""
        self.limiter.wait()
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
