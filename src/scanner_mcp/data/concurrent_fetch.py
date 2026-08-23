"""Bounded-concurrency helper for fetching multiple tickers' history at once."""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator

from scanner_mcp.data.provider import DataProvider

log = logging.getLogger(__name__)

_DEFAULT_FETCH_WORKERS = 8


def _resolve_fetch_workers(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, explicit)
    return max(1, int(os.environ.get("SCANNER_MCP_FETCH_WORKERS", str(_DEFAULT_FETCH_WORKERS))))


def fetch_histories_concurrently(
    provider: DataProvider,
    requests: list[tuple[Any, dict[str, Any]]],
    *,
    max_workers: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[tuple[Any, Any, Exception | None]]:
    """Fetch `provider.get_history(key, **kwargs)` for each request, in parallel.

    Yields `(key, df, None)` on success or `(key, None, exc)` on failure, in
    completion order (not submission order). Callers that need a stable order
    should key their own bookkeeping by `key` rather than relying on yield order.

    `cancel_check`, if given, is checked before submitting any work and again
    before each yield; once it returns truthy, no further results are yielded
    and any not-yet-started futures are cancelled (already-running fetches are
    left to finish in the background rather than blocking the caller).
    """
    if not requests:
        return
    if cancel_check and cancel_check():
        return

    workers = _resolve_fetch_workers(max_workers)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scanner-mcp-fetch")
    try:
        future_to_key: dict[Future[Any], Any] = {
            executor.submit(provider.get_history, key, **kwargs): key for key, kwargs in requests
        }
        for future in as_completed(future_to_key):
            if cancel_check and cancel_check():
                for pending in future_to_key:
                    pending.cancel()
                return
            key = future_to_key[future]
            try:
                df = future.result()
            except Exception as exc:  # noqa: BLE001
                yield key, None, exc
                continue
            yield key, df, None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
