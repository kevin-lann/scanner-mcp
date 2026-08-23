"""FastMCP entry point: tool handlers and resources only."""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from typing import Annotated, Any, Literal

import anyio
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from scanner_mcp.charts.service import chart_tool_result
from scanner_mcp.data.movers import screen_movers
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.indicators.core import Indicators
from scanner_mcp.market.service import (
    as_float,
    compute_indicators,
    first_present,
    market_snapshot as build_market_snapshot,
    quote_snapshot,
)
from scanner_mcp.mcp_schemas import YFINANCE_PERIOD_DESC
from scanner_mcp.research.forward_returns import forward_returns_markdown
from scanner_mcp.runtime import (
    SCAN_TRIGGER_SECRET_ENV,
    SCAN_TRIGGER_SECRET_HEADER,
    configure_logging,
    get_provider,
    get_request_user_id,
    get_scheduler_mode,
    get_store,
    install_signal_handlers,
    lifespan,
    restore_signal_handlers,
    scan_trigger_authorized,
    start_scan_job,
    shutdown_scheduler,
    scan_job_payload,
)
from scanner_mcp.scanner import scheduler as scan_sched
from scanner_mcp.scanner.scheduler import ET
from scanner_mcp.signals.catalog import list_catalog_entries
from scanner_mcp.signals.service import create_signal_payload, run_scan_payload

log = logging.getLogger(__name__)

mcp = FastMCP("HorusMCP", lifespan=lifespan)


@mcp.tool()
def debug_quote(symbol: str) -> str:
    """Debug yfinance `fast_info` keys and daily-history fallback for one symbol.

    `symbol`: yfinance ticker (e.g. `AAPL`, `^GSPC`, `BTC-USD`).
    """
    provider = get_provider()
    info = provider.get_fast_info(symbol) or {}
    df = provider.get_history(symbol, period="10d", interval="1d")
    quote = quote_snapshot(provider, symbol)
    hist_tail: dict[str, Any] | None = None
    if df is not None and not df.empty:
        row = df.tail(1).iloc[0]
        hist_tail = {str(key): as_float(value) for key, value in row.items()}
    return json.dumps(
        {
            "symbol": symbol.upper(),
            "server_file": __file__,
            "fast_info_keys": sorted(info.keys()),
            "fast_info_sample": {
                "last_price": first_present(info, "last_price", "lastPrice"),
                "previous_close": first_present(info, "previous_close", "previousClose"),
                "regular_market_previous_close": first_present(info, "regular_market_previous_close", "regularMarketPreviousClose"),
                "regular_market_change_percent": first_present(info, "regular_market_change_percent", "regularMarketChangePercent"),
                "last_volume": first_present(info, "last_volume", "lastVolume"),
            },
            "history": {
                "empty": df is None or df.empty,
                "rows": 0 if df is None else len(df),
                "columns": [] if df is None else list(df.columns),
                "last_index": None if df is None or df.empty else str(df.index[-1]),
                "tail": hist_tail,
            },
            "resolved": quote,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def get_price(symbol: str) -> str:
    """Current / last price, day change, volume, and market cap from yfinance `fast_info` (with history fallback).

    `symbol`: yfinance ticker string (e.g. `AAPL`, `^VIX`, `BTC-USD`).
    """
    provider = get_provider()
    info = provider.get_fast_info(symbol)
    quote = quote_snapshot(provider, symbol)
    if not info and quote["last_price"] is None:
        return json.dumps({"error": "no data", "symbol": symbol})
    last = quote["last_price"]
    prev = quote["previous_close"]
    day_change = as_float(first_present(info, "day_change", "dayChange")) if info else None
    if day_change is None and last is not None and prev is not None:
        day_change = last - prev
    payload = {
        "symbol": symbol.upper(),
        "last_price": last,
        "previous_close": prev,
        "day_change": day_change,
        "day_change_pct": quote["day_change_pct"],
        "volume": first_present(info, "last_volume", "lastVolume", "shares_unlisted", "sharesUnlisted"),
        "market_cap": first_present(info, "market_cap", "marketCap"),
    }
    return json.dumps(payload, indent=2, default=str)


def _fetch_quote(provider: DataProvider, symbol: str) -> dict[str, Any]:
    info = provider.get_fast_info(symbol)
    quote = quote_snapshot(provider, symbol)
    if not info and quote["last_price"] is None:
        return {"symbol": symbol.upper(), "error": "no data"}
    last = quote["last_price"]
    prev = quote["previous_close"]
    day_change = as_float(first_present(info, "day_change", "dayChange")) if info else None
    if day_change is None and last is not None and prev is not None:
        day_change = last - prev
    return {
        "symbol": symbol.upper(),
        "last_price": last,
        "previous_close": prev,
        "day_change": day_change,
        "day_change_pct": quote["day_change_pct"],
        "volume": first_present(info, "last_volume", "lastVolume", "shares_unlisted", "sharesUnlisted"),
        "market_cap": first_present(info, "market_cap", "marketCap"),
    }


@mcp.tool()
def get_quotes(symbols: list[str]) -> str:
    """Batch version of `get_price` for multiple symbols in a single call — prefer this over looping `get_price`.

    Fetches symbols concurrently (each is a blocking yfinance network call), so latency stays close to a single
    symbol's fetch time instead of the sum across the whole list.

    `symbols`: list of yfinance ticker strings (e.g. `["AAPL", "TSLA"]`).
    Returns JSON text: array of `{symbol, last_price, previous_close, day_change, day_change_pct, volume, market_cap}`,
    or `{symbol, error}` for any ticker with no data. Order matches the input `symbols` order.
    """
    provider = get_provider()
    if not symbols:
        return json.dumps([])
    with ThreadPoolExecutor(max_workers=min(24, len(symbols))) as pool:
        quotes = list(pool.map(lambda symbol: _fetch_quote(provider, symbol), symbols))
    return json.dumps(quotes, indent=2, default=str)


_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
# Widen the fetch only when the requested intraday day has no data (weekend /
# holiday / pre-market) so we can fall back to the previous active session.
_INTRADAY_FALLBACK_PERIOD = "7d"


def _fetch_price_history(
    provider: DataProvider, symbol: str, period: str, interval: str
) -> dict[str, Any]:
    df = provider.get_history(symbol, period=period, interval=interval)
    if interval in _INTRADAY_INTERVALS and period == "1d" and (df is None or getattr(df, "empty", True)):
        df = provider.get_history(
            symbol, period=_INTRADAY_FALLBACK_PERIOD, interval=interval
        )
    if df is None or getattr(df, "empty", True) or "Close" not in getattr(df, "columns", []):
        return {"symbol": symbol.upper(), "error": "no data"}
    # Only single-day requests collapse to the most recent trading session
    # (today's, or the previous active day when the fallback window kicked
    # in). Multi-day periods at an intraday interval (e.g. `5d`/`15m`) keep
    # every session the provider returned.
    if interval in _INTRADAY_INTERVALS and period == "1d":
        try:
            last_day = df.index[-1].date()
            df = df[df.index.date == last_day]
        except Exception:  # noqa: BLE001 - non-datetime index; fall back to full frame
            pass
    closes = [c for c in (as_float(v) for v in df["Close"].tolist()) if c is not None]
    if len(closes) < 2:
        return {"symbol": symbol.upper(), "error": "no data"}
    first, last = closes[0], closes[-1]
    change_pct = ((last - first) / first * 100.0) if first else None
    return {
        "symbol": symbol.upper(),
        "closes": [round(c, 4) for c in closes],
        "change_pct": round(change_pct, 4) if change_pct is not None else None,
    }


@mcp.tool()
def get_price_histories(
    symbols: list[str],
    period: str = "3mo",
    interval: str = "1d",
) -> str:
    """Batch daily close-price series for lightweight sparkline / mini-chart rendering.

    Returns only the sequence of closing prices per symbol (no OHLC / volume), keeping the
    payload small enough to fetch for a whole watchlist at once. Symbols are fetched
    concurrently; daily history is cached ~6h upstream, so warm calls return immediately.

    `symbols`: list of yfinance ticker strings (e.g. `["AAPL", "TSLA"]`).
    `period`: yfinance history window for the series, e.g. `1mo`, `3mo`, `6mo`, `1y` (default `3mo`).
    `interval`: bar interval; `1d` (default) for a daily chart, or an intraday interval
    (`1m`, `5m`, `15m`, `60m`, ...) for a single-session chart. For intraday intervals only one
    trading session is returned; pass `period="1d"` and it auto-falls-back to the previous active
    session outside market hours (weekends / holidays).
    Returns JSON text: array of `{symbol, closes: number[], change_pct}` (oldest→newest), or
    `{symbol, error}` for any ticker with no data. Order matches the input `symbols` order.
    """
    provider = get_provider()
    if not symbols:
        return json.dumps([])
    with ThreadPoolExecutor(max_workers=min(24, len(symbols))) as pool:
        histories = list(
            pool.map(
                lambda symbol: _fetch_price_history(provider, symbol, period, interval),
                symbols,
            )
        )
    return json.dumps(histories, default=str)


def _fetch_price_history_ohlc(
    provider: DataProvider, symbol: str, period: str, interval: str
) -> dict[str, Any]:
    df = provider.get_history(symbol, period=period, interval=interval)
    if interval in _INTRADAY_INTERVALS and period == "1d" and (df is None or getattr(df, "empty", True)):
        df = provider.get_history(
            symbol, period=_INTRADAY_FALLBACK_PERIOD, interval=interval
        )
    if df is None or getattr(df, "empty", True) or "Close" not in getattr(df, "columns", []):
        return {"symbol": symbol.upper(), "error": "no data"}
    # Only single-day requests collapse to the most recent trading session
    # (today's, or the previous active day when the fallback window kicked
    # in). Multi-day periods at an intraday interval (e.g. `5d`/`15m`) keep
    # every session the provider returned.
    if interval in _INTRADAY_INTERVALS and period == "1d":
        try:
            last_day = df.index[-1].date()
            df = df[df.index.date == last_day]
        except Exception:  # noqa: BLE001 - non-datetime index; fall back to full frame
            pass
    timestamps: list[str] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[int] = []
    for ts, row in df.iterrows():
        o = as_float(row.get("Open"))
        h = as_float(row.get("High"))
        l = as_float(row.get("Low"))
        c = as_float(row.get("Close"))
        if o is None or h is None or l is None or c is None:
            continue
        timestamps.append(ts.isoformat() if hasattr(ts, "isoformat") else str(ts))
        opens.append(round(o, 4))
        highs.append(round(h, 4))
        lows.append(round(l, 4))
        closes.append(round(c, 4))
        v = as_float(row.get("Volume"))
        volumes.append(int(v) if v is not None else 0)
    if len(closes) < 2:
        return {"symbol": symbol.upper(), "error": "no data"}
    return {
        "symbol": symbol.upper(),
        "timestamps": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }


@mcp.tool()
def get_price_histories_ohlc(
    symbols: list[str],
    period: str = "3mo",
    interval: str = "1d",
) -> str:
    """Batch OHLCV history series for candlestick charting.

    Unlike `get_price_histories` (close-only, for sparklines), this returns full
    open/high/low/close/volume bars per symbol so a candlestick chart can render
    wicks and bodies. Symbols are fetched concurrently; daily history is cached
    ~6h upstream, so warm calls return immediately.

    `symbols`: list of yfinance ticker strings (e.g. `["AAPL", "TSLA"]`).
    `period`: yfinance history window for the series, e.g. `1mo`, `3mo`, `6mo`, `1y` (default `3mo`).
    `interval`: bar interval; `1d` (default) for a daily chart, or an intraday interval
    (`1m`, `5m`, `15m`, `60m`, ...) for a single-session chart. For intraday intervals only one
    trading session is returned; pass `period="1d"` and it auto-falls-back to the previous active
    session outside market hours (weekends / holidays).
    Returns JSON text: array of `{symbol, timestamps: string[], open: number[], high: number[],
    low: number[], close: number[], volume: number[]}` (oldest→newest, ISO8601 timestamps), or
    `{symbol, error}` for any ticker with no data. Order matches the input `symbols` order.
    """
    provider = get_provider()
    if not symbols:
        return json.dumps([])
    with ThreadPoolExecutor(max_workers=min(24, len(symbols))) as pool:
        histories = list(
            pool.map(
                lambda symbol: _fetch_price_history_ohlc(provider, symbol, period, interval),
                symbols,
            )
        )
    return json.dumps(histories, default=str)


@mcp.tool()
def get_indicators(symbol: str, indicators: list[str], period: str = "6mo") -> str:
    """Compute indicators with buy/hold/sell rating each plus consensus.

    `symbol`: yfinance ticker.
    `indicators`: each element is one token — `rsi` or `rsi:<n>` (length `n`, default 14); `macd`;
    `bbands`; `sma` or `sma:<n>` (default 50); `ema` or `ema:<n>` (default 20); `ath_distance`; `beta`.
    `period`: yfinance history window for daily bars (`1d` interval), e.g. `1mo`, `6mo`, `1y`, `5y`, `max` (default `6mo`).
    """
    result = compute_indicators(get_provider(), symbol, indicators, period)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def get_ath_distance(symbol: str) -> str:
    """Percent distance of last daily close below the running all-time high (full `max` history window).

    `symbol`: yfinance ticker.
    """
    df = get_provider().get_history(symbol, period="max", interval="1d")
    if df is None or getattr(df, "empty", True):
        return json.dumps({"error": "no data"})
    distance = Indicators(df).ath_distance()
    return json.dumps({"symbol": symbol.upper(), "pct_from_ath": distance})


@mcp.tool()
def get_option_chain(symbol: str, expiry: str | None = None) -> str:
    """Options chain preview as plain text (up to ~20 rows per calls/puts).

    `symbol`: underlying yfinance ticker.
    `expiry`: `YYYY-MM-DD` for a specific expiry, or omit / null to use the first listed expiry.
    """
    result = get_provider().get_option_chain(symbol, expiry)
    if result.get("error"):
        return f"Error: {result['error']}\nExpiries: {result.get('expiries', [])[:10]}"
    buffer = StringIO()
    calls = result.get("calls")
    puts = result.get("puts")
    if calls is not None and not calls.empty:
        buffer.write("=== CALLS ===\n")
        buffer.write(calls.head(20).to_string())
        buffer.write("\n\n")
    if puts is not None and not puts.empty:
        buffer.write("=== PUTS ===\n")
        buffer.write(puts.head(20).to_string())
    return buffer.getvalue() or "empty chain"


@mcp.tool()
def market_snapshot() -> str:
    """Major US indices, ETFs, crypto, VIX: `last_price` and `day_change_pct` per symbol (no arguments).

    Buckets: `us_indices`, `etfs`, `crypto`, `volatility` (fixed symbol lists in server config).
    """
    return json.dumps(build_market_snapshot(get_provider()), indent=2, default=str)


@mcp.tool()
def top_gainers(exchange: str, limit: int = 20) -> str:
    """Top daily percentage gainers for an equity exchange or a fixed crypto pair list.

    `exchange`: `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (case-insensitive; `CRYPTO` ranks ~20 liquid USD pairs).
    `limit`: max rows (default 20).
    """
    try:
        return json.dumps(screen_movers("gainers", exchange, limit=limit), default=str, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


@mcp.tool()
def top_losers(exchange: str, limit: int = 20) -> str:
    """Top daily percentage losers; same `exchange` and `limit` semantics as `top_gainers`."""
    try:
        return json.dumps(screen_movers("losers", exchange, limit=limit), default=str, indent=2)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


@mcp.tool()
def list_signal_catalog() -> str:
    """List predefined `signal_type` keys with descriptions and `default_params` (input for `create_signal`)."""
    return json.dumps(list_catalog_entries(), indent=2, default=str)


@mcp.tool()
def create_signal(
    name: str,
    signal_type: str,
    params: dict[str, Any] | None = None,
    ticker_scope: Literal["tickers", "watchlist", "exchange"] = "watchlist",
    ticker_overrides: list[str] | None = None,
    exchange: str | None = None,
    history_period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    interval: Annotated[str, Field(description="yfinance bar size for signal evaluation, e.g. 1d, 1wk, 1mo.")] = "1d",
    scan_time: Annotated[str, Field(description="Daily scheduled scan time, 24-hour HH:MM Eastern Time, e.g. 16:30.")] = "16:30",
) -> str:
    """Create a persisted enabled signal row.

    `name`: human-readable label (not necessarily unique).
    `signal_type`: must match a catalog key from `list_signal_catalog`.
    `params`: overrides merged on top of that type's catalog defaults (JSON object; use numbers not strings).
    `ticker_scope`: `watchlist` (scan the global watchlist; omit `ticker_overrides` and `exchange`),
    `tickers` (set non-empty `ticker_overrides`; omit `exchange`), or `exchange` (set `exchange`; omit overrides).
    `ticker_overrides`: required when scope is `tickers` — list of ticker strings.
    `exchange`: when scope is `exchange`, exactly `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (omit otherwise).
    `history_period`: yfinance history window fetched for this signal, e.g. `1y`, `2y`, `5y`, `max`.
    `interval`: yfinance bar interval for this signal, e.g. `1d`, `1wk`, `1mo`.
    `scan_time`: time of day the scheduled scan runs for this signal, 24-hour `HH:MM` Eastern Time.
    """
    return create_signal_payload(
        get_store(),
        get_request_user_id(),
        name,
        signal_type,
        params,
        ticker_scope,
        ticker_overrides,
        exchange,
        history_period,
        interval,
        scan_time,
    )


@mcp.tool()
def list_signals() -> str:
    """List configured signals (includes `id` for `delete_signal` / `run_scan`)."""
    user_id = get_request_user_id()
    rows = get_store().signal_list(user_id)
    payload = [
        {
            "id": row.id,
            "name": row.name,
            "signal_type": row.signal_type,
            "params": row.params,
            "ticker_scope": row.ticker_scope,
            "ticker_overrides": row.ticker_overrides,
            "exchange": row.exchange,
            "history_period": row.history_period,
            "interval": row.interval,
            "scan_time": row.scan_time,
            "enabled": row.enabled,
        }
        for row in rows
    ]
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
def delete_signal(signal_id: int) -> str:
    """Delete a signal and its alert history. `signal_id` is the integer `id` from `list_signals`."""
    user_id = get_request_user_id()
    return json.dumps({"ok": get_store().signal_delete(user_id, signal_id)})


@mcp.tool()
def run_scan(
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Run an on-demand scan synchronously and return only triggered rows.

    Prefer `start_scan` for large universes or many signal types. This tool blocks until the
    scan completes, so MCP clients may time out on long runs. Use it only when you expect the
    scan to finish quickly.

    Universe — pass **at most one** of:
    - omit `symbol`, `tickers`, and `exchange`: each enabled signal uses its saved scope / overrides / exchange;
    - `symbol`: single ticker string (e.g. `NVDA`);
    - `tickers`: list of ticker strings;
    - `exchange`: `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO` (full Yahoo screener list for equities; fixed crypto list for `CRYPTO`).

    `signal_id`: optional; when set, only that **enabled** signal runs (ignored when `all_signal_types` is true).
    `all_signal_types`: when true together with exactly one universe selector above, run **every** catalog
    signal type against that universe (`signal_id` must be omitted); uses 1 year of daily bars per symbol.
    """
    user_id = get_request_user_id()
    return run_scan_payload(get_store(), get_provider(), user_id, signal_id, tickers, all_signal_types, symbol, exchange)


@mcp.tool()
def start_scan(
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Start a background scan job and return immediately with a persistent `job_id`.

    Preferred client flow for long scans:
    1. Call `start_scan(...)`.
    2. Read `job_id` from the JSON response.
    3. Poll `get_scan_status(job_id)` until `status` is `completed`, `failed`, or `cancelled`.
    4. When `status` is `completed`, call `get_scan_result(job_id)` exactly once or as needed.

    This is the recommended tool for MCP clients such as Claude Desktop because it avoids
    request timeouts. The server stores progress and final results in SQLite so the client can
    reconnect later and continue polling with the same `job_id`.

    Universe — pass **at most one** of:
    - omit `symbol`, `tickers`, and `exchange`: each enabled signal uses its saved scope / overrides / exchange;
    - `symbol`: single ticker string (e.g. `NVDA`);
    - `tickers`: list of ticker strings;
    - `exchange`: `NYSE`, `NASDAQ`, `AMEX`, or `CRYPTO`.

    `signal_id`: optional; when set, only that enabled signal runs (ignored when `all_signal_types` is true).
    `all_signal_types`: when true together with exactly one universe selector above, run every catalog
    signal type against that universe (`signal_id` must be omitted).
    """
    user_id = get_request_user_id()
    job_id = start_scan_job(
        user_id=user_id,
        signal_id=signal_id,
        tickers=tickers,
        all_signal_types=all_signal_types,
        symbol=symbol,
        exchange=exchange,
    )
    payload = scan_job_payload(job_id, user_id)
    payload["poll_after_seconds"] = 2
    payload["next_action"] = (
        f"Poll get_scan_status({job_id}) until status is completed, failed, or cancelled. "
        f"Then call get_scan_result({job_id}) if completed."
    )
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
def get_scan_status(job_id: int) -> str:
    """Return scan job status and progress counters for a previously started background scan.

    Client behavior:
    - Keep polling while `status` is `queued` or `running`.
    - Stop polling when `status` is `completed`, `failed`, or `cancelled`.
    - If `status` is `completed`, call `get_scan_result(job_id)` to retrieve the stored results.

    Counters:
    - `checked_count`: symbol/signal evaluations completed so far.
    - `total_count`: estimated total evaluations, when known.
    - `fired_count`: number of triggered matches found so far.
    - `result_count`: number of rows that will appear in `get_scan_result`.
    """
    user_id = get_request_user_id()
    payload = scan_job_payload(job_id, user_id)
    if "error" not in payload and payload["status"] in {"queued", "running"}:
        payload["poll_after_seconds"] = 2
        payload["next_action"] = f"Poll get_scan_status({job_id}) again after a short delay."
    elif "error" not in payload and payload["status"] == "completed":
        payload["next_action"] = f"Call get_scan_result({job_id}) to read the final stored results."
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
def get_scan_result(job_id: int, limit: int | None = None, offset: int = 0) -> str:
    """Return the stored final result for a completed background scan.

    Use this only after `get_scan_status(job_id)` reports `status=completed`. This tool never
    reruns the scan; it only reads the persisted result from SQLite.

    Pagination:
    - `offset`: zero-based row offset into the stored `results` array (default 0).
    - `limit`: optional maximum number of rows to return from the stored `results` array.

    If the job is not completed yet, this tool returns a small status payload instead of partial results.
    """
    user_id = get_request_user_id()
    row = get_store().scan_job_get(user_id, job_id)
    if row is None:
        return json.dumps({"error": "scan job not found", "job_id": job_id})
    if row.status != "completed":
        payload = scan_job_payload(job_id, user_id)
        payload["next_action"] = f"Poll get_scan_status({job_id}) until status=completed, then retry get_scan_result({job_id})."
        return json.dumps(payload, indent=2, default=str)
    result = dict(row.result or {})
    rows = list(result.get("results", []))
    start = max(0, offset)
    stop = None if limit is None else max(start, start + max(0, limit))
    sliced = rows[start:stop]
    result["job_id"] = job_id
    result["status"] = row.status
    result["offset"] = start
    result["returned_count"] = len(sliced)
    result["total_result_count"] = len(rows)
    result["results"] = sliced
    if stop is not None and stop < len(rows):
        result["next_offset"] = stop
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def cancel_scan(job_id: int) -> str:
    """Request cancellation for a queued or running background scan job.

    Cancellation is best-effort and cooperative. After calling this tool, poll
    `get_scan_status(job_id)` until the job reaches `cancelled`, `completed`, or `failed`.
    """
    user_id = get_request_user_id()
    ok = get_store().scan_job_request_cancel(user_id, job_id)
    payload = scan_job_payload(job_id, user_id)
    payload["cancel_request_accepted"] = ok
    payload["next_action"] = f"Poll get_scan_status({job_id}) to observe the terminal state."
    return json.dumps(payload, indent=2, default=str)


@mcp.tool()
def add_to_watchlist(symbols: list[str]) -> str:
    """`add_to_watchlist`: append ticker strings supplied in `symbols` to the global watchlist.

    `symbols` is a native Python list of tickers (`list[str]`); under MCP structured tool calls this is encoded as a JSON array.
    Returns JSON text: `{"added": [...]}` where `added` lists tickers inserted in this invocation (symbols already stored are omitted).
    """
    user_id = get_request_user_id()
    return json.dumps({"added": get_store().watchlist_add(user_id, [str(value) for value in symbols])})


@mcp.tool()
def remove_from_watchlist(symbols: list[str]) -> str:
    """Remove symbols from the watchlist (`symbols`: array of tickers)."""
    user_id = get_request_user_id()
    return json.dumps({"removed": get_store().watchlist_remove(user_id, [str(value) for value in symbols])})


@mcp.tool()
def get_watchlist() -> str:
    """Return the current watchlist as a JSON array of ticker strings."""
    user_id = get_request_user_id()
    return json.dumps([row.symbol for row in get_store().watchlist_get(user_id)], indent=2)


@mcp.tool()
def get_watchlist_detail() -> str:
    """Return the current watchlist with metadata (JSON array of `{symbol, added_at}`, sorted by symbol)."""
    user_id = get_request_user_id()
    rows = get_store().watchlist_get(user_id)
    return json.dumps([{"symbol": row.symbol, "added_at": row.added_at} for row in rows], indent=2)


@mcp.tool()
def chart_price_history(
    symbol: str = "SPY",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    interval: Annotated[str, Field(description="yfinance bar size (e.g. 1d, 1h); intraday couples to allowed period ranges")] = "1d",
    show_ma: Annotated[bool, Field(description="Overlay a simple moving average.")] = False,
    ma_period: Annotated[int, Field(description="SMA period used when show_ma is true.")] = 50,
    show_bollinger_bands: Annotated[bool, Field(description="Overlay Bollinger Bands.")] = False,
    bb_period: Annotated[int, Field(description="Bollinger Band moving-average period.")] = 20,
    bb_std: Annotated[float, Field(description="Bollinger Band standard-deviation multiplier.")] = 2.0,
    show_ema: Annotated[bool, Field(description="Overlay an exponential moving average.")] = False,
    ema_period: Annotated[int, Field(description="EMA period used when show_ema is true.")] = 21,
    show_ma_cloud: Annotated[bool, Field(description="Overlay a filled SMA cloud between fast and slow averages.")] = False,
    ma_cloud_fast: Annotated[int, Field(description="Fast SMA period used for the MA cloud.")] = 50,
    ma_cloud_slow: Annotated[int, Field(description="Slow SMA period used for the MA cloud.")] = 200,
    show_fib_retracement: Annotated[bool, Field(description="Overlay Fibonacci retracement levels from the visible high/low swing.")] = False,
    show_avwap: Annotated[bool, Field(description="Overlay anchored VWAP from the first visible bar, or avwap_anchor when provided.")] = False,
    avwap_anchor: Annotated[str | None, Field(description="Optional aVWAP anchor date/time parseable by pandas, for example 2025-01-02.")] = None,
    pe_subchart: Annotated[bool, Field(description="When true, add a lower panel for P/E history.")] = False,
) -> Image | str:
    """Candlestick price history chart. Returns PNG image; on failure JSON text with `error`."""
    return chart_tool_result(
        get_provider(),
        "price_history",
        {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "show_ma": show_ma,
            "ma_period": ma_period,
            "show_bollinger_bands": show_bollinger_bands,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "show_ema": show_ema,
            "ema_period": ema_period,
            "show_ma_cloud": show_ma_cloud,
            "ma_cloud_fast": ma_cloud_fast,
            "ma_cloud_slow": ma_cloud_slow,
            "show_fib_retracement": show_fib_retracement,
            "show_avwap": show_avwap,
            "avwap_anchor": avwap_anchor,
            "pe_subchart": pe_subchart,
        },
    )


@mcp.tool()
def chart_price_overlay(
    symbols: list[str] | None = None,
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    normalize: bool = True,
) -> Image | str:
    """Multi-symbol line overlay (default symbols SPY and QQQ if `symbols` omitted). Normalized to 100 when `normalize`."""
    params: dict[str, Any] = {"period": period, "normalize": normalize}
    if symbols is not None:
        params["symbols"] = symbols
    return chart_tool_result(get_provider(), "price_overlay", params)


@mcp.tool()
def chart_ratio(
    symbol: str = "SPY",
    benchmark: str = "XLP",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
) -> Image | str:
    """Single-line ratio chart of one asset divided by another asset."""
    return chart_tool_result(get_provider(), "ratio_chart", {"symbol": symbol, "benchmark": benchmark, "period": period})


@mcp.tool()
def chart_relative_strength(
    symbol: str = "AAPL",
    benchmark: str = "SPY",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "2y",
    ma_period: int = 50,
) -> Image | str:
    """Stock-vs-benchmark ratio with moving average and leadership shading."""
    return chart_tool_result(get_provider(), "relative_strength", {"symbol": symbol, "benchmark": benchmark, "period": period, "ma_period": ma_period})


@mcp.tool()
def chart_sector_rotation(
    symbols: list[str] | None = None,
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "2y",
    return_window: int = 63,
) -> Image | str:
    """Normalized sector/ETF comparison with a rolling return panel."""
    params: dict[str, Any] = {"period": period, "return_window": return_window}
    if symbols is not None:
        params["symbols"] = symbols
    return chart_tool_result(get_provider(), "sector_rotation", params)


@mcp.tool()
def chart_fundamental_overlay(
    symbol: str = "AAPL",
    metric: Literal["revenue", "earnings"] = "revenue",
    frequency: Literal["quarterly", "annual"] = "quarterly",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "5y",
    interval: Annotated[str, Field(description="yfinance price bar size, usually 1d for this chart.")] = "1d",
    price_style: Literal["candlestick", "line"] = "line",
) -> Image | str:
    """Price chart overlaid with income-statement bars for revenue or earnings.

    `symbol`: yfinance ticker.
    `metric`: `revenue` uses Total Revenue; `earnings` uses Net Income.
    `frequency`: quarterly by default; uses Alpha Vantage historical statements when configured, with Yahoo statement fallback.
    Returns PNG image; on failure JSON text with `error`.
    """
    return chart_tool_result(get_provider(), "fundamental_overlay", {"symbol": symbol, "metric": metric, "frequency": frequency, "period": period, "interval": interval, "price_style": price_style})


@mcp.tool()
def chart_fundamental_momentum(
    symbol: str = "AAPL",
    frequency: Literal["quarterly", "annual"] = "quarterly",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "5y",
    interval: Annotated[str, Field(description="yfinance price bar size, usually 1d for this chart.")] = "1d",
    profitability_metric: Literal["net_margin", "earnings_growth"] = "net_margin",
    price_style: Literal["candlestick", "line"] = "line",
) -> Image | str:
    """Multi-panel price + fundamentals chart: revenue growth, profitability, and valuation.

    Panel 1: price. Panel 2: YoY revenue growth. Panel 3: net margin or earnings growth.
    Panel 4: historical P/E series.
    """
    return chart_tool_result(
        get_provider(),
        "fundamental_momentum",
        {
            "symbol": symbol,
            "frequency": frequency,
            "period": period,
            "interval": interval,
            "profitability_metric": profitability_metric,
            "price_style": price_style,
        },
    )


@mcp.tool()
def chart_forward_returns(
    symbol: str = "SPY",
    event_type: Literal["golden_cross", "macd_bullish_crossover", "pct_from_ma", "rsi_oversold", "rsi_overbought"] = "rsi_oversold",
    windows: list[int] | None = None,
    event_params: Annotated[dict[str, Any] | None, Field(description='Optional event detector parameters. For pct_from_ma, use {"ma_type":"ema","ma_period":200,"pct":3}.')] = None,
    signal_dates: Annotated[list[str] | None, Field(description="Optional explicit signal dates in YYYY-MM-DD format. When provided, these dates are used instead of calculating events from event_type. Non-trading dates map to the next trading session in the 10y daily history window.")] = None,
) -> Image | str:
    """Price chart with signal markers plus a forward-return table after historical signal events.

    Uses ~10y daily history; `period` is not configurable.
    `windows`: forward horizons in **trading bars** along the daily close series
    (default 5, 10, 21, 42, 63, 126, 252; about 1-6 and 12 months).
    Supported `event_type`: rsi_oversold, rsi_overbought, golden_cross,
    macd_bullish_crossover, pct_from_ma.
    `event_params`: optional event-specific detector parameters. For `pct_from_ma`,
    supported keys are `ma_type` ("sma" or "ema"), `ma_period`, and `pct`.
    `signal_dates`: optional explicit event dates. When present, `event_type` and
    `event_params` are ignored for event detection and the study uses those dates.
    """
    params: dict[str, Any] = {"symbol": symbol, "event_type": event_type}
    if windows is not None:
        params["windows"] = windows
    if event_params is not None:
        params["event_params"] = event_params
    if signal_dates is not None:
        params["signal_dates"] = signal_dates
    return chart_tool_result(get_provider(), "forward_returns", params)


@mcp.tool()
def chart_basket_breadth(
    symbols: list[str] | None = None,
    benchmark: str = "QQQ",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    sma_period: int = 50,
    corr_window: int = 63,
) -> Image | str:
    """Equal-weight basket vs benchmark with rolling correlation and breadth panels.

    Panel 1: basket vs benchmark normalized to 100. Panel 2: rolling correlation.
    Panel 3: count of basket members above their SMA.
    """
    params: dict[str, Any] = {
        "benchmark": benchmark,
        "period": period,
        "sma_period": sma_period,
        "corr_window": corr_window,
    }
    if symbols is not None:
        params["symbols"] = symbols
    return chart_tool_result(get_provider(), "basket_breadth", params)


@mcp.tool()
def chart_pairs_spread(
    symbol: str = "KO",
    benchmark: str = "PEP",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "1y",
    spread_mode: Literal["ratio", "price_spread"] = "ratio",
    z_window: int = 63,
) -> Image | str:
    """Pairs chart with normalized prices, spread/ratio, and z-score bands."""
    return chart_tool_result(
        get_provider(),
        "pairs_spread",
        {
            "symbol": symbol,
            "benchmark": benchmark,
            "period": period,
            "spread_mode": spread_mode,
            "z_window": z_window,
        },
    )


@mcp.tool()
def chart_drawdown_comparison(
    symbols: list[str] | None = None,
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "5y",
) -> Image | str:
    """Underwater (drawdown %) chart vs running high. Default symbols: ^GSPC and QQQ."""
    params: dict[str, Any] = {"period": period}
    if symbols is not None:
        params["symbols"] = symbols
    return chart_tool_result(get_provider(), "drawdown_comparison", params)


@mcp.tool()
def chart_log_cycle(
    symbol: str = "BTC-USD",
    period: Annotated[str, Field(description=YFINANCE_PERIOD_DESC)] = "max",
) -> Image | str:
    """Weekly log10(close) line chart for long-horizon inspection."""
    return chart_tool_result(get_provider(), "log_cycle", {"symbol": symbol, "period": period})


@mcp.resource("signals://triggered/{user_id}", mime_type="application/json")
def resource_triggered(user_id: str) -> str:
    """Last 50 fired alerts for one user (JSON)."""
    rows = get_store().alerts_recent(user_id, 50)
    payload = [
        {
            "id": row.id,
            "signal_id": row.signal_id,
            "symbol": row.symbol,
            "triggered_at": row.triggered_at,
            "details": row.details,
            "source": row.source,
        }
        for row in rows
    ]
    return json.dumps(payload, default=str, indent=2)


@mcp.resource("signals://watchlist/{user_id}", mime_type="application/json")
def resource_watchlist(user_id: str) -> str:
    """Active watchlist tickers for one user (JSON)."""
    return json.dumps([row.symbol for row in get_store().watchlist_get(user_id)])


@mcp.resource("research://forward-returns/{symbol}/{event_type}", mime_type="text/markdown")
def resource_forward_returns(symbol: str, event_type: str) -> str:
    """Markdown summary of mean/median forward returns (7/30/90 trading days) after RSI events.

    URI path `symbol`: yfinance ticker. `event_type`: `rsi_oversold` or `rsi_overbought` (same as `chart_forward_returns`).
    """
    return forward_returns_markdown(get_provider(), symbol, event_type)


@mcp.custom_route("/internal/scan-tick", methods=["POST"])
async def internal_scan_tick(request: Request) -> Response:
    """Run one signal-scan tick, for an external scheduler (e.g. Cloud Scheduler) to call.

    Only meaningful when `SCANNER_MCP_SCHEDULER_MODE=external` (see `runtime.lifespan`),
    where the in-process APScheduler is intentionally not started. Guarded by a shared
    secret (`SCAN_TRIGGER_SECRET`), mandatory in external mode.
    """
    mode = get_scheduler_mode()
    expected_secret = os.environ.get(SCAN_TRIGGER_SECRET_ENV)
    provided_secret = request.headers.get(SCAN_TRIGGER_SECRET_HEADER)
    if not scan_trigger_authorized(provided_secret, expected_secret, mode):
        status = 503 if not (expected_secret or "").strip() else 401
        return JSONResponse({"error": "unauthorized"}, status_code=status)

    at_time = datetime.now(ET).strftime("%H:%M")
    result = await anyio.to_thread.run_sync(
        lambda: scan_sched.run_full_scan(get_store(), get_provider(), notify=True, at_time=at_time)
    )
    return JSONResponse({"ok": True, "at_time": at_time, **result})


def main() -> None:
    """Configure logging and run the FastMCP stdio server."""
    configure_logging()
    log.info("Starting HorusMCP from %s", __file__)
    old_sigint, old_sigterm = install_signal_handlers()
    try:
        mcp.run()
    finally:
        shutdown_scheduler()
        restore_signal_handlers(old_sigint, old_sigterm)


def main_http() -> None:
    """Configure logging and run the FastMCP HTTP server for sidecar deployment."""
    import os

    configure_logging()
    host = os.environ.get("SCANNER_MCP_HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("SCANNER_MCP_HTTP_PORT", "5050"))
    log.info("Starting HorusMCP HTTP server on %s:%s from %s", host, port, __file__)
    old_sigint, old_sigterm = install_signal_handlers()
    try:
        mcp.run(transport="http", host=host, port=port)
    finally:
        shutdown_scheduler()
        restore_signal_handlers(old_sigint, old_sigterm)


if __name__ == "__main__":
    main()
