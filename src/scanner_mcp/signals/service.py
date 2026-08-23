"""Signal validation and scan execution helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from scanner_mcp.data.concurrent_fetch import fetch_histories_concurrently
from scanner_mcp.data.exchange_universe import fetch_exchange_tickers
from scanner_mcp.data.provider import DataProvider
from scanner_mcp.db.store import DEFAULT_USER_ID, SignalRow, Store
from scanner_mcp.signals.catalog import CATALOG, merge_params
from scanner_mcp.signals.evaluator import evaluate
from scanner_mcp.signals.models import ActiveSignal

EXCHANGES = frozenset({"NYSE", "NASDAQ", "AMEX", "CRYPTO"})
ALLOWED_HISTORY_PERIODS = frozenset({"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"})
ALLOWED_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d", "5d", "1wk", "1mo", "3mo"})
INTRADAY_INTERVALS = frozenset({"1m", "2m", "5m", "15m", "30m", "60m", "1h"})
INTRADAY_ALLOWED_HISTORY_PERIODS = frozenset({"1d", "5d", "1mo"})
SCAN_TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class ScanCancelledError(RuntimeError):
    """Raised when a background scan job is cancelled."""


def normalize_exchange(exchange: str | None) -> str | None:
    """Normalize an optional exchange string."""
    if exchange and exchange.strip():
        return exchange.strip().upper()
    return None


def normalize_history_period(history_period: str | None) -> str:
    """Normalize a yfinance history period token for signal scans."""
    value = str(history_period or "1y").strip().lower()
    if value not in ALLOWED_HISTORY_PERIODS:
        raise ValueError(
            "invalid history_period: "
            f"{history_period}; use one of {', '.join(sorted(ALLOWED_HISTORY_PERIODS))}"
        )
    return value


def normalize_interval(interval: str | None) -> str:
    """Normalize a yfinance interval token for signal scans."""
    value = str(interval or "1d").strip().lower()
    if value not in ALLOWED_INTERVALS:
        raise ValueError(
            f"invalid interval: {interval}; use one of {', '.join(sorted(ALLOWED_INTERVALS))}"
        )
    return value


def validate_history_request_pair(history_period: str, interval: str) -> None:
    """Validate supported yfinance period/interval combinations for signal scans."""
    if interval in INTRADAY_INTERVALS and history_period not in INTRADAY_ALLOWED_HISTORY_PERIODS:
        raise ValueError(
            "unsupported history_period/interval combination: "
            f"{history_period} with {interval}; intraday intervals require history_period of 1d, 5d, or 1mo"
        )


def resolve_signal_universe(signal_row: SignalRow, watchlist: list[str]) -> list[str]:
    """Tickers to scan for one persisted signal: overrides, watchlist, or exchange list."""
    scope = signal_row.ticker_scope
    if scope == "tickers":
        return list(signal_row.ticker_overrides or [])
    if scope == "exchange":
        if not signal_row.exchange:
            return []
        try:
            return fetch_exchange_tickers(signal_row.exchange)
        except ValueError:
            return []
    return list(watchlist)


def normalize_scan_time(scan_time: str | None) -> str:
    """Normalize a per-signal daily scan time (`HH:MM`, Eastern Time).

    Falls back to the `SCAN_TIME` env var (or `16:30`) when omitted, preserving
    the previous global-default behavior for signals that don't set their own time.
    """
    value = str(scan_time or os.environ.get("SCAN_TIME", "16:30")).strip()
    if not SCAN_TIME_PATTERN.match(value):
        raise ValueError(f"invalid scan_time: {scan_time}; use 24-hour HH:MM Eastern Time, e.g. 16:30")
    return value


def validate_signal_creation(
    name: str,
    signal_type: str,
    params: dict[str, Any] | None,
    ticker_scope: str,
    ticker_overrides: list[str] | None,
    exchange: str | None,
    history_period: str | None = None,
    interval: str | None = None,
    scan_time: str | None = None,
) -> dict[str, Any]:
    """Validate signal creation inputs and return normalized values or an error payload."""
    if signal_type not in CATALOG:
        return {"error": f"invalid signal_type: {signal_type}"}

    normalized_params = dict(params) if params else {}
    merge_params(signal_type, normalized_params)

    exchange_value = normalize_exchange(exchange)
    try:
        history_period_value = normalize_history_period(history_period)
        interval_value = normalize_interval(interval)
        validate_history_request_pair(history_period_value, interval_value)
        scan_time_value = normalize_scan_time(scan_time)
    except ValueError as exc:
        return {"error": str(exc)}
    overrides: list[str] | None = None

    if ticker_scope == "tickers":
        if not ticker_overrides:
            return {"error": "ticker_scope=tickers requires non-empty ticker_overrides list"}
        if exchange_value:
            return {"error": "exchange must be omitted when ticker_scope is tickers"}
        overrides = [str(value).upper() for value in ticker_overrides if str(value).strip()]
        if not overrides:
            return {"error": "ticker_overrides list is empty"}
    elif ticker_scope == "watchlist":
        if ticker_overrides:
            return {"error": "omit ticker_overrides when ticker_scope is watchlist"}
        if exchange_value:
            return {"error": "omit exchange when ticker_scope is watchlist"}
    else:
        if ticker_overrides:
            return {"error": "omit ticker_overrides when ticker_scope is exchange"}
        if not exchange_value:
            return {"error": "ticker_scope=exchange requires exchange (NYSE, NASDAQ, AMEX, or CRYPTO)"}
        if exchange_value not in EXCHANGES:
            return {"error": f"invalid exchange: {exchange_value}; use NYSE, NASDAQ, AMEX, or CRYPTO"}

    return {
        "name": name,
        "signal_type": signal_type,
        "params": normalized_params,
        "ticker_scope": ticker_scope,
        "ticker_overrides": overrides,
        "exchange": exchange_value,
        "history_period": history_period_value,
        "interval": interval_value,
        "scan_time": scan_time_value,
    }


def create_signal_payload(
    store: Store,
    user_id: str,
    name: str,
    signal_type: str,
    params: dict[str, Any] | None,
    ticker_scope: str,
    ticker_overrides: list[str] | None,
    exchange: str | None,
    history_period: str | None = None,
    interval: str | None = None,
    scan_time: str | None = None,
) -> str:
    """Create a persisted signal or return a JSON error payload."""
    validated = validate_signal_creation(
        name, signal_type, params, ticker_scope, ticker_overrides, exchange, history_period, interval, scan_time
    )
    if "error" in validated:
        return json.dumps(validated)
    try:
        signal_id = store.signal_create(
            user_id,
            validated["name"],
            validated["signal_type"],
            validated["params"],
            validated["ticker_overrides"],
            ticker_scope=validated["ticker_scope"],
            exchange=validated["exchange"],
            history_period=validated["history_period"],
            interval=validated["interval"],
            scan_time=validated["scan_time"],
        )
    except TypeError:
        # Backward compatibility for older test doubles or stores that still expose
        # the pre-timeframe signal_create signature.
        signal_id = store.signal_create(
            user_id,
            validated["name"],
            validated["signal_type"],
            validated["params"],
            validated["ticker_overrides"],
            validated["ticker_scope"],
            validated["exchange"],
        )
    return json.dumps(
        {
            "id": signal_id,
            "name": validated["name"],
            "signal_type": validated["signal_type"],
            "ticker_scope": validated["ticker_scope"],
            "exchange": validated["exchange"],
            "history_period": validated["history_period"],
            "interval": validated["interval"],
            "scan_time": validated["scan_time"],
        }
    )


def _resolve_scan_tickers(
    tickers: list[str] | None = None,
    symbol: str | None = None,
    exchange: str | None = None,
) -> tuple[list[str] | None, str | None]:
    tick_list: list[str] | None = None
    has_symbol = bool(symbol and symbol.strip())
    has_tickers = tickers is not None
    has_exchange = bool(exchange and exchange.strip())
    if sum(1 for value in (has_symbol, has_tickers, has_exchange) if value) > 1:
        raise ValueError("pass at most one of symbol, tickers, or exchange")
    if symbol:
        tick_list = [symbol.strip().upper()]
    if tickers is not None:
        tick_list = [str(value).upper() for value in tickers if str(value).strip()]
    exchange_value = normalize_exchange(exchange)
    if exchange_value:
        if exchange_value not in EXCHANGES:
            raise ValueError(f"invalid exchange: {exchange}; use NYSE, NASDAQ, AMEX, or CRYPTO")
        tick_list = fetch_exchange_tickers(exchange_value)
        if not tick_list:
            raise ValueError("no symbols returned for exchange")
    return tick_list, exchange_value


def _count_scan_work_items(
    store: Store,
    user_id: str,
    signal_rows: list[SignalRow],
    tick_list: list[str] | None,
) -> int:
    watch_symbols = [row.symbol for row in store.watchlist_get(user_id)]
    total = 0
    for signal_row in signal_rows:
        universe = tick_list if tick_list is not None else resolve_signal_universe(signal_row, watch_symbols)
        total += len(universe)
    return total

def execute_scan(
    store: Store,
    provider: DataProvider,
    user_id: str = DEFAULT_USER_ID,
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
    *,
    progress_callback: Callable[[int, int, int, int | None], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute a scan and return the parsed payload.

    Optional callbacks are used by background jobs:
    - `progress_callback(checked_count, fired_count, result_count, total_count)`
    - `cancel_check()` returns truthy to stop the scan
    """
    tick_list, exchange_value = _resolve_scan_tickers(tickers=tickers, symbol=symbol, exchange=exchange)

    def _raise_if_cancelled() -> None:
        if cancel_check and cancel_check():
            raise ScanCancelledError("scan cancelled")

    if all_signal_types:
        if signal_id is not None:
            raise ValueError("signal_id cannot be used with all_signal_types")
        if not tick_list:
            raise ValueError("all_signal_types requires symbol, tickers, or exchange")

        triggered_results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        checked = 0
        total_count = len(tick_list) * len(CATALOG)
        fetch_requests = [(ticker, {"period": "1y", "interval": "1d"}) for ticker in tick_list]
        for ticker, df, exc in fetch_histories_concurrently(provider, fetch_requests, cancel_check=cancel_check):
            _raise_if_cancelled()
            if exc is not None:
                errors.append({"symbol": ticker, "error": str(exc)})
                if progress_callback:
                    progress_callback(checked, len(triggered_results), len(triggered_results), total_count)
                continue
            if df is None or df.empty:
                errors.append({"symbol": ticker, "error": "no_history"})
                if progress_callback:
                    progress_callback(checked, len(triggered_results), len(triggered_results), total_count)
                continue

            for signal_type in CATALOG:
                _raise_if_cancelled()
                checked += 1
                params = merge_params(signal_type, {})
                signal = ActiveSignal(
                    id=0,
                    name=signal_type,
                    signal_type=signal_type,
                    params=params,
                    ticker_overrides=[ticker],
                    history_period="1y",
                    interval="1d",
                )
                triggered, details = evaluate(signal, df)
                if triggered:
                    triggered_results.append(
                        {
                            "signal_type": signal_type,
                            "name": signal_type,
                            "symbol": ticker,
                            "params": params,
                            "triggered": True,
                            "details": details,
                        }
                    )
                if progress_callback:
                    progress_callback(checked, len(triggered_results), len(triggered_results), total_count)
        return {
            "symbols": tick_list,
            "exchange": exchange_value,
            "mode": "all_signal_types",
            "history_period": "1y",
            "interval": "1d",
            "results": triggered_results,
            "count": len(triggered_results),
            "triggered_count": len(triggered_results),
            "checked_count": checked,
            "total_count": total_count,
            "errors": errors,
        }

    results: list[dict[str, Any]] = []
    ticker_errors: list[dict[str, Any]] = []
    watch_symbols = [row.symbol for row in store.watchlist_get(user_id)]
    signal_rows = store.signal_list(user_id)
    if signal_id is not None:
        signal_rows = [row for row in signal_rows if row.id == signal_id and row.enabled]
    else:
        signal_rows = [row for row in signal_rows if row.enabled]
    total_count = _count_scan_work_items(store, user_id, signal_rows, tick_list)
    checked_count = 0
    fired_count = 0
    for signal_row in signal_rows:
        universe = tick_list if tick_list is not None else resolve_signal_universe(signal_row, watch_symbols)
        if not universe:
            continue
        signal = ActiveSignal(
            id=signal_row.id,
            name=signal_row.name,
            signal_type=signal_row.signal_type,
            params=signal_row.params,
            ticker_overrides=signal_row.ticker_overrides,
            history_period=getattr(signal_row, "history_period", "1y"),
            interval=getattr(signal_row, "interval", "1d"),
        )
        fetch_requests = [
            (ticker, {"period": signal.history_period, "interval": signal.interval}) for ticker in universe
        ]
        for ticker, df, exc in fetch_histories_concurrently(provider, fetch_requests, cancel_check=cancel_check):
            _raise_if_cancelled()
            checked_count += 1
            if exc is not None:
                ticker_errors.append({"symbol": ticker, "signal_id": signal_row.id, "error": str(exc)})
                if progress_callback:
                    progress_callback(checked_count, fired_count, len(results), total_count)
                continue
            if df is None or df.empty:
                ticker_errors.append({"symbol": ticker, "signal_id": signal_row.id, "error": "no_history"})
                if progress_callback:
                    progress_callback(checked_count, fired_count, len(results), total_count)
                continue
            try:
                triggered, details = evaluate(signal, df)
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": ticker, "error": str(exc), "signal_id": signal_row.id})
                if progress_callback:
                    progress_callback(checked_count, fired_count, len(results), total_count)
                continue
            if triggered:
                fired_count += 1
                try:
                    store.alert_insert(user_id, signal_row.id, ticker, details, source="manual")
                except Exception as exc:  # noqa: BLE001
                    ticker_errors.append({"symbol": ticker, "signal_id": signal_row.id, "error": f"alert_insert: {exc}"})
                results.append(
                    {
                        "signal_id": signal_row.id,
                        "name": signal_row.name,
                        "symbol": ticker,
                        "triggered": True,
                        "details": details,
                    }
                )
            if progress_callback:
                progress_callback(checked_count, fired_count, len(results), total_count)
    return {
        "results": results,
        "count": len(results),
        "errors": ticker_errors,
        "checked_count": checked_count,
        "triggered_count": fired_count,
        "total_count": total_count,
        "exchange": exchange_value,
    }


def run_scan_payload(
    store: Store,
    provider: DataProvider,
    user_id: str = DEFAULT_USER_ID,
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> str:
    """Run an on-demand scan and return the JSON payload."""
    try:
        payload = execute_scan(
            store,
            provider,
            user_id=user_id,
            signal_id=signal_id,
            tickers=tickers,
            all_signal_types=all_signal_types,
            symbol=symbol,
            exchange=exchange,
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(payload, indent=2, default=str)
