"""Application runtime helpers for shared provider/store state and startup."""

from __future__ import annotations

import logging
import os
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from apscheduler.schedulers.base import BaseScheduler
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token, get_http_request

from scanner_mcp.data.provider import CompositeDataProvider, DataProvider
from scanner_mcp.db.store import DEFAULT_USER_ID, Store
from scanner_mcp.scanner import scheduler as scan_sched
from scanner_mcp.signals.service import ScanCancelledError, execute_scan

log = logging.getLogger(__name__)

_store: Store | None = None
_store_lock = threading.Lock()
_provider: DataProvider | None = None
_sched: BaseScheduler | None = None
_scan_executor: ThreadPoolExecutor | None = None
_scan_futures: dict[int, Future[None]] = {}
TRANSPORT_USER_ID_HEADER = "x-scanner-user-id"
STDIO_USER_ID_ENV = "SCANNER_MCP_USER_ID"
SCHEDULER_MODE_ENV = "SCANNER_MCP_SCHEDULER_MODE"
SCHEDULER_MODE_APSCHEDULER = "apscheduler"
SCHEDULER_MODE_EXTERNAL = "external"
SCAN_TRIGGER_SECRET_ENV = "SCAN_TRIGGER_SECRET"
SCAN_TRIGGER_SECRET_HEADER = "x-scan-trigger-secret"


def scan_trigger_authorized(provided: str | None, expected: str | None, mode: str) -> bool:
    """Authorize a call to `/internal/scan-tick`.

    In `external` mode `expected` (from `SCAN_TRIGGER_SECRET`) is mandatory: an
    unset secret always denies, so production can never end up with an
    unauthenticated tick endpoint by accident. In `apscheduler` (local/dev) mode,
    an unset secret allows any caller through, for frictionless local curl testing;
    if the developer does set it locally, it's still enforced.
    """
    expected = (expected or "").strip()
    if not expected:
        return mode != SCHEDULER_MODE_EXTERNAL
    return (provided or "").strip() == expected


def get_scheduler_mode() -> str:
    """Resolve the scheduler mode: in-process APScheduler (default) or external trigger.

    `external` is for the production Cloud Run deployment, where an outside Cloud
    Scheduler job ticks `/internal/scan-tick` once a minute instead -- Cloud Run's
    default billing throttles CPU between requests, which would otherwise silently
    stall an in-process per-minute cron. Local/stdio and docker-compose usage never
    set this, so they keep running the in-process scheduler unchanged.
    """
    mode = os.environ.get(SCHEDULER_MODE_ENV, SCHEDULER_MODE_APSCHEDULER).strip().lower()
    if mode not in (SCHEDULER_MODE_APSCHEDULER, SCHEDULER_MODE_EXTERNAL):
        raise RuntimeError(
            f"{SCHEDULER_MODE_ENV} must be {SCHEDULER_MODE_APSCHEDULER!r} or {SCHEDULER_MODE_EXTERNAL!r}, got {mode!r}"
        )
    return mode


def shutdown_scheduler() -> None:
    """Stop the background scheduler if it was started."""
    global _sched
    if _sched is None:
        return
    try:
        if _sched.running:
            _sched.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("Scheduler shutdown skipped: %s", exc)
    finally:
        _sched = None


def shutdown_scan_executor() -> None:
    """Stop the background scan executor."""
    global _scan_executor
    if _scan_executor is None:
        return
    try:
        _scan_executor.shutdown(wait=False, cancel_futures=False)
    except Exception as exc:  # noqa: BLE001
        log.debug("Scan executor shutdown skipped: %s", exc)
    finally:
        _scan_executor = None
        _scan_futures.clear()


def get_store() -> Store:
    """Lazily create the store.

    - `TURSO_DATABASE_URL=libsql://...` (+ `TURSO_AUTH_TOKEN`): remote Turso, shared
      with the web backend.
    - `TURSO_DATABASE_URL=file:...`: local SQLite at that path (e.g. for compose
      testing without a real Turso account).
    - Unset: local SQLite honoring `SCANNER_MCP_DB`, or `~/.scanner_mcp/data.db`.
      This is the standalone MCP default and never requires Turso.
    - Set to anything else: raises, since a typo'd or unrecognized URL silently
      falling back to local SQLite would defeat the shared web deployment.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                turso_url = os.environ.get("TURSO_DATABASE_URL", "").strip()
                if turso_url.startswith("libsql://"):
                    _store = Store(turso_url=turso_url, turso_auth_token=os.environ.get("TURSO_AUTH_TOKEN"))
                elif turso_url.startswith("file:"):
                    _store = Store(turso_url[len("file:") :])
                elif turso_url:
                    raise RuntimeError(
                        "TURSO_DATABASE_URL is set but not recognized (expected it to start "
                        f"with 'libsql://' or 'file:'): {turso_url!r}"
                    )
                else:
                    _store = Store(os.environ.get("SCANNER_MCP_DB"))
    return _store


def get_provider() -> DataProvider:
    """Lazily create the shared market data provider."""
    global _provider
    if _provider is None:
        _provider = CompositeDataProvider.default()
    return _provider


def get_request_user_id() -> str:
    """Resolve the authenticated tenant from FastMCP auth, transport headers, or stdio env."""
    access_token = get_access_token()
    if access_token is not None:
        for claim_name in ("sub", "user_id"):
            claim_value = access_token.claims.get(claim_name)
            if str(claim_value or "").strip():
                return str(claim_value).strip()

    try:
        request = get_http_request()
    except RuntimeError:
        request = None

    if request is not None:
        header_value = request.headers.get(TRANSPORT_USER_ID_HEADER, "").strip()
        if header_value:
            return header_value

    env_user_id = os.environ.get(STDIO_USER_ID_ENV, "").strip()
    if env_user_id:
        return env_user_id

    return DEFAULT_USER_ID


def scan_job_payload(job_id: int, user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    row = get_store().scan_job_get(user_id, job_id)
    if row is None:
        return {"error": "scan job not found", "job_id": job_id}
    return {
        "job_id": row.id,
        "job_type": row.job_type,
        "status": row.status,
        "requested_at": row.requested_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
        "checked_count": row.checked_count,
        "total_count": row.total_count,
        "fired_count": row.fired_count,
        "result_count": row.result_count,
        "cancel_requested": row.cancel_requested,
        "params": row.params,
        "error": row.error,
        "user_id": row.user_id,
    }


def _scan_executor_instance() -> ThreadPoolExecutor:
    global _scan_executor
    if _scan_executor is None:
        workers = max(1, int(os.environ.get("SCANNER_MCP_SCAN_WORKERS", "2")))
        _scan_executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="scanner-mcp-scan",
        )
    return _scan_executor


def start_scan_job(
    *,
    user_id: str = DEFAULT_USER_ID,
    signal_id: int | None = None,
    tickers: list[str] | None = None,
    all_signal_types: bool = False,
    symbol: str | None = None,
    exchange: str | None = None,
) -> int:
    """Create and dispatch a background scan job, returning its persistent job ID."""
    store = get_store()
    params = {
        "signal_id": signal_id,
        "tickers": tickers,
        "all_signal_types": all_signal_types,
        "symbol": symbol,
        "exchange": exchange,
    }
    job_id = store.scan_job_create(user_id, "run_scan", params)

    def _run() -> None:
        if not store.scan_job_mark_running(user_id, job_id):
            store.scan_job_mark_cancelled(
                user_id,
                job_id,
                checked_count=0,
                fired_count=0,
                result_count=0,
                total_count=0,
            )
            return
        try:
            result = execute_scan(
                store,
                get_provider(),
                user_id=user_id,
                signal_id=signal_id,
                tickers=tickers,
                all_signal_types=all_signal_types,
                symbol=symbol,
                exchange=exchange,
                progress_callback=lambda checked_count, fired_count, result_count, total_count: store.scan_job_update_progress(
                    user_id,
                    job_id,
                    checked_count=checked_count,
                    fired_count=fired_count,
                    result_count=result_count,
                    total_count=total_count,
                ),
                cancel_check=lambda: store.scan_job_is_cancel_requested(user_id, job_id),
            )
            store.scan_job_complete(user_id, job_id, result)
        except ScanCancelledError:
            latest = store.scan_job_get(user_id, job_id)
            checked_count = latest.checked_count if latest else 0
            fired_count = latest.fired_count if latest else 0
            result_count = latest.result_count if latest else 0
            total_count = latest.total_count if latest else None
            store.scan_job_mark_cancelled(
                user_id,
                job_id,
                checked_count=checked_count,
                fired_count=fired_count,
                result_count=result_count,
                total_count=total_count,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("background scan job %s failed", job_id)
            store.scan_job_fail(user_id, job_id, str(exc))
        finally:
            _scan_futures.pop(job_id, None)

    fut = _scan_executor_instance().submit(_run)
    _scan_futures[job_id] = fut
    return job_id


@asynccontextmanager
async def lifespan(_: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """FastMCP lifespan hook that starts and stops the scan scheduler."""
    global _sched
    st = get_store()
    provider = get_provider()
    if get_scheduler_mode() == SCHEDULER_MODE_APSCHEDULER:
        try:
            _sched = scan_sched.start_scheduler(st, provider)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not start scheduler: %s", exc)
    else:
        log.info("Scheduler mode is external: skipping in-process APScheduler, expecting /internal/scan-tick calls")
    try:
        yield {"store": st, "provider": provider, "scheduler": _sched}
    finally:
        shutdown_scheduler()
        shutdown_scan_executor()


def configure_logging() -> None:
    """Configure process logging from environment variables."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("SCANNER_MCP_LOG_FILE")
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def install_signal_handlers() -> tuple[Any, Any]:
    """Install SIGINT/SIGTERM handlers and return the previous handlers."""

    def _handle_stop(signum: int, _: Any) -> None:
        log.info("Received signal %s, shutting down", signum)
        shutdown_scheduler()
        shutdown_scan_executor()
        logging.shutdown()
        raise SystemExit(128 + signum)

    old_sigint = signal.signal(signal.SIGINT, _handle_stop)
    old_sigterm = signal.signal(signal.SIGTERM, _handle_stop)
    return old_sigint, old_sigterm


def restore_signal_handlers(old_sigint: Any, old_sigterm: Any) -> None:
    """Restore the previous process signal handlers."""
    signal.signal(signal.SIGINT, old_sigint)
    signal.signal(signal.SIGTERM, old_sigterm)
