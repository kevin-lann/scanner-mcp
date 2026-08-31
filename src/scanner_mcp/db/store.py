"""SQLite persistence for user-scoped watchlists, signals, alerts, and scan jobs."""

from __future__ import annotations

import json
import queue
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from scanner_mcp.db import libsql_adapter

DEFAULT_USER_ID = "local_admin"

DBConnection = sqlite3.Connection | libsql_adapter.ConnectionAdapter

# Number of long-lived Turso connections kept in the pool. Bounds concurrent
# outstanding requests to the remote database without serializing every
# operation behind a single process-wide mutex.
_TURSO_POOL_SIZE = 5


@dataclass
class WatchlistRow:
    id: int
    symbol: str
    added_at: str
    user_id: str = DEFAULT_USER_ID


@dataclass
class SignalRow:
    id: int
    name: str
    signal_type: str
    params: dict[str, Any]
    ticker_overrides: list[str] | None
    ticker_scope: str
    exchange: str | None
    enabled: bool
    created_at: str
    history_period: str = "1y"
    interval: str = "1d"
    scan_time: str = "16:30"
    user_id: str = DEFAULT_USER_ID


@dataclass
class AlertRow:
    id: int
    signal_id: int
    symbol: str
    triggered_at: str
    details: dict[str, Any]
    source: str = "scheduled"
    user_id: str = DEFAULT_USER_ID


@dataclass
class ScanJobRow:
    id: int
    job_type: str
    status: str
    params: dict[str, Any]
    requested_at: str
    started_at: str | None
    finished_at: str | None
    checked_count: int
    total_count: int | None
    fired_count: int
    result_count: int
    cancel_requested: bool
    result: dict[str, Any] | None
    error: str | None
    user_id: str = DEFAULT_USER_ID


def _default_db_path() -> Path:
    """Return the default SQLite DB path and ensure its parent exists."""
    p = Path.home() / ".scanner_mcp" / "data.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class Store:
    """Thread-safe store for user-scoped persistence, backed by local SQLite or remote Turso.

    Pass `turso_url`/`turso_auth_token` to use a remote `libsql://` Turso database
    (a small pool of long-lived connections is reused across calls, so concurrent
    operations aren't serialized behind a single network round trip). Otherwise
    this is a local SQLite file at `path` (or the default `~/.scanner_mcp/data.db`),
    opening and closing a connection per operation as before.
    """

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        turso_url: str | None = None,
        turso_auth_token: str | None = None,
    ) -> None:
        self._turso_url = turso_url.strip() if turso_url else None
        self._turso_auth_token = turso_auth_token or ""
        self._path = None if self._turso_url else (Path(path) if path else _default_db_path())
        self._lock = threading.RLock()
        self._libsql_pool: queue.Queue[libsql_adapter.ConnectionAdapter | None] | None = None
        if self._turso_url:
            self._libsql_pool = queue.Queue()
            for _ in range(_TURSO_POOL_SIZE):
                self._libsql_pool.put(None)
        self._init_schema()

    def _acquire_libsql_connection(self) -> libsql_adapter.ConnectionAdapter:
        """Take a connection from the pool, lazily opening one if this slot is unused."""
        assert self._libsql_pool is not None
        c = self._libsql_pool.get()
        if c is None:
            c = libsql_adapter.connect(self._turso_url, self._turso_auth_token)  # type: ignore[arg-type]
        return c

    @contextmanager
    def _conn(self) -> Iterator[DBConnection]:
        """Yield a database connection and commit after successful use.

        Local SQLite opens and closes a fresh connection per call, guarded by a
        process-wide lock. Turso connections are pooled and long-lived, so a
        failed operation must roll back explicitly rather than rely on closing
        the connection to discard it; pool checkout/checkin bounds concurrency
        without holding a lock across the network round trip.
        """
        if self._turso_url:
            c = self._acquire_libsql_connection()
            healthy = True
            try:
                c.execute("PRAGMA foreign_keys = ON")
                yield c
                c.commit()
            except Exception:
                healthy = False
                try:
                    c.rollback()
                except Exception:
                    # The connection itself is likely what's broken (e.g. a
                    # Turso/Hrana stream that expired server-side after being
                    # idle) -- rollback failing the same way just confirms it.
                    # Swallow this so the original exception below is what
                    # the caller sees.
                    pass
                raise
            finally:
                if healthy:
                    self._libsql_pool.put(c)  # type: ignore[union-attr]
                else:
                    # Never recycle a connection that failed mid-use: Turso's
                    # Hrana streams expire after enough idle time (e.g. the
                    # once-a-minute scan-tick job), and putting a dead stream
                    # back in the pool just means the next caller to draw it
                    # hits the exact same error forever, since nothing else
                    # here ever replaces it. `None` means the next checkout
                    # lazily opens a fresh connection instead.
                    try:
                        c.close()
                    except Exception:
                        pass
                    self._libsql_pool.put(None)  # type: ignore[union-attr]
        else:
            with self._lock:
                c = sqlite3.connect(self._path)
                c.row_factory = sqlite3.Row
                c.execute("PRAGMA foreign_keys = ON")
                try:
                    yield c
                    c.commit()
                except Exception:
                    c.rollback()
                    raise
                finally:
                    c.close()

    def _init_schema(self) -> None:
        """Create tables and indexes if this is a fresh database."""
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    UNIQUE (user_id, symbol)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    ticker_overrides TEXT,
                    ticker_scope TEXT NOT NULL DEFAULT 'watchlist',
                    exchange TEXT,
                    history_period TEXT NOT NULL DEFAULT '1y',
                    interval TEXT NOT NULL DEFAULT '1d',
                    scan_time TEXT NOT NULL DEFAULT '16:30',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE (id, user_id)
                );

                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    signal_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    details TEXT,
                    source TEXT NOT NULL DEFAULT 'scheduled',
                    FOREIGN KEY (signal_id, user_id) REFERENCES signals (id, user_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS scan_jobs (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    checked_count INTEGER NOT NULL DEFAULT 0,
                    total_count INTEGER,
                    fired_count INTEGER NOT NULL DEFAULT 0,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT
                );
                """
            )
            self._migrate_legacy_schema(c)
            self._ensure_column(c, "signals", "scan_time", "TEXT NOT NULL DEFAULT '16:30'")
            self._ensure_column(c, "alerts", "source", "TEXT NOT NULL DEFAULT 'scheduled'")
            c.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_watchlist_user_symbol ON watchlist (user_id, symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_user_created ON signals (user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_alerts_user_time ON alerts (user_id, triggered_at DESC);
                CREATE INDEX IF NOT EXISTS idx_scan_jobs_user_requested_time ON scan_jobs (user_id, requested_at DESC);
                """
            )

    def _migrate_legacy_schema(self, conn: DBConnection) -> None:
        """Upgrade older single-user tables to the current user-scoped schema."""
        self._ensure_user_scoped_table(
            conn,
            table_name="watchlist",
            create_sql="""
                CREATE TABLE watchlist (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    added_at TEXT NOT NULL,
                    UNIQUE (user_id, symbol)
                )
            """,
            copy_sql="""
                INSERT INTO watchlist (id, user_id, symbol, added_at)
                SELECT id, ?, symbol, added_at FROM watchlist_legacy
            """,
        )
        self._ensure_user_scoped_table(
            conn,
            table_name="signals",
            create_sql="""
                CREATE TABLE signals (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    ticker_overrides TEXT,
                    ticker_scope TEXT NOT NULL DEFAULT 'watchlist',
                    exchange TEXT,
                    history_period TEXT NOT NULL DEFAULT '1y',
                    interval TEXT NOT NULL DEFAULT '1d',
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE (id, user_id)
                )
            """,
            copy_sql="""
                INSERT INTO signals (
                    id, user_id, name, signal_type, params, ticker_overrides,
                    ticker_scope, exchange, history_period, interval, enabled, created_at
                )
                SELECT
                    id,
                    ?,
                    name,
                    signal_type,
                    params,
                    ticker_overrides,
                    COALESCE(ticker_scope, CASE WHEN ticker_overrides IS NULL OR ticker_overrides = '' THEN 'watchlist' ELSE 'tickers' END),
                    exchange,
                    COALESCE(history_period, '1y'),
                    COALESCE(interval, '1d'),
                    COALESCE(enabled, 1),
                    created_at
                FROM signals_legacy
            """,
        )
        self._ensure_user_scoped_table(
            conn,
            table_name="alerts",
            create_sql="""
                CREATE TABLE alerts (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    signal_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (signal_id, user_id) REFERENCES signals (id, user_id) ON DELETE CASCADE
                )
            """,
            copy_sql="""
                INSERT INTO alerts (id, user_id, signal_id, symbol, triggered_at, details)
                SELECT id, ?, signal_id, symbol, triggered_at, details FROM alerts_legacy
            """,
        )
        self._ensure_user_scoped_table(
            conn,
            table_name="scan_jobs",
            create_sql="""
                CREATE TABLE scan_jobs (
                    id INTEGER PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    checked_count INTEGER NOT NULL DEFAULT 0,
                    total_count INTEGER,
                    fired_count INTEGER NOT NULL DEFAULT 0,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    error TEXT
                )
            """,
            copy_sql="""
                INSERT INTO scan_jobs (
                    id, user_id, job_type, status, params, requested_at, started_at, finished_at,
                    checked_count, total_count, fired_count, result_count, cancel_requested, result_json, error
                )
                SELECT
                    id,
                    ?,
                    job_type,
                    status,
                    params,
                    requested_at,
                    started_at,
                    finished_at,
                    COALESCE(checked_count, 0),
                    total_count,
                    COALESCE(fired_count, 0),
                    COALESCE(result_count, 0),
                    COALESCE(cancel_requested, 0),
                    result_json,
                    error
                FROM scan_jobs_legacy
            """,
        )

    def _ensure_column(self, conn: DBConnection, table_name: str, column_name: str, column_type_and_default: str) -> None:
        """Add a column to an existing table if it isn't already present."""
        cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table_name})")}
        if column_name not in cols:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type_and_default}")

    def _ensure_user_scoped_table(
        self,
        conn: DBConnection,
        *,
        table_name: str,
        create_sql: str,
        copy_sql: str,
    ) -> None:
        cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table_name})")}
        if "user_id" in cols:
            return
        conn.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_legacy")
        conn.execute(create_sql)
        conn.execute(copy_sql, (DEFAULT_USER_ID,))
        conn.execute(f"DROP TABLE {table_name}_legacy")

    def all_user_ids(self) -> list[str]:
        """Return distinct user ids seen in any persisted user-owned table."""
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT DISTINCT user_id FROM (
                    SELECT user_id FROM watchlist
                    UNION ALL
                    SELECT user_id FROM signals
                    UNION ALL
                    SELECT user_id FROM alerts
                    UNION ALL
                    SELECT user_id FROM scan_jobs
                )
                WHERE user_id IS NOT NULL AND user_id <> ''
                ORDER BY user_id
                """
            ).fetchall()
        values = [str(row[0]) for row in rows]
        return values or [DEFAULT_USER_ID]

    # watchlist
    def watchlist_add(self, user_id: str, symbols: Sequence[str]) -> list[str]:
        """Insert uppercase watchlist symbols and return only newly added ones."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        added: list[str] = []
        with self._conn() as c:
            for sym in symbols:
                s = sym.strip().upper()
                if not s:
                    continue
                c.execute(
                    "INSERT OR IGNORE INTO watchlist (user_id, symbol, added_at) VALUES (?, ?, ?)",
                    (user_id, s, now),
                )
                n = c.execute("SELECT changes()").fetchone()[0]
                if n:
                    added.append(s)
        return added

    def watchlist_remove(self, user_id: str, symbols: Sequence[str]) -> int:
        """Remove watchlist symbols and return the number of deleted rows."""
        user_id = _normalize_user_id(user_id)
        n = 0
        with self._conn() as c:
            for sym in symbols:
                cur = c.execute(
                    "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?",
                    (user_id, sym.strip().upper()),
                )
                n += cur.rowcount or 0
        return n

    def watchlist_get(self, user_id: str) -> list[WatchlistRow]:
        """Return the current watchlist sorted by symbol."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, user_id, symbol, added_at FROM watchlist WHERE user_id = ? ORDER BY symbol",
                (user_id,),
            )
            return [WatchlistRow(int(r["id"]), str(r["symbol"]), str(r["added_at"]), str(r["user_id"])) for r in rows]

    # signals
    def signal_create(
        self,
        user_id: str,
        name: str,
        signal_type: str,
        params: dict[str, Any],
        ticker_overrides: list[str] | None,
        ticker_scope: str = "watchlist",
        exchange: str | None = None,
        history_period: str = "1y",
        interval: str = "1d",
        scan_time: str = "16:30",
    ) -> int:
        """Persist an enabled signal and return its database ID."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO signals (
                    user_id, name, signal_type, params, ticker_overrides,
                    ticker_scope, exchange, history_period, interval, scan_time, enabled, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    user_id,
                    name,
                    signal_type,
                    json.dumps(params),
                    json.dumps(ticker_overrides) if ticker_overrides is not None else None,
                    ticker_scope,
                    exchange.strip().upper() if exchange else None,
                    history_period.strip().lower(),
                    interval.strip().lower(),
                    scan_time.strip(),
                    now,
                ),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def signal_list(self, user_id: str) -> list[SignalRow]:
        """Return configured signals for one user ordered by ID."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            rows = c.execute("SELECT * FROM signals WHERE user_id = ? ORDER BY id", (user_id,))
            return [_row_to_signal(r) for r in rows]

    def signal_get(self, user_id: str, signal_id: int) -> SignalRow | None:
        """Return one user-owned signal by ID, or None if it does not exist."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM signals WHERE user_id = ? AND id = ?",
                (user_id, signal_id),
            ).fetchone()
            return _row_to_signal(r) if r else None

    def signal_delete(self, user_id: str, signal_id: int) -> bool:
        """Delete a user-owned signal and its alert history; return whether it existed."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            c.execute("DELETE FROM alerts WHERE user_id = ? AND signal_id = ?", (user_id, signal_id))
            cur = c.execute("DELETE FROM signals WHERE user_id = ? AND id = ?", (user_id, signal_id))
            return (cur.rowcount or 0) > 0

    def signal_set_enabled(self, user_id: str, signal_id: int, enabled: bool) -> bool:
        """Enable or disable one user-owned signal by ID."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                "UPDATE signals SET enabled = ? WHERE user_id = ? AND id = ?",
                (1 if enabled else 0, user_id, signal_id),
            )
            return (cur.rowcount or 0) > 0

    # alerts
    def alert_insert(
        self,
        user_id: str,
        signal_id: int,
        symbol: str,
        details: dict[str, Any],
        *,
        source: str = "scheduled",
    ) -> int:
        """Persist one triggered alert and return its database ID."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            if not c.execute(
                "SELECT id FROM signals WHERE id = ? AND user_id = ?",
                (signal_id, user_id),
            ).fetchone():
                raise ValueError(f"signal {signal_id} not found for user {user_id!r}")
            c.execute(
                "INSERT INTO alerts (user_id, signal_id, symbol, triggered_at, details, source) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, signal_id, symbol.upper(), now, json.dumps(details), source),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def alerts_recent(self, user_id: str, limit: int = 50) -> list[AlertRow]:
        """Return one user's newest alert rows, newest first."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM alerts WHERE user_id = ? ORDER BY triggered_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [_row_to_alert(r) for r in rows]

    # scan jobs
    def scan_job_create(self, user_id: str, job_type: str, params: dict[str, Any]) -> int:
        """Create a queued scan job and return its ID."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            c.execute(
                """
                INSERT INTO scan_jobs (user_id, job_type, status, params, requested_at)
                VALUES (?, ?, 'queued', ?, ?)
                """,
                (user_id, job_type, json.dumps(params), now),
            )
            return int(c.execute("SELECT last_insert_rowid()").fetchone()[0])

    def scan_job_get(self, user_id: str, job_id: int) -> ScanJobRow | None:
        """Return one user-owned scan job by ID, or None if missing."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM scan_jobs WHERE user_id = ? AND id = ?",
                (user_id, job_id),
            ).fetchone()
            return _row_to_scan_job(row) if row else None

    def scan_jobs_recent(self, user_id: str, limit: int = 20) -> list[ScanJobRow]:
        """Return one user's recent scan jobs, newest first."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM scan_jobs WHERE user_id = ? ORDER BY requested_at DESC LIMIT ?",
                (user_id, limit),
            )
            return [_row_to_scan_job(r) for r in rows]

    def scan_job_mark_running(self, user_id: str, job_id: int, *, total_count: int | None = None) -> bool:
        """Transition a queued job to running unless it was already cancelled."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'running', started_at = ?, total_count = COALESCE(?, total_count)
                WHERE user_id = ? AND id = ? AND status = 'queued' AND cancel_requested = 0
                """,
                (now, total_count, user_id, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_update_progress(
        self,
        user_id: str,
        job_id: int,
        *,
        checked_count: int,
        fired_count: int,
        result_count: int,
        total_count: int | None = None,
    ) -> bool:
        """Persist running job progress counters."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET checked_count = ?, fired_count = ?, result_count = ?, total_count = COALESCE(?, total_count)
                WHERE user_id = ? AND id = ? AND status = 'running'
                """,
                (checked_count, fired_count, result_count, total_count, user_id, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_complete(self, user_id: str, job_id: int, result: dict[str, Any]) -> bool:
        """Mark a job completed and persist its final result payload."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        checked_count = int(result.get("checked_count", 0))
        total_count = result.get("total_count")
        fired_count = int(result.get("triggered_count", result.get("fired_count", result.get("count", 0))))
        result_count = int(result.get("count", 0))
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'completed',
                    finished_at = ?,
                    checked_count = ?,
                    total_count = COALESCE(?, total_count),
                    fired_count = ?,
                    result_count = ?,
                    result_json = ?,
                    error = NULL
                WHERE user_id = ? AND id = ?
                """,
                (now, checked_count, total_count, fired_count, result_count, json.dumps(result), user_id, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_fail(self, user_id: str, job_id: int, error: str) -> bool:
        """Mark a job failed and store the error string."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'failed', finished_at = ?, error = ?
                WHERE user_id = ? AND id = ? AND status IN ('queued', 'running')
                """,
                (now, error, user_id, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_request_cancel(self, user_id: str, job_id: int) -> bool:
        """Request cancellation for a queued or running job."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET cancel_requested = 1
                WHERE user_id = ? AND id = ? AND status IN ('queued', 'running')
                """,
                (user_id, job_id),
            )
            return (cur.rowcount or 0) > 0

    def scan_job_is_cancel_requested(self, user_id: str, job_id: int) -> bool:
        """Return whether cancellation has been requested for a user-owned job."""
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            row = c.execute(
                "SELECT cancel_requested FROM scan_jobs WHERE user_id = ? AND id = ?",
                (user_id, job_id),
            ).fetchone()
            return bool(row[0]) if row else False

    def scan_job_mark_cancelled(
        self,
        user_id: str,
        job_id: int,
        *,
        checked_count: int,
        fired_count: int,
        result_count: int,
        total_count: int | None = None,
    ) -> bool:
        """Mark a user-owned job cancelled with the latest progress counters."""
        now = _utc_now()
        user_id = _normalize_user_id(user_id)
        with self._conn() as c:
            cur = c.execute(
                """
                UPDATE scan_jobs
                SET status = 'cancelled',
                    finished_at = ?,
                    checked_count = ?,
                    total_count = COALESCE(?, total_count),
                    fired_count = ?,
                    result_count = ?,
                    error = NULL
                WHERE user_id = ? AND id = ? AND status IN ('queued', 'running')
                """,
                (now, checked_count, total_count, fired_count, result_count, user_id, job_id),
            )
            return (cur.rowcount or 0) > 0


def _row_to_signal(r: sqlite3.Row | libsql_adapter.Row) -> SignalRow:
    """Convert a SQLite signal row into a typed dataclass."""
    ov = r["ticker_overrides"]
    ov_list = json.loads(ov) if ov else None
    return SignalRow(
        id=int(r["id"]),
        name=str(r["name"]),
        signal_type=str(r["signal_type"]),
        params=json.loads(r["params"]),
        ticker_overrides=ov_list,
        ticker_scope=str(r["ticker_scope"]),
        exchange=str(r["exchange"]).strip().upper() if r["exchange"] else None,
        enabled=bool(r["enabled"]),
        created_at=str(r["created_at"]),
        history_period=str(r["history_period"]).strip().lower() if r["history_period"] else "1y",
        interval=str(r["interval"]).strip().lower() if r["interval"] else "1d",
        scan_time=str(r["scan_time"]).strip() if r["scan_time"] else "16:30",
        user_id=str(r["user_id"]),
    )


def _row_to_alert(r: sqlite3.Row | libsql_adapter.Row) -> AlertRow:
    """Convert a SQLite alert row into a typed dataclass."""
    d = r["details"]
    return AlertRow(
        id=int(r["id"]),
        signal_id=int(r["signal_id"]),
        symbol=str(r["symbol"]),
        triggered_at=str(r["triggered_at"]),
        details=json.loads(d) if d else {},
        source=str(r["source"]) if r["source"] else "scheduled",
        user_id=str(r["user_id"]),
    )


def _row_to_scan_job(r: sqlite3.Row | libsql_adapter.Row) -> ScanJobRow:
    """Convert a SQLite scan job row into a typed dataclass."""
    payload = r["result_json"]
    return ScanJobRow(
        id=int(r["id"]),
        job_type=str(r["job_type"]),
        status=str(r["status"]),
        params=json.loads(r["params"]),
        requested_at=str(r["requested_at"]),
        started_at=str(r["started_at"]) if r["started_at"] else None,
        finished_at=str(r["finished_at"]) if r["finished_at"] else None,
        checked_count=int(r["checked_count"]),
        total_count=int(r["total_count"]) if r["total_count"] is not None else None,
        fired_count=int(r["fired_count"]),
        result_count=int(r["result_count"]),
        cancel_requested=bool(r["cancel_requested"]),
        result=json.loads(payload) if payload else None,
        error=str(r["error"]) if r["error"] else None,
        user_id=str(r["user_id"]),
    )


def _normalize_user_id(user_id: str | None) -> str:
    value = str(user_id or DEFAULT_USER_ID).strip()
    return value or DEFAULT_USER_ID


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()
