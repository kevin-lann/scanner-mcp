from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scanner_mcp import runtime
from scanner_mcp.db.store import SignalRow, Store
from scanner_mcp.scanner import scheduler
from scanner_mcp.signals import service as signals_service


class StoreIntegrationTest(unittest.TestCase):
    def test_watchlist_signal_alert_lifecycle_is_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            self.assertEqual(store.watchlist_add("user-a", [" aapl ", "AAPL", "", "msft"]), ["AAPL", "MSFT"])
            self.assertEqual(store.watchlist_add("user-b", ["msft", "tsla"]), ["MSFT", "TSLA"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["AAPL", "MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-b")], ["MSFT", "TSLA"])
            self.assertEqual(store.watchlist_remove("user-a", ["aapl", "missing"]), 1)
            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-b")], ["MSFT", "TSLA"])

            sid = store.signal_create(
                "user-a",
                "Dip",
                "pct_from_ath",
                {"min_pct_below_ath": 10},
                ["spy"],
                ticker_scope="tickers",
                exchange=None,
            )
            other_sid = store.signal_create("user-b", "Other", "rsi_oversold", {}, ["qqq"], ticker_scope="tickers", exchange=None)
            sig = store.signal_get("user-a", sid)
            self.assertIsNotNone(sig)
            assert sig is not None
            self.assertEqual(sig.ticker_overrides, ["spy"])
            self.assertEqual(sig.ticker_scope, "tickers")
            self.assertTrue(sig.enabled)
            self.assertIsNone(store.signal_get("user-a", other_sid))

            self.assertTrue(store.signal_set_enabled("user-a", sid, False))
            self.assertFalse(store.signal_get("user-a", sid).enabled)  # type: ignore[union-attr]
            aid = store.alert_insert("user-a", sid, "spy", {"x": 1})
            alerts = store.alerts_recent("user-a")
            self.assertEqual(alerts[0].id, aid)
            self.assertEqual(alerts[0].symbol, "SPY")
            self.assertEqual(alerts[0].details, {"x": 1})
            self.assertEqual(store.alerts_recent("user-b"), [])
            self.assertFalse(store.signal_delete("user-b", sid))
            self.assertTrue(store.signal_delete("user-a", sid))
            self.assertIsNone(store.signal_get("user-a", sid))
            self.assertEqual(store.alerts_recent("user-a"), [])

    def test_signal_scan_time_defaults_and_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            default_sid = store.signal_create("user-a", "Default Time", "rsi_oversold", {}, ["spy"], ticker_scope="tickers")
            custom_sid = store.signal_create(
                "user-a", "Custom Time", "rsi_oversold", {}, ["qqq"], ticker_scope="tickers", scan_time="09:45"
            )

            self.assertEqual(store.signal_get("user-a", default_sid).scan_time, "16:30")  # type: ignore[union-attr]
            self.assertEqual(store.signal_get("user-a", custom_sid).scan_time, "09:45")  # type: ignore[union-attr]

    def test_alert_insert_defaults_to_scheduled_and_accepts_explicit_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            sid = store.signal_create("user-a", "RSI", "rsi_oversold", {}, ["spy"], ticker_scope="tickers")

            scheduled_id = store.alert_insert("user-a", sid, "spy", {"x": 1})
            manual_id = store.alert_insert("user-a", sid, "spy", {"x": 2}, source="manual")

            alerts = {a.id: a for a in store.alerts_recent("user-a")}
            self.assertEqual(alerts[scheduled_id].source, "scheduled")
            self.assertEqual(alerts[manual_id].source, "manual")

    def test_scan_jobs_are_user_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            job_id = store.scan_job_create("user-a", "run_scan", {"symbol": "SPY"})

            self.assertIsNotNone(store.scan_job_get("user-a", job_id))
            self.assertIsNone(store.scan_job_get("user-b", job_id))
            self.assertFalse(store.scan_job_request_cancel("user-b", job_id))
            self.assertTrue(store.scan_job_request_cancel("user-a", job_id))


class SchedulerTest(unittest.TestCase):
    def test_tickers_for_signal_resolves_scope(self) -> None:
        tickers = SignalRow(1, "T", "rsi_oversold", {}, ["AAPL"], "tickers", None, True, "now")
        watch = SignalRow(2, "W", "rsi_oversold", {}, None, "watchlist", None, True, "now")
        exchange = SignalRow(3, "E", "rsi_oversold", {}, None, "exchange", "NASDAQ", True, "now")

        self.assertEqual(scheduler._tickers_for_signal(tickers, ["MSFT"]), ["AAPL"])
        self.assertEqual(scheduler._tickers_for_signal(watch, ["MSFT"]), ["MSFT"])
        with patch("scanner_mcp.scanner.scheduler.fetch_exchange_tickers", return_value=["QQQ"]):
            self.assertEqual(scheduler._tickers_for_signal(exchange, ["MSFT"]), ["QQQ"])

    def test_run_full_scan_persists_and_notifies_triggered_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            store.watchlist_add("user-a", ["SPY"])
            sid_a = store.signal_create("user-a", "RSI", "rsi_oversold", {}, None)
            store.watchlist_add("user-b", ["QQQ"])
            sid_b = store.signal_create("user-b", "RSI-B", "rsi_oversold", {}, None)

            class Provider:
                def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
                    self.symbol = symbol
                    return pd.DataFrame({"Close": [1.0, 2.0]})

            with (
                patch("scanner_mcp.scanner.scheduler.evaluate", return_value=(True, {"rsi": 20.0})) as eval_mock,
                patch("scanner_mcp.scanner.scheduler.notify_desktop") as notify_mock,
            ):
                result = scheduler.run_full_scan(store, Provider(), notify=True)  # type: ignore[arg-type]

            self.assertEqual(result["checked"], 2)
            self.assertEqual(result["fired"], 2)
            user_ids_in_alerts = {a["user_id"] for a in result["alerts"]}
            self.assertIn("user-a", user_ids_in_alerts)
            self.assertIn("user-b", user_ids_in_alerts)
            alert_a = next(a for a in result["alerts"] if a["user_id"] == "user-a")
            alert_b = next(a for a in result["alerts"] if a["user_id"] == "user-b")
            self.assertEqual(alert_a["signal_id"], sid_a)
            self.assertEqual(alert_b["signal_id"], sid_b)
            self.assertEqual(store.alerts_recent("user-a")[0].details, {"rsi": 20.0})
            self.assertEqual(store.alerts_recent("user-b")[0].details, {"rsi": 20.0})
            self.assertEqual(eval_mock.call_count, 2)
            self.assertEqual(notify_mock.call_count, 2)

    def test_scan_job_swallows_exceptions(self) -> None:
        with patch("scanner_mcp.scanner.scheduler.run_full_scan", side_effect=RuntimeError("boom")):
            scheduler._scan_job(object(), object())  # type: ignore[arg-type]

    def test_run_full_scan_at_time_only_scans_signals_due_now(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            store.watchlist_add("user-a", ["SPY"])
            due_sid = store.signal_create("user-a", "Due", "rsi_oversold", {}, None, scan_time="09:45")
            not_due_sid = store.signal_create("user-a", "Not due", "rsi_oversold", {}, None, scan_time="16:30")

            class Provider:
                def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
                    return pd.DataFrame({"Close": [1.0, 2.0]})

            with (
                patch("scanner_mcp.scanner.scheduler.evaluate", return_value=(True, {"rsi": 20.0})),
                patch("scanner_mcp.scanner.scheduler.notify_desktop"),
            ):
                result = scheduler.run_full_scan(store, Provider(), notify=False, at_time="09:45")  # type: ignore[arg-type]

            self.assertEqual(result["checked"], 1)
            fired_signal_ids = {a["signal_id"] for a in result["alerts"]}
            self.assertEqual(fired_signal_ids, {due_sid})
            self.assertNotIn(not_due_sid, fired_signal_ids)

    def test_run_full_scan_fetches_tickers_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            tickers = [f"T{i}" for i in range(6)]
            store.watchlist_add("user-a", tickers)
            store.signal_create("user-a", "RSI", "rsi_oversold", {}, None)

            lock = threading.Lock()
            state = {"concurrent": 0, "peak": 0}

            class SlowProvider:
                def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
                    with lock:
                        state["concurrent"] += 1
                        state["peak"] = max(state["peak"], state["concurrent"])
                    time.sleep(0.05)
                    with lock:
                        state["concurrent"] -= 1
                    return pd.DataFrame({"Close": [1.0, 2.0]})

            with (
                patch.dict(os.environ, {"SCANNER_MCP_FETCH_WORKERS": "6"}),
                patch("scanner_mcp.scanner.scheduler.evaluate", return_value=(False, {})),
            ):
                result = scheduler.run_full_scan(store, SlowProvider(), notify=False)  # type: ignore[arg-type]

            self.assertEqual(result["checked"], len(tickers))
            self.assertGreater(state["peak"], 1)


class SchedulerModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_starts_apscheduler_by_default(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(runtime, "get_store", return_value="store"),
            patch.object(runtime, "get_provider", return_value="provider"),
            patch("scanner_mcp.scanner.scheduler.start_scheduler") as start_mock,
        ):
            os.environ.pop("SCANNER_MCP_SCHEDULER_MODE", None)
            async with runtime.lifespan(object()):  # type: ignore[arg-type]
                pass
            start_mock.assert_called_once_with("store", "provider")

    async def test_lifespan_skips_apscheduler_in_external_mode(self) -> None:
        with (
            patch.dict(os.environ, {"SCANNER_MCP_SCHEDULER_MODE": "external"}),
            patch.object(runtime, "get_store", return_value="store"),
            patch.object(runtime, "get_provider", return_value="provider"),
            patch("scanner_mcp.scanner.scheduler.start_scheduler") as start_mock,
        ):
            async with runtime.lifespan(object()):  # type: ignore[arg-type]
                pass
            start_mock.assert_not_called()

    def test_get_scheduler_mode_rejects_unrecognized_value(self) -> None:
        with patch.dict(os.environ, {"SCANNER_MCP_SCHEDULER_MODE": "bogus"}):
            with self.assertRaises(RuntimeError):
                runtime.get_scheduler_mode()

    def test_scan_trigger_authorized_requires_secret_in_external_mode(self) -> None:
        self.assertFalse(runtime.scan_trigger_authorized(None, None, "external"))
        self.assertFalse(runtime.scan_trigger_authorized("anything", "", "external"))

    def test_scan_trigger_authorized_skips_check_when_secret_unset_locally(self) -> None:
        self.assertTrue(runtime.scan_trigger_authorized(None, None, "apscheduler"))

    def test_scan_trigger_authorized_enforces_match_when_secret_set(self) -> None:
        self.assertTrue(runtime.scan_trigger_authorized("s3cr3t", "s3cr3t", "apscheduler"))
        self.assertFalse(runtime.scan_trigger_authorized("wrong", "s3cr3t", "apscheduler"))
        self.assertTrue(runtime.scan_trigger_authorized("s3cr3t", "s3cr3t", "external"))
        self.assertFalse(runtime.scan_trigger_authorized("wrong", "s3cr3t", "external"))


class ExecuteScanAlertPersistenceTest(unittest.TestCase):
    """`execute_scan` backs both `run_scan` and `start_scan` — both should persist
    triggered alerts for saved-signal scans, but never for ad-hoc `all_signal_types`
    scans, since those have no persisted `signal_id` to attach an alert to."""

    class Provider:
        def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
            return pd.DataFrame({"Close": [1.0, 2.0]})

    def test_saved_signal_scan_persists_alert_with_manual_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")
            sid = store.signal_create("user-a", "RSI", "rsi_oversold", {}, ["SPY"], ticker_scope="tickers")

            with patch("scanner_mcp.signals.service.evaluate", return_value=(True, {"rsi": 20.0})):
                payload = signals_service.execute_scan(store, self.Provider(), user_id="user-a", signal_id=sid)

            self.assertEqual(payload["count"], 1)
            alerts = store.alerts_recent("user-a")
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0].source, "manual")
            self.assertEqual(alerts[0].signal_id, sid)

    def test_all_signal_types_scan_does_not_persist_any_alert(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td) / "test.db")

            with patch("scanner_mcp.signals.service.evaluate", return_value=(True, {"rsi": 20.0})):
                payload = signals_service.execute_scan(
                    store, self.Provider(), user_id="user-a", all_signal_types=True, symbol="SPY"
                )

            self.assertGreater(payload["count"], 0)
            self.assertEqual(store.alerts_recent("user-a"), [])


class TursoBackendTest(unittest.TestCase):
    """Exercises `db/libsql_adapter.py` via `Store(turso_url=...)`, using a local
    embedded libsql database in place of a real Turso account -- the adapter code
    path is otherwise only reachable with a live `libsql://` URL."""

    def _require_libsql(self) -> None:
        try:
            import libsql  # noqa: F401
        except ImportError:
            self.skipTest("libsql package not installed (pip install 'horus-mcp[turso]')")

    def test_watchlist_and_signal_lifecycle_over_libsql_adapter(self) -> None:
        self._require_libsql()
        with tempfile.TemporaryDirectory() as td:
            store = Store(turso_url=str(Path(td) / "turso.db"), turso_auth_token="")

            self.assertEqual(store.watchlist_add("user-a", ["aapl", "AAPL", "msft"]), ["AAPL", "MSFT"])
            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["AAPL", "MSFT"])

            sid = store.signal_create(
                "user-a", "Dip", "pct_from_ath", {"min_pct_below_ath": 10}, ["spy"], ticker_scope="tickers"
            )
            sig = store.signal_get("user-a", sid)
            self.assertIsNotNone(sig)
            assert sig is not None
            self.assertEqual(sig.ticker_overrides, ["spy"])

            aid = store.alert_insert("user-a", sid, "spy", {"x": 1})
            alerts = store.alerts_recent("user-a")
            self.assertEqual(alerts[0].id, aid)
            self.assertEqual(alerts[0].details, {"x": 1})

    def test_failed_operation_rolls_back_the_shared_connection(self) -> None:
        self._require_libsql()
        with tempfile.TemporaryDirectory() as td:
            store = Store(turso_url=str(Path(td) / "turso.db"), turso_auth_token="")
            store.watchlist_add("user-a", ["AAPL"])

            with self.assertRaises(RuntimeError):
                with store._conn() as c:
                    c.execute(
                        "INSERT INTO watchlist (user_id, symbol, added_at) VALUES (?, ?, ?)",
                        ("user-a", "ZZZZ", "now"),
                    )
                    raise RuntimeError("boom")

            self.assertEqual([w.symbol for w in store.watchlist_get("user-a")], ["AAPL"])
            self.assertEqual(store.watchlist_add("user-a", ["TSLA"]), ["TSLA"])


if __name__ == "__main__":
    unittest.main()
