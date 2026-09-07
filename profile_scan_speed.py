#!/usr/bin/env python3
"""Profile local MCP scan speed against a full exchange universe (NASDAQ/NYSE)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SCANNER_MCP_DB", str(ROOT / ".profile_scan.db"))
os.environ.setdefault("LOG_LEVEL", "WARNING")

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scanner_mcp.data.exchange_universe import fetch_exchange_tickers  # noqa: E402
from scanner_mcp.db.store import Store  # noqa: E402
from scanner_mcp.runtime import get_provider  # noqa: E402
from scanner_mcp.signals.catalog import CATALOG  # noqa: E402
from scanner_mcp.signals.service import execute_scan  # noqa: E402


def main() -> int:
    exchange = (sys.argv[1] if len(sys.argv) > 1 else "NASDAQ").upper()
    fetch_workers = os.environ.get("SCANNER_MCP_FETCH_WORKERS", "8")
    print(f"exchange={exchange}")
    print(f"SCANNER_MCP_FETCH_WORKERS={fetch_workers}")
    print(f"catalog_signal_types={len(CATALOG)}")
    print(f"db={os.environ['SCANNER_MCP_DB']}")
    print()

    t0 = time.perf_counter()
    symbols = fetch_exchange_tickers(exchange, use_cache=False)
    universe_s = time.perf_counter() - t0
    print(f"universe_fetch_seconds={universe_s:.2f}")
    print(f"universe_size={len(symbols)}")
    if not symbols:
        print("ERROR: empty universe")
        return 1
    print(f"universe_sample={symbols[:8]}")
    print()

    store = Store(os.environ["SCANNER_MCP_DB"])
    provider = get_provider()
    last_print = time.perf_counter()
    scan_start = time.perf_counter()

    def on_progress(checked: int, fired: int, result_count: int, total: int | None) -> None:
        nonlocal last_print
        now = time.perf_counter()
        if now - last_print < 10 and total and checked < total:
            return
        last_print = now
        elapsed = now - scan_start
        rate = checked / elapsed if elapsed > 0 else 0.0
        denom = total or "?"
        eta = ""
        if total and rate > 0:
            remaining = max(0, total - checked) / rate
            eta = f" eta_s={remaining:.0f}"
        print(
            f"progress checked={checked}/{denom} triggered={fired} "
            f"elapsed_s={elapsed:.1f} checks_per_s={rate:.2f}{eta}",
            flush=True,
        )

    print("starting execute_scan(all_signal_types=True) — same path as MCP run_scan/start_scan")
    result = execute_scan(
        store,
        provider,
        all_signal_types=True,
        exchange=exchange,
        progress_callback=on_progress,
    )
    scan_s = time.perf_counter() - scan_start
    n_sym = len(result.get("symbols") or symbols)
    checked = int(result.get("checked_count") or 0)
    triggered = int(result.get("triggered_count") or 0)
    errors = result.get("errors") or []
    total = int(result.get("total_count") or 0)

    summary = {
        "exchange": exchange,
        "fetch_workers": int(fetch_workers),
        "universe_size": n_sym,
        "universe_fetch_seconds": round(universe_s, 2),
        "scan_seconds": round(scan_s, 2),
        "scan_minutes": round(scan_s / 60, 2),
        "checked_count": checked,
        "total_count": total,
        "triggered_count": triggered,
        "error_count": len(errors),
        "symbols_per_second": round(n_sym / scan_s, 3) if scan_s else None,
        "checks_per_second": round(checked / scan_s, 3) if scan_s else None,
        "seconds_per_symbol": round(scan_s / n_sym, 3) if n_sym else None,
        "mode": result.get("mode"),
        "history_period": result.get("history_period"),
        "interval": result.get("interval"),
    }
    print()
    print("=== SCAN SPEED PROFILE ===")
    print(json.dumps(summary, indent=2))
    if errors:
        print()
        print(f"first_errors={json.dumps(errors[:8], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
