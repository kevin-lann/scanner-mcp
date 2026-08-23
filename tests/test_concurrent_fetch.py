from __future__ import annotations

import threading
import time
import unittest

from scanner_mcp.data.concurrent_fetch import fetch_histories_concurrently


class FakeProvider:
    def __init__(self, *, delay: float = 0.0, fail_for: frozenset[str] = frozenset()) -> None:
        self._delay = delay
        self._fail_for = fail_for
        self._lock = threading.Lock()
        self._concurrent = 0
        self.peak_concurrent = 0

    def get_history(self, symbol: str, **_kwargs: object) -> str:
        with self._lock:
            self._concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self._concurrent)
        try:
            if self._delay:
                time.sleep(self._delay)
            if symbol in self._fail_for:
                raise RuntimeError(f"boom {symbol}")
            return f"history:{symbol}"
        finally:
            with self._lock:
                self._concurrent -= 1


class FetchHistoriesConcurrentlyTest(unittest.TestCase):
    def test_all_keys_returned_exactly_once_with_success_and_failure(self) -> None:
        provider = FakeProvider(fail_for=frozenset({"BAD"}))
        requests = [(sym, {}) for sym in ["AAPL", "MSFT", "BAD", "TSLA"]]

        results = list(fetch_histories_concurrently(provider, requests, max_workers=4))

        self.assertEqual({key for key, _, _ in results}, {"AAPL", "MSFT", "BAD", "TSLA"})
        by_key = {key: (df, exc) for key, df, exc in results}
        self.assertEqual(by_key["AAPL"], ("history:AAPL", None))
        self.assertIsNone(by_key["BAD"][0])
        self.assertIsInstance(by_key["BAD"][1], RuntimeError)

    def test_respects_max_workers(self) -> None:
        provider = FakeProvider(delay=0.05)
        requests = [(str(i), {}) for i in range(10)]

        list(fetch_histories_concurrently(provider, requests, max_workers=3))

        self.assertLessEqual(provider.peak_concurrent, 3)
        self.assertGreater(provider.peak_concurrent, 1)

    def test_cancel_check_true_upfront_yields_nothing(self) -> None:
        provider = FakeProvider()
        requests = [(str(i), {}) for i in range(5)]

        results = list(fetch_histories_concurrently(provider, requests, cancel_check=lambda: True))

        self.assertEqual(results, [])

    def test_empty_requests_yields_nothing(self) -> None:
        provider = FakeProvider()

        results = list(fetch_histories_concurrently(provider, []))

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
