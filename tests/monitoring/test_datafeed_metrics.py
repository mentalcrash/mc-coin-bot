"""통합 데이터 피드 메트릭 (mcbot_datafeed_*) 단위 테스트."""

from __future__ import annotations

import time

from prometheus_client import REGISTRY

from src.eda._feed_metrics import (
    inc_feed_cache_refresh,
    record_feed_fetch,
    update_feed_cache_metrics,
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


class TestRecordFeedFetch:
    def test_success_increments_counter_and_gauges(self) -> None:
        """성공 fetch → counter inc + rows gauge + last_success gauge."""
        before = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "derivatives", "source": "binance_futures", "status": "success"},
        )

        record_feed_fetch("derivatives", "binance_futures", "funding_rate", 1.5, "success", 50)

        after = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "derivatives", "source": "binance_futures", "status": "success"},
        )
        assert after is not None
        assert after > (before or 0)

        # rows gauge
        rows = _sample(
            "mcbot_datafeed_fetch_rows",
            {"feed": "derivatives", "source": "binance_futures", "name": "funding_rate"},
        )
        assert rows == 50

        # last success gauge (should be recent)
        last_ts = _sample(
            "mcbot_datafeed_last_success_timestamp",
            {"feed": "derivatives", "source": "binance_futures"},
        )
        assert last_ts is not None
        assert abs(last_ts - time.time()) < 5

    def test_failure_increments_counter_only(self) -> None:
        """실패 fetch → counter inc, rows/last_success 변경 없음."""
        before = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "macro", "source": "fred", "status": "failure"},
        )

        record_feed_fetch("macro", "fred", "dxy", 2.0, "failure", 0)

        after = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "macro", "source": "fred", "status": "failure"},
        )
        assert after is not None
        assert after > (before or 0)

    def test_empty_increments_counter_no_rows_gauge(self) -> None:
        """빈 결과 fetch → counter inc, rows gauge 미갱신."""
        before = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "options", "source": "deribit", "status": "empty"},
        )

        record_feed_fetch("options", "deribit", "btc_dvol", 0.8, "empty", 0)

        after = _sample(
            "mcbot_datafeed_fetch_total",
            {"feed": "options", "source": "deribit", "status": "empty"},
        )
        assert after is not None
        assert after > (before or 0)

    def test_latency_histogram_observed(self) -> None:
        """fetch latency가 histogram에 기록."""
        record_feed_fetch("deriv_ext", "coinalyze", "agg_oi", 3.5, "success", 10)

        count = _sample(
            "mcbot_datafeed_fetch_latency_seconds_count",
            {"feed": "deriv_ext", "source": "coinalyze"},
        )
        assert count is not None
        assert count > 0


class TestUpdateFeedCacheMetrics:
    def test_per_symbol_cache(self) -> None:
        """per-symbol cache → symbol별 gauge 갱신."""
        cache: dict[str, dict[str, float]] = {
            "BTC/USDT": {"funding_rate": 0.01, "mark_price": 50000},
            "ETH/USDT": {"funding_rate": 0.02},
        }
        update_feed_cache_metrics("derivatives", cache)

        btc = _sample("mcbot_datafeed_cache_size", {"feed": "derivatives", "symbol": "BTC/USDT"})
        eth = _sample("mcbot_datafeed_cache_size", {"feed": "derivatives", "symbol": "ETH/USDT"})
        assert btc == 2
        assert eth == 1

    def test_global_cache(self) -> None:
        """global cache → GLOBAL symbol gauge 갱신."""
        cache: dict[str, float] = {"macro_dxy": 104.5, "macro_vix": 15.0, "macro_spy_close": 500}
        update_feed_cache_metrics("macro", cache)

        val = _sample("mcbot_datafeed_cache_size", {"feed": "macro", "symbol": "GLOBAL"})
        assert val == 3

    def test_empty_cache_noop(self) -> None:
        """빈 캐시 → gauge 갱신 안 함."""
        # Should not raise
        update_feed_cache_metrics("options", {})


class TestIncFeedCacheRefresh:
    def test_success_increments(self) -> None:
        """success refresh → counter 증가."""
        before = _sample(
            "mcbot_datafeed_cache_refresh_total",
            {"feed": "onchain", "status": "success"},
        )

        inc_feed_cache_refresh("onchain", "success")

        after = _sample(
            "mcbot_datafeed_cache_refresh_total",
            {"feed": "onchain", "status": "success"},
        )
        assert after is not None
        assert after > (before or 0)

    def test_failure_increments(self) -> None:
        """failure refresh → counter 증가."""
        before = _sample(
            "mcbot_datafeed_cache_refresh_total",
            {"feed": "deriv_ext", "status": "failure"},
        )

        inc_feed_cache_refresh("deriv_ext", "failure")

        after = _sample(
            "mcbot_datafeed_cache_refresh_total",
            {"feed": "deriv_ext", "status": "failure"},
        )
        assert after is not None
        assert after > (before or 0)
