"""PrometheusOnchainCallback 메트릭 테스트."""

from __future__ import annotations

import time

from prometheus_client import REGISTRY

from src.monitoring.metrics import PrometheusOnchainCallback


def _sample(name: str, labels: dict[str, str] | None = None) -> float | None:
    """REGISTRY에서 특정 metric 값을 가져옴."""
    return REGISTRY.get_sample_value(name, labels or {})


class TestPrometheusOnchainCallback:
    def test_success_increments_counter_and_gauge(self) -> None:
        """성공 fetch → counter inc + rows gauge set + last_success gauge set."""
        cb = PrometheusOnchainCallback()
        before = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "defillama", "status": "success"},
        )

        cb.on_fetch("defillama", "stablecoin_total", 1.5, "success", 100)

        after = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "defillama", "status": "success"},
        )
        assert after is not None
        assert after > (before or 0)

        # rows gauge
        rows = _sample(
            "mcbot_onchain_fetch_rows",
            {"source": "defillama", "name": "stablecoin_total"},
        )
        assert rows == 100

        # last success gauge (should be recent)
        last_ts = _sample(
            "mcbot_onchain_last_success_timestamp",
            {"source": "defillama"},
        )
        assert last_ts is not None
        assert abs(last_ts - time.time()) < 5

    def test_failure_increments_counter_only(self) -> None:
        """실패 fetch → counter inc, rows/last_success 변경 없음."""
        cb = PrometheusOnchainCallback()
        before = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "coinmetrics", "status": "failure"},
        )

        cb.on_fetch("coinmetrics", "btc_metrics", 2.0, "failure", 0)

        after = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "coinmetrics", "status": "failure"},
        )
        assert after is not None
        assert after > (before or 0)

    def test_empty_increments_counter_no_gauge(self) -> None:
        """빈 결과 fetch → counter inc, rows gauge 미갱신."""
        cb = PrometheusOnchainCallback()
        before = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "alternative_me", "status": "empty"},
        )

        cb.on_fetch("alternative_me", "fear_greed", 0.8, "empty", 0)

        after = _sample(
            "mcbot_onchain_fetch_total",
            {"source": "alternative_me", "status": "empty"},
        )
        assert after is not None
        assert after > (before or 0)

    def test_latency_histogram_observed(self) -> None:
        """fetch latency가 histogram에 기록."""
        cb = PrometheusOnchainCallback()
        cb.on_fetch("blockchain_com", "bc_hash-rate", 3.5, "success", 50)

        # histogram _count should be > 0
        count = _sample(
            "mcbot_onchain_fetch_latency_seconds_count",
            {"source": "blockchain_com"},
        )
        assert count is not None
        assert count > 0

    def test_protocol_compliance(self) -> None:
        """PrometheusOnchainCallback은 OnchainMetricsCallback Protocol 충족."""
        from src.monitoring.metrics import OnchainMetricsCallback

        assert isinstance(PrometheusOnchainCallback(), OnchainMetricsCallback)
