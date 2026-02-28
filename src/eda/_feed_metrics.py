"""통합 데이터 피드 메트릭 헬퍼.

5개 피드(onchain/derivatives/macro/options/deriv_ext)가 공통으로 사용하는
mcbot_datafeed_* 메트릭 기록 함수.

monitoring 패키지 미설치 환경에서도 graceful degradation (ImportError 무시).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def record_feed_fetch(
    feed: str,
    source: str,
    name: str,
    elapsed: float,
    status: str,
    row_count: int,
) -> None:
    """통합 메트릭 기록.

    Args:
        feed: 피드 이름 (onchain|derivatives|macro|options|deriv_ext)
        source: 데이터 소스 (defillama, binance_futures, fred, deribit, ...)
        name: 데이터 이름 (stablecoin_total, funding_rate, ...)
        elapsed: 소요 시간 (초)
        status: "success" | "failure" | "empty"
        row_count: 반환된 행 수
    """
    try:
        from src.monitoring.metrics import (
            datafeed_fetch_latency_histogram,
            datafeed_fetch_rows_gauge,
            datafeed_fetch_total,
            datafeed_last_success_gauge,
        )

        datafeed_fetch_total.labels(feed=feed, source=source, status=status).inc()
        datafeed_fetch_latency_histogram.labels(feed=feed, source=source).observe(elapsed)
        if status == "success" and row_count > 0:
            datafeed_fetch_rows_gauge.labels(feed=feed, source=source, name=name).set(row_count)
            datafeed_last_success_gauge.labels(feed=feed, source=source).set(time.time())
    except ImportError:
        pass


def update_feed_cache_metrics(feed: str, cache: Mapping[str, dict[str, float] | float]) -> None:
    """캐시 크기 gauge 갱신.

    per-symbol dict[str, dict[str, float]] → symbol별 gauge
    global dict[str, float] → symbol="GLOBAL" gauge

    Args:
        feed: 피드 이름
        cache: 피드의 캐시 dict
    """
    try:
        from src.monitoring.metrics import datafeed_cache_size_gauge

        if not cache:
            return

        # per-symbol cache (onchain/derivatives/deriv_ext): {symbol: {col: val}}
        first_val = next(iter(cache.values()))
        if isinstance(first_val, dict):
            for symbol, cols in cache.items():
                assert isinstance(cols, dict)
                datafeed_cache_size_gauge.labels(feed=feed, symbol=symbol).set(len(cols))
        else:
            # global cache (macro/options): {col: val}
            datafeed_cache_size_gauge.labels(feed=feed, symbol="GLOBAL").set(len(cache))
    except ImportError:
        pass


def inc_feed_cache_refresh(feed: str, status: str) -> None:
    """캐시 리프레시 카운터 증가.

    Args:
        feed: 피드 이름
        status: "success" | "failure"
    """
    try:
        from src.monitoring.metrics import datafeed_cache_refresh_total

        datafeed_cache_refresh_total.labels(feed=feed, status=status).inc()
    except ImportError:
        pass
