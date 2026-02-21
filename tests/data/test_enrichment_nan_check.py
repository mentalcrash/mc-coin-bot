"""Enrichment NaN 비율 경고 테스트.

MarketDataService._check_enrichment_nan_ratios()의 경고 로직을 검증합니다.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.data.service import MarketDataService


@pytest.fixture
def service() -> MarketDataService:
    """MarketDataService 인스턴스 (실제 데이터 로드 없이 메서드 테스트)."""
    return MarketDataService.__new__(MarketDataService)


def _make_df(n: int, enriched_cols: dict[str, float]) -> pd.DataFrame:
    """테스트용 DataFrame 생성.

    Args:
        n: 행 수
        enriched_cols: {컬럼명: NaN 비율} — 0.0=NaN 없음, 1.0=전부 NaN
    """
    index = pd.date_range("2025-01-01", periods=n, freq="1D", tz=UTC)
    data: dict[str, list[float]] = {
        "open": [100.0] * n,
        "high": [110.0] * n,
        "low": [90.0] * n,
        "close": [105.0] * n,
        "volume": [1000.0] * n,
    }
    for col, nan_ratio in enriched_cols.items():
        nan_count = int(n * nan_ratio)
        values = [1.0] * (n - nan_count) + [np.nan] * nan_count
        data[col] = values

    return pd.DataFrame(data, index=index)


OHLCV_COLS = {"open", "high", "low", "close", "volume"}


class TestEnrichmentNanCheck:
    """_check_enrichment_nan_ratios 경고 로직 검증."""

    def test_no_warning_when_nan_below_threshold(
        self, service: MarketDataService, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """NaN 비율이 30% 미만이면 경고 없음."""
        df = _make_df(100, {"oc_stablecoin_total": 0.1, "macro_dxy": 0.2})

        # loguru의 caplog 대신 sink를 사용
        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 0

    def test_warning_when_nan_above_30_pct(self, service: MarketDataService) -> None:
        """NaN 비율이 30% 초과 시 WARNING 경고."""
        df = _make_df(100, {"oc_tvl_total": 0.5})

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 1
        assert "oc_tvl_total" in warnings[0]
        assert "50%" in warnings[0]

    def test_drop_suggestion_when_nan_above_80_pct(
        self, service: MarketDataService
    ) -> None:
        """NaN 비율이 80% 초과 시 drop 제안 경고."""
        df = _make_df(100, {"macro_gold": 0.9})

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 1
        assert "consider dropping" in warnings[0]
        assert "macro_gold" in warnings[0]

    def test_multiple_columns_reported(self, service: MarketDataService) -> None:
        """여러 컬럼의 NaN 비율이 각각 올바른 카테고리로 보고된다."""
        df = _make_df(
            100,
            {
                "oc_fear_greed": 0.5,   # > 30%, warning
                "opt_btc_dvol": 0.85,   # > 80%, drop
                "dext_agg_oi": 0.1,     # < 30%, OK
            },
        )

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 2
        # 하나는 warning, 하나는 drop suggestion
        combined = " ".join(warnings)
        assert "oc_fear_greed" in combined
        assert "opt_btc_dvol" in combined
        assert "dext_agg_oi" not in combined

    def test_non_enrichment_columns_ignored(self, service: MarketDataService) -> None:
        """enrichment prefix가 아닌 컬럼은 검사 대상에서 제외된다."""
        df = _make_df(100, {})
        # 비 enrichment 컬럼 추가 (전부 NaN)
        df["custom_indicator"] = np.nan

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 0

    def test_empty_dataframe_no_error(self, service: MarketDataService) -> None:
        """빈 DataFrame에서 에러 없이 종료."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "oc_col"])

        # 에러 없이 실행되면 OK
        service._check_enrichment_nan_ratios(df, OHLCV_COLS)

    def test_no_enrichment_columns_no_warning(
        self, service: MarketDataService
    ) -> None:
        """enrichment 컬럼이 없으면 경고 없이 조기 반환."""
        df = _make_df(100, {})

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 0

    def test_funding_rate_and_oi_columns_checked(
        self, service: MarketDataService
    ) -> None:
        """funding_rate, open_interest 컬럼도 검사 대상이다."""
        df = _make_df(
            100,
            {
                "funding_rate": 0.5,
                "open_interest": 0.85,
            },
        )

        warnings: list[str] = []
        handler_id = logger.add(lambda msg: warnings.append(str(msg)), level="WARNING")
        try:
            service._check_enrichment_nan_ratios(df, OHLCV_COLS)
        finally:
            logger.remove(handler_id)

        assert len(warnings) == 2
        combined = " ".join(warnings)
        assert "funding_rate" in combined
        assert "open_interest" in combined
