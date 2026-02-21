"""Tests for MarketDataService on-chain enrichment."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.market_data import MarketDataRequest
from src.data.service import MarketDataService


def _make_ohlcv(n: int = 10) -> pd.DataFrame:
    """테스트용 OHLCV DataFrame 생성."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(100, 100 + n),
            "high": range(105, 105 + n),
            "low": range(95, 95 + n),
            "close": range(100, 100 + n),
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def _make_onchain_enriched(index: pd.DatetimeIndex) -> pd.DataFrame:
    """On-chain enrichment 결과 DataFrame."""
    return pd.DataFrame(
        {
            "oc_dex_volume_usd": [1e6] * len(index),
            "oc_stablecoin_total": [1e9] * len(index),
        },
        index=index,
    )


class TestOnchainEnrichment:
    """MarketDataService on-chain enrichment 검증."""

    @patch("src.data.onchain.service.OnchainDataService")
    def test_maybe_enrich_onchain_no_data(self, mock_cls: MagicMock) -> None:
        """On-chain 데이터 없으면 원본 반환."""
        mock_instance = mock_cls.return_value
        mock_instance.precompute.return_value = pd.DataFrame()

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = MagicMock(spec=MarketDataRequest)
        request.symbol = "BTC/USDT"

        result = service._maybe_enrich_onchain(df, request)

        assert result is df
        mock_instance.precompute.assert_called_once()

    @patch("src.data.onchain.service.OnchainDataService")
    def test_maybe_enrich_onchain_with_data(self, mock_cls: MagicMock) -> None:
        """On-chain 데이터 있으면 oc_* 컬럼 주입."""
        df = _make_ohlcv()
        enriched = _make_onchain_enriched(df.index)
        mock_instance = mock_cls.return_value
        mock_instance.precompute.return_value = enriched

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        request = MagicMock(spec=MarketDataRequest)
        request.symbol = "BTC/USDT"

        result = service._maybe_enrich_onchain(df, request)

        assert "oc_dex_volume_usd" in result.columns
        assert "oc_stablecoin_total" in result.columns
        assert len(result) == len(df)
        # 원래 OHLCV 컬럼도 보존
        assert "close" in result.columns

    def test_maybe_enrich_onchain_graceful_degradation(self) -> None:
        """On-chain 서비스 예외 시 원본 반환."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = MagicMock(spec=MarketDataRequest)
        request.symbol = "BTC/USDT"

        # OnchainDataService import가 실패해도 graceful degradation
        with patch(
            "src.data.onchain.service.OnchainDataService",
            side_effect=Exception("service unavailable"),
        ):
            result = service._maybe_enrich_onchain(df, request)

        assert result is df
