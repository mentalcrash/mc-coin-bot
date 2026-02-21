"""Tests for MarketDataService always-on enrichment (5 data types)."""

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


def _make_request() -> MagicMock:
    """테스트용 MarketDataRequest mock."""
    request = MagicMock(spec=MarketDataRequest)
    request.symbol = "BTC/USDT"
    request.start = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    request.end = pd.Timestamp("2024-01-10", tz="UTC").to_pydatetime()
    return request


class TestDerivativesEnrichment:
    """Derivatives enrichment 검증."""

    @patch("src.data.derivatives_service.DerivativesDataService")
    def test_enrichment_applied(self, mock_cls: MagicMock) -> None:
        """Derivatives enrichment가 적용되면 컬럼 추가."""
        df = _make_ohlcv()
        enriched = df.copy()
        enriched["funding_rate"] = 0.001
        enriched["open_interest"] = 1e6
        mock_cls.return_value.enrich.return_value = enriched

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        request = _make_request()

        result = service._maybe_enrich_derivatives(df, request)

        assert "funding_rate" in result.columns
        assert "open_interest" in result.columns
        mock_cls.return_value.enrich.assert_called_once()

    def test_graceful_degradation(self) -> None:
        """Derivatives 데이터 없으면 원본 반환."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        with patch(
            "src.data.derivatives_service.DerivativesDataService",
            side_effect=Exception("no data"),
        ):
            result = service._maybe_enrich_derivatives(df, request)

        assert result is df


class TestMacroEnrichment:
    """Macro enrichment 검증."""

    @patch("src.data.macro.service.MacroDataService")
    def test_enrichment_applied(self, mock_cls: MagicMock) -> None:
        """Macro enrichment가 적용되면 macro_* 컬럼 추가."""
        df = _make_ohlcv()
        enriched_cols = pd.DataFrame(
            {
                "macro_dxy": [100.0] * len(df),
                "macro_vix": [20.0] * len(df),
            },
            index=df.index,
        )
        mock_cls.return_value.precompute.return_value = enriched_cols

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        request = _make_request()

        result = service._maybe_enrich_macro(df, request)

        assert "macro_dxy" in result.columns
        assert "macro_vix" in result.columns
        assert len(result) == len(df)
        mock_cls.return_value.precompute.assert_called_once_with(ohlcv_index=df.index)

    @patch("src.data.macro.service.MacroDataService")
    def test_no_data_returns_original(self, mock_cls: MagicMock) -> None:
        """Macro 데이터 없으면 원본 반환."""
        mock_cls.return_value.precompute.return_value = pd.DataFrame()

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        result = service._maybe_enrich_macro(df, request)

        assert result is df

    def test_graceful_degradation(self) -> None:
        """Macro 서비스 예외 시 원본 반환."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        with patch(
            "src.data.macro.service.MacroDataService",
            side_effect=Exception("service unavailable"),
        ):
            result = service._maybe_enrich_macro(df, request)

        assert result is df


class TestOptionsEnrichment:
    """Options enrichment 검증."""

    @patch("src.data.options.service.OptionsDataService")
    def test_enrichment_applied(self, mock_cls: MagicMock) -> None:
        """Options enrichment가 적용되면 opt_* 컬럼 추가."""
        df = _make_ohlcv()
        enriched_cols = pd.DataFrame(
            {
                "opt_btc_dvol": [50.0] * len(df),
                "opt_eth_dvol": [60.0] * len(df),
            },
            index=df.index,
        )
        mock_cls.return_value.precompute.return_value = enriched_cols

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        request = _make_request()

        result = service._maybe_enrich_options(df, request)

        assert "opt_btc_dvol" in result.columns
        assert "opt_eth_dvol" in result.columns
        mock_cls.return_value.precompute.assert_called_once_with(ohlcv_index=df.index)

    @patch("src.data.options.service.OptionsDataService")
    def test_no_data_returns_original(self, mock_cls: MagicMock) -> None:
        """Options 데이터 없으면 원본 반환."""
        mock_cls.return_value.precompute.return_value = pd.DataFrame()

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        result = service._maybe_enrich_options(df, request)

        assert result is df

    def test_graceful_degradation(self) -> None:
        """Options 서비스 예외 시 원본 반환."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        with patch(
            "src.data.options.service.OptionsDataService",
            side_effect=Exception("service unavailable"),
        ):
            result = service._maybe_enrich_options(df, request)

        assert result is df


class TestDerivExtEnrichment:
    """Extended derivatives enrichment 검증."""

    @patch("src.data.deriv_ext.service.DerivExtDataService")
    def test_enrichment_applied(self, mock_cls: MagicMock) -> None:
        """DerivExt enrichment가 적용되면 dext_* 컬럼 추가."""
        df = _make_ohlcv()
        enriched_cols = pd.DataFrame(
            {
                "dext_agg_oi": [1e9] * len(df),
                "dext_funding_rate": [0.01] * len(df),
            },
            index=df.index,
        )
        mock_cls.return_value.precompute.return_value = enriched_cols

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        request = _make_request()

        result = service._maybe_enrich_deriv_ext(df, request)

        assert "dext_agg_oi" in result.columns
        assert "dext_funding_rate" in result.columns
        mock_cls.return_value.precompute.assert_called_once_with(
            ohlcv_index=df.index,
            asset="BTC",
        )

    def test_asset_extraction(self) -> None:
        """Symbol에서 asset 추출 (BTC/USDT → BTC)."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()

        request = MagicMock(spec=MarketDataRequest)
        request.symbol = "ETH/USDT"

        with patch("src.data.deriv_ext.service.DerivExtDataService") as mock_cls:
            mock_cls.return_value.precompute.return_value = pd.DataFrame()
            service._maybe_enrich_deriv_ext(df, request)
            mock_cls.return_value.precompute.assert_called_once_with(
                ohlcv_index=df.index,
                asset="ETH",
            )

    @patch("src.data.deriv_ext.service.DerivExtDataService")
    def test_no_data_returns_original(self, mock_cls: MagicMock) -> None:
        """DerivExt 데이터 없으면 원본 반환."""
        mock_cls.return_value.precompute.return_value = pd.DataFrame()

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        result = service._maybe_enrich_deriv_ext(df, request)

        assert result is df

    def test_graceful_degradation(self) -> None:
        """DerivExt 서비스 예외 시 원본 반환."""
        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        df = _make_ohlcv()
        request = _make_request()

        with patch(
            "src.data.deriv_ext.service.DerivExtDataService",
            side_effect=Exception("service unavailable"),
        ):
            result = service._maybe_enrich_deriv_ext(df, request)

        assert result is df


class TestGetMultiEnrichesAll:
    """get_multi()도 전체 enrichment 수행 검증."""

    @patch.object(MarketDataService, "get")
    def test_get_multi_calls_get_without_flags(self, mock_get: MagicMock) -> None:
        """get_multi()가 get()을 플래그 없이 호출."""
        from datetime import UTC, datetime

        df = _make_ohlcv()
        mock_data = MagicMock()
        mock_data.ohlcv = df
        mock_data.ohlcv.index = df.index
        mock_get.return_value = mock_data

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        service.silver_processor = MagicMock()

        service.get_multi(
            symbols=["BTC/USDT"],
            timeframe="1D",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 12, 31, tzinfo=UTC),
        )

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        # include_onchain / include_derivatives 키워드가 없어야 함
        assert "include_onchain" not in kwargs
        assert "include_derivatives" not in kwargs


class TestAllEnrichmentsCalledInGet:
    """get()에서 5종 enrichment이 모두 호출되는지 검증."""

    @patch.object(MarketDataService, "_maybe_enrich_deriv_ext", return_value=_make_ohlcv())
    @patch.object(MarketDataService, "_maybe_enrich_options", return_value=_make_ohlcv())
    @patch.object(MarketDataService, "_maybe_enrich_macro", return_value=_make_ohlcv())
    @patch.object(MarketDataService, "_maybe_enrich_onchain", return_value=_make_ohlcv())
    @patch.object(MarketDataService, "_maybe_enrich_derivatives", return_value=_make_ohlcv())
    @patch.object(MarketDataService, "_resample", return_value=_make_ohlcv())
    def test_all_enrichments_called(
        self,
        mock_resample: MagicMock,
        mock_deriv: MagicMock,
        mock_onchain: MagicMock,
        mock_macro: MagicMock,
        mock_options: MagicMock,
        mock_deriv_ext: MagicMock,
    ) -> None:
        """get() 호출 시 5종 enrichment 메서드가 모두 호출됨."""
        from datetime import UTC, datetime

        service = MarketDataService.__new__(MarketDataService)
        service.settings = MagicMock()
        service.silver_processor = MagicMock()

        # silver_processor.load가 OHLCV 반환하도록 설정
        service.silver_processor.load.return_value = _make_ohlcv()

        request = MarketDataRequest(
            symbol="BTC/USDT",
            timeframe="1D",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 10, tzinfo=UTC),
        )

        service.get(request)

        mock_deriv.assert_called_once()
        mock_onchain.assert_called_once()
        mock_macro.assert_called_once()
        mock_options.assert_called_once()
        mock_deriv_ext.assert_called_once()
