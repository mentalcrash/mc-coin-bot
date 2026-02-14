"""DerivativesSnapshotFetcher 테스트."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.derivatives_snapshot import DerivativesSnapshotFetcher
from src.notification.health_models import SymbolDerivativesSnapshot

# ─── Fixtures ──────────────────────────────────────────────


def _make_mock_futures_client() -> MagicMock:
    """BinanceFuturesClient mock 생성."""
    client = MagicMock()
    client.fetch_ticker = AsyncMock(return_value={"last": 97420.0})
    client.fetch_funding_rate_history = AsyncMock(return_value=[{"fundingRate": 0.00023}])
    client.fetch_open_interest_history = AsyncMock(
        return_value=[{"sumOpenInterestValue": 5_000_000_000}]
    )
    client.fetch_long_short_ratio = AsyncMock(return_value=[{"longShortRatio": 1.47}])
    client.fetch_taker_buy_sell_ratio = AsyncMock(return_value=[{"buySellRatio": 1.12}])
    return client


# ─── Via BinanceFuturesClient 테스트 ─────────────────────


class TestFetchViaFuturesClient:
    @pytest.mark.asyncio
    async def test_fetch_symbol_success(self) -> None:
        client = _make_mock_futures_client()
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        result = await fetcher.fetch_symbol("BTC/USDT")

        assert result is not None
        assert isinstance(result, SymbolDerivativesSnapshot)
        assert result.symbol == "BTC/USDT"
        assert result.price == 97420.0
        assert result.funding_rate == pytest.approx(0.00023)
        assert result.ls_ratio == pytest.approx(1.47)
        assert result.taker_ratio == pytest.approx(1.12)

    @pytest.mark.asyncio
    async def test_fetch_symbol_annualized_rate(self) -> None:
        client = _make_mock_futures_client()
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        result = await fetcher.fetch_symbol("BTC/USDT")

        assert result is not None
        # 0.00023 * 3 * 365 * 100 = 25.185
        expected_ann = 0.00023 * 3 * 365 * 100
        assert result.funding_rate_annualized == pytest.approx(expected_ann, rel=0.01)

    @pytest.mark.asyncio
    async def test_fetch_symbol_empty_responses(self) -> None:
        """빈 응답 시 기본값 사용."""
        client = _make_mock_futures_client()
        client.fetch_funding_rate_history = AsyncMock(return_value=[])
        client.fetch_open_interest_history = AsyncMock(return_value=[])
        client.fetch_long_short_ratio = AsyncMock(return_value=[])
        client.fetch_taker_buy_sell_ratio = AsyncMock(return_value=[])
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        result = await fetcher.fetch_symbol("BTC/USDT")

        assert result is not None
        assert result.funding_rate == 0.0
        assert result.open_interest == 0.0
        assert result.ls_ratio == 1.0
        assert result.taker_ratio == 1.0

    @pytest.mark.asyncio
    async def test_fetch_symbol_exception_returns_none(self) -> None:
        client = _make_mock_futures_client()
        client.fetch_ticker = AsyncMock(side_effect=RuntimeError("network error"))
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        result = await fetcher.fetch_symbol("BTC/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_success(self) -> None:
        client = _make_mock_futures_client()
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        results = await fetcher.fetch_all(["BTC/USDT", "ETH/USDT"])

        assert len(results) == 2
        assert results[0].symbol == "BTC/USDT"
        assert results[1].symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_fetch_all_partial_failure(self) -> None:
        """일부 심볼 실패 시 성공한 것만 반환."""
        client = _make_mock_futures_client()

        async def ticker_side_effect(symbol: str, **_: Any) -> dict[str, Any]:
            if "BTC" in symbol:
                raise RuntimeError("fail")
            return {"last": 3280.0}

        client.fetch_ticker = AsyncMock(side_effect=ticker_side_effect)
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        results = await fetcher.fetch_all(["BTC/USDT", "ETH/USDT"])

        assert len(results) == 1
        assert results[0].symbol == "ETH/USDT"


# ─── Start/Stop 라이프사이클 테스트 ──────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_with_futures_client_noop(self) -> None:
        """futures_client 있으면 내부 exchange 생성 안 함."""
        client = _make_mock_futures_client()
        fetcher = DerivativesSnapshotFetcher(futures_client=client)

        await fetcher.start()

        assert fetcher._own_exchange is None

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        """start 없이 stop 해도 에러 안 남."""
        fetcher = DerivativesSnapshotFetcher()
        await fetcher.stop()  # 예외 없음

    @pytest.mark.asyncio
    async def test_fetch_without_start_returns_none(self) -> None:
        """start 없이 fetch 시 None 반환."""
        fetcher = DerivativesSnapshotFetcher()
        result = await fetcher.fetch_symbol("BTC/USDT")
        assert result is None


# ─── Via Own Exchange 테스트 ─────────────────────────────


class TestFetchViaOwnExchange:
    @pytest.mark.asyncio
    async def test_fetch_via_own_exchange(self) -> None:
        """내부 ccxt exchange를 통한 조회."""
        fetcher = DerivativesSnapshotFetcher()

        # Mock internal exchange
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 3280.0})
        mock_exchange.fetch_funding_rate_history = AsyncMock(
            return_value=[{"fundingRate": 0.00015}]
        )
        mock_exchange.fapipublic_get_futures_data_openinteresthist = AsyncMock(
            return_value=[{"sumOpenInterestValue": 2_000_000_000}]
        )
        mock_exchange.fapipublic_get_futures_data_globallongshortaccountratio = AsyncMock(
            return_value=[{"longShortRatio": 1.31}]
        )
        mock_exchange.fapipublic_get_futures_data_takerlongshortratio = AsyncMock(
            return_value=[{"buySellRatio": 0.98}]
        )

        fetcher._own_exchange = mock_exchange

        result = await fetcher.fetch_symbol("ETH/USDT")

        assert result is not None
        assert result.symbol == "ETH/USDT"
        assert result.price == 3280.0
        assert result.funding_rate == pytest.approx(0.00015)
        assert result.ls_ratio == pytest.approx(1.31)

    @pytest.mark.asyncio
    async def test_start_creates_own_exchange(self) -> None:
        """futures_client 없으면 start()에서 내부 exchange 생성."""
        fetcher = DerivativesSnapshotFetcher()

        mock_exchange = MagicMock()
        mock_exchange.load_markets = AsyncMock()

        with patch("src.data.derivatives_snapshot.ccxt.binance", return_value=mock_exchange):
            await fetcher.start()

        assert fetcher._own_exchange is mock_exchange
        mock_exchange.load_markets.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_closes_own_exchange(self) -> None:
        """stop()에서 내부 exchange close."""
        fetcher = DerivativesSnapshotFetcher()
        mock_exchange = MagicMock()
        mock_exchange.close = AsyncMock()
        fetcher._own_exchange = mock_exchange

        await fetcher.stop()

        mock_exchange.close.assert_awaited_once()
        assert fetcher._own_exchange is None
