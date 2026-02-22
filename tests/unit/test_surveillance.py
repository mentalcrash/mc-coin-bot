"""Tests for MarketSurveillanceService."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pydantic
import pytest

from src.orchestrator.surveillance import (
    MarketSurveillanceService,
    ScanResult,
    SurveillanceConfig,
)


class TestSurveillanceConfig:
    """SurveillanceConfig Pydantic 기본값/검증."""

    def test_defaults(self) -> None:
        cfg = SurveillanceConfig()
        assert cfg.enabled is False
        assert cfg.scan_interval_hours == 168.0
        assert cfg.min_24h_volume_usd == 50_000_000.0
        assert cfg.min_listing_age_days == 90
        assert cfg.exclude_stablecoins is True
        assert cfg.max_assets_per_pod == 10
        assert cfg.max_total_assets == 20
        assert cfg.quote_currency == "USDT"

    def test_frozen(self) -> None:
        cfg = SurveillanceConfig()
        with pytest.raises(pydantic.ValidationError):
            cfg.enabled = True  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = SurveillanceConfig(
            enabled=True,
            scan_interval_hours=24.0,
            min_24h_volume_usd=100_000_000.0,
            max_total_assets=50,
        )
        assert cfg.enabled is True
        assert cfg.scan_interval_hours == 24.0
        assert cfg.max_total_assets == 50


class TestScanResult:
    """ScanResult frozen dataclass."""

    def test_immutable(self) -> None:
        result = ScanResult(
            timestamp=datetime.now(UTC),
            qualified_symbols=("BTC/USDT",),
            added=("BTC/USDT",),
            dropped=(),
            retained=(),
            scan_duration_seconds=1.0,
            total_scanned=100,
        )
        with pytest.raises(AttributeError):
            result.added = ("ETH/USDT",)  # type: ignore[misc]


class TestMarketSurveillanceService:
    """MarketSurveillanceService 단위 테스트."""

    def _make_service(
        self,
        *,
        enabled: bool = True,
        max_total_assets: int = 5,
    ) -> tuple[MarketSurveillanceService, AsyncMock]:
        cfg = SurveillanceConfig(
            enabled=enabled,
            max_total_assets=max_total_assets,
        )
        mock_client = AsyncMock()
        service = MarketSurveillanceService(config=cfg, client=mock_client)
        return service, mock_client

    @pytest.mark.asyncio
    async def test_scan_basic(self) -> None:
        service, client = self._make_service(max_total_assets=3)
        client.fetch_top_symbols.return_value = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "XRP/USDT",
        ]

        result = await service.scan()

        assert len(result.qualified_symbols) == 3
        assert result.added == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
        assert result.dropped == ()
        assert result.retained == ()
        assert result.total_scanned == 4

    @pytest.mark.asyncio
    async def test_scan_filters_stablecoins(self) -> None:
        service, client = self._make_service(max_total_assets=10)
        client.fetch_top_symbols.return_value = [
            "BTC/USDT",
            "USDC/USDT",
            "ETH/USDT",
            "FDUSD/USDT",
        ]

        result = await service.scan()

        assert "USDC/USDT" not in result.qualified_symbols
        assert "FDUSD/USDT" not in result.qualified_symbols
        assert "BTC/USDT" in result.qualified_symbols
        assert "ETH/USDT" in result.qualified_symbols

    @pytest.mark.asyncio
    async def test_scan_diff_computation(self) -> None:
        service, client = self._make_service(max_total_assets=10)

        # First scan
        client.fetch_top_symbols.return_value = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        r1 = await service.scan()
        assert len(r1.added) == 3
        assert len(r1.dropped) == 0

        # Second scan — XRP added, SOL dropped
        client.fetch_top_symbols.return_value = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        r2 = await service.scan()
        assert "XRP/USDT" in r2.added
        assert "SOL/USDT" in r2.dropped
        assert "BTC/USDT" in r2.retained
        assert "ETH/USDT" in r2.retained

    @pytest.mark.asyncio
    async def test_max_total_assets_cap(self) -> None:
        service, client = self._make_service(max_total_assets=2)
        client.fetch_top_symbols.return_value = [
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
        ]

        result = await service.scan()
        assert len(result.qualified_symbols) == 2

    @pytest.mark.asyncio
    async def test_persistence_round_trip(self) -> None:
        service, client = self._make_service(max_total_assets=10)
        client.fetch_top_symbols.return_value = ["BTC/USDT", "ETH/USDT"]
        await service.scan()

        # Serialize
        data = service.to_dict()
        assert "BTC/USDT" in data["current_universe"]
        assert data["last_scan_ts"] is not None

        # Restore to new service
        service2, _ = self._make_service(max_total_assets=10)
        service2.restore_from_dict(data)
        assert service2.current_universe == service.current_universe
        assert service2.last_scan_ts is not None

    @pytest.mark.asyncio
    async def test_scan_history_trimmed(self) -> None:
        service, client = self._make_service(max_total_assets=5)
        client.fetch_top_symbols.return_value = ["BTC/USDT"]

        for _ in range(105):
            await service.scan()

        assert len(service.scan_history) == 100

    def test_current_universe_sorted(self) -> None:
        service, _ = self._make_service()
        service._current_universe = {"SOL/USDT", "BTC/USDT", "ETH/USDT"}
        assert service.current_universe == ("BTC/USDT", "ETH/USDT", "SOL/USDT")
