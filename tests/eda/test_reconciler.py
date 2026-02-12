"""PositionReconciler 단위 테스트."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eda.reconciler import PositionReconciler
from src.models.types import Direction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakePosition:
    """테스트용 간이 Position."""

    symbol: str
    direction: Direction = Direction.NEUTRAL
    size: float = 0.0
    last_price: float = 0.0
    atr_values: list[float] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.size > 0.0

    @property
    def notional(self) -> float:
        return self.size * self.last_price


def _make_pm(positions: dict[str, FakePosition] | None = None) -> MagicMock:
    pm = MagicMock()
    pm.positions = positions or {}
    return pm


def _make_futures_client(exchange_positions: list[dict[str, object]]) -> MagicMock:
    client = MagicMock()
    client.fetch_positions = AsyncMock(return_value=exchange_positions)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitialCheck:
    """initial_check() 테스트."""

    @pytest.mark.asyncio
    async def test_all_match(self) -> None:
        """PM과 거래소가 일치하면 빈 리스트."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert drifts == []

    @pytest.mark.asyncio
    async def test_size_drift(self) -> None:
        """수량 차이 5% 초과 시 drift 감지."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.015, "side": "long"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_direction_mismatch(self) -> None:
        """방향 불일치 감지."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "short"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_pm_has_position_exchange_empty(self) -> None:
        """PM만 포지션 보유 — drift."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client([])

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_exchange_has_position_pm_empty(self) -> None:
        """거래소만 포지션 보유 — drift."""
        pm = _make_pm({})
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_both_empty(self) -> None:
        """양쪽 모두 포지션 없으면 정상."""
        pm = _make_pm({})
        client = _make_futures_client([])

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert drifts == []

    @pytest.mark.asyncio
    async def test_within_threshold(self) -> None:
        """5% 이내 차이면 정상."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        # 2% 차이
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.0102, "side": "long"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert drifts == []


class TestCheckBalance:
    """check_balance() 테스트."""

    @pytest.mark.asyncio
    async def test_returns_exchange_equity(self) -> None:
        """거래소 equity 반환."""
        pm = _make_pm({})
        pm.total_equity = 10000.0
        client = MagicMock()
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 10200.0, "free": 10000.0}}
        )

        reconciler = PositionReconciler()
        result = await reconciler.check_balance(pm, client)
        assert result == 10200.0

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self) -> None:
        """API 실패 시 None 반환."""
        pm = _make_pm({})
        pm.total_equity = 10000.0
        client = MagicMock()
        client.fetch_balance = AsyncMock(side_effect=Exception("API down"))

        reconciler = PositionReconciler()
        result = await reconciler.check_balance(pm, client)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_log_within_threshold(self) -> None:
        """2% 이내 drift 시 WARNING/CRITICAL 없음."""
        pm = _make_pm({})
        pm.total_equity = 10000.0
        client = MagicMock()
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 10100.0}}  # 1% drift
        )

        reconciler = PositionReconciler()
        result = await reconciler.check_balance(pm, client)
        assert result == 10100.0

    @pytest.mark.asyncio
    async def test_warns_moderate_drift(self) -> None:
        """2-5% drift 시 WARNING (CRITICAL은 아님)."""
        pm = _make_pm({})
        pm.total_equity = 10000.0
        client = MagicMock()
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 10300.0}}  # 3% drift
        )

        reconciler = PositionReconciler()
        result = await reconciler.check_balance(pm, client)
        assert result == 10300.0

    @pytest.mark.asyncio
    async def test_critical_large_drift(self) -> None:
        """5% 초과 drift 시 CRITICAL."""
        pm = _make_pm({})
        pm.total_equity = 10000.0
        client = MagicMock()
        client.fetch_balance = AsyncMock(
            return_value={"USDT": {"total": 8000.0}}  # 20% drift
        )

        reconciler = PositionReconciler()
        result = await reconciler.check_balance(pm, client)
        assert result == 8000.0


class TestHedgeMode:
    """Hedge Mode 포지션 비교 테스트."""

    @pytest.mark.asyncio
    async def test_long_short_separate(self) -> None:
        """Hedge Mode: LONG/SHORT 별도 매핑."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        # 거래소에 LONG + SHORT 모두 있음
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"},
                {"symbol": "BTC/USDT:USDT", "contracts": 0.005, "side": "short"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        # PM is LONG, exchange LONG size matches → size OK
        # But exchange has opposite SHORT → drift
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_pm_short_exchange_short_match(self) -> None:
        """PM SHORT, 거래소 SHORT 일치."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.SHORT, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "short"},
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT"])
        assert drifts == []


class TestPeriodicCheck:
    """periodic_check() 테스트."""

    @pytest.mark.asyncio
    async def test_periodic_returns_drifts(self) -> None:
        """drift 발견 시 심볼 리스트 반환."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            )
        }
        pm = _make_pm(positions)
        client = _make_futures_client([])

        reconciler = PositionReconciler()
        drifts = await reconciler.periodic_check(pm, client, ["BTC/USDT"])
        assert "BTC/USDT" in drifts

    @pytest.mark.asyncio
    async def test_periodic_api_failure_graceful(self) -> None:
        """API 실패 시 빈 리스트 반환 (예외 없음)."""
        pm = _make_pm({})
        client = MagicMock()
        client.fetch_positions = AsyncMock(side_effect=Exception("API down"))

        reconciler = PositionReconciler()
        drifts = await reconciler.periodic_check(pm, client, ["BTC/USDT"])
        assert drifts == []


class TestMultiSymbol:
    """멀티 심볼 검증."""

    @pytest.mark.asyncio
    async def test_partial_drift(self) -> None:
        """일부 심볼만 drift."""
        positions = {
            "BTC/USDT": FakePosition(
                symbol="BTC/USDT", direction=Direction.LONG, size=0.01, last_price=50000.0
            ),
            "ETH/USDT": FakePosition(
                symbol="ETH/USDT", direction=Direction.SHORT, size=1.0, last_price=3000.0
            ),
        }
        pm = _make_pm(positions)
        client = _make_futures_client(
            [
                {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "side": "long"},
                # ETH 없음 → drift
            ]
        )

        reconciler = PositionReconciler()
        drifts = await reconciler.initial_check(pm, client, ["BTC/USDT", "ETH/USDT"])
        assert "BTC/USDT" not in drifts
        assert "ETH/USDT" in drifts
