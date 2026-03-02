"""BacktestSurveillanceSimulator 단위 테스트."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.orchestrator.backtest_surveillance import BacktestSurveillanceSimulator
from src.orchestrator.surveillance import SurveillanceConfig
from src.orchestrator.volume_matrix import VolumeMatrix


# ── Helpers ───────────────────────────────────────────────────────


def _make_volume_matrix(
    symbols: list[str],
    volumes: list[float],
    start: str = "2025-01-01",
    days: int = 30,
) -> VolumeMatrix:
    """테스트용 VolumeMatrix 생성.

    Args:
        symbols: 심볼 리스트
        volumes: 심볼별 일별 거래대금 (고정)
    """
    dates = pd.date_range(start, periods=days, freq="1D", tz=UTC)
    daily_volume: dict[str, pd.Series] = {}
    for sym, vol in zip(symbols, volumes):
        daily_volume[sym] = pd.Series([vol] * days, index=dates)
    return VolumeMatrix(daily_volume=daily_volume)


def _default_config(**kwargs: object) -> SurveillanceConfig:
    """기본 SurveillanceConfig 생성."""
    defaults = {
        "enabled": True,
        "scan_interval_hours": 168.0,  # 7일
        "max_total_assets": 5,
    }
    defaults.update(kwargs)
    return SurveillanceConfig(**defaults)  # type: ignore[arg-type]


# ── Tests: should_scan ────────────────────────────────────────────


class TestShouldScan:
    def test_first_scan_always_true(self) -> None:
        matrix = _make_volume_matrix(["A"], [100.0])
        sim = BacktestSurveillanceSimulator(
            config=_default_config(),
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A"},
        )
        assert sim.should_scan(datetime(2025, 1, 1, tzinfo=UTC))

    def test_disabled_never_scans(self) -> None:
        matrix = _make_volume_matrix(["A"], [100.0])
        sim = BacktestSurveillanceSimulator(
            config=_default_config(enabled=False),
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A"},
        )
        assert not sim.should_scan(datetime(2025, 1, 1, tzinfo=UTC))

    def test_interval_check(self) -> None:
        matrix = _make_volume_matrix(["A"], [100.0])
        config = _default_config(scan_interval_hours=24.0)
        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A"},
        )
        ts1 = datetime(2025, 1, 1, tzinfo=UTC)
        sim.scan_at(ts1)  # sets _last_scan_ts

        # 12시간 후 → 아직 스캔 안함
        assert not sim.should_scan(ts1 + timedelta(hours=12))

        # 24시간 후 → 스캔 실행
        assert sim.should_scan(ts1 + timedelta(hours=24))


# ── Tests: scan_at ────────────────────────────────────────────────


class TestScanAt:
    def test_basic_scan(self) -> None:
        """초기 시드가 아닌 심볼이 volume 순위에 포함 → added."""
        symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
        volumes = [1000.0, 800.0, 600.0, 400.0, 200.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=3)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"BTC"},
            available_symbols=set(symbols),
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        # 상위 3: BTC, ETH, SOL
        assert set(result.qualified_symbols) == {"BTC", "ETH", "SOL"}
        assert "BTC" in result.retained
        assert "ETH" in result.added
        assert "SOL" in result.added
        assert len(result.dropped) == 0

    def test_dropped_symbols(self) -> None:
        """기존 유니버스에서 탈락."""
        symbols = ["A", "B", "C"]
        volumes = [1000.0, 800.0, 600.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=2)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A", "B", "C"},
            available_symbols=set(symbols),
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        # 상위 2: A, B → C dropped
        assert "C" in result.dropped
        assert set(result.retained) == {"A", "B"}

    def test_stablecoin_filter(self) -> None:
        """스테이블코인은 제외."""
        symbols = ["BTC/USDT", "USDC/USDT", "ETH/USDT"]
        volumes = [1000.0, 5000.0, 800.0]  # USDC가 거래량 최대
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=3)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"BTC/USDT"},
            available_symbols=set(symbols),
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        # USDC/USDT는 stablecoin → 제외
        assert "USDC/USDT" not in result.qualified_symbols
        assert "BTC/USDT" in result.qualified_symbols
        assert "ETH/USDT" in result.qualified_symbols

    def test_available_symbols_filter(self) -> None:
        """데이터 미보유 심볼은 자동 제외."""
        symbols = ["A", "B", "C"]
        volumes = [1000.0, 800.0, 600.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=3)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A", "B"},  # C 미보유
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        assert "C" not in result.qualified_symbols
        assert set(result.qualified_symbols) == {"A", "B"}

    def test_scan_history(self) -> None:
        """스캔 히스토리 누적."""
        symbols = ["A"]
        volumes = [100.0]
        matrix = _make_volume_matrix(symbols, volumes, days=30)
        config = _default_config(scan_interval_hours=24.0, max_total_assets=1)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A"},
        )

        sim.scan_at(datetime(2025, 1, 1, tzinfo=UTC))
        sim.scan_at(datetime(2025, 1, 8, tzinfo=UTC))

        assert len(sim.scan_history) == 2
        assert sim.scan_history[0].timestamp == datetime(2025, 1, 1, tzinfo=UTC)
        assert sim.scan_history[1].timestamp == datetime(2025, 1, 8, tzinfo=UTC)

    def test_universe_update_on_scan(self) -> None:
        """scan 후 current_universe 갱신."""
        symbols = ["A", "B", "C"]
        volumes = [1000.0, 800.0, 600.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=2)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols=set(symbols),
        )

        sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        assert set(sim.current_universe) == {"A", "B"}

    def test_no_change_scan(self) -> None:
        """변경 없는 경우 added/dropped 비어있음."""
        symbols = ["A", "B"]
        volumes = [1000.0, 800.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=2)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A", "B"},
            available_symbols=set(symbols),
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))

        assert len(result.added) == 0
        assert len(result.dropped) == 0
        assert set(result.retained) == {"A", "B"}

    def test_scan_result_frozen(self) -> None:
        """ScanResult는 frozen dataclass."""
        symbols = ["A"]
        volumes = [100.0]
        matrix = _make_volume_matrix(symbols, volumes)
        config = _default_config(max_total_assets=1)

        sim = BacktestSurveillanceSimulator(
            config=config,
            volume_matrix=matrix,
            seed_symbols={"A"},
            available_symbols={"A"},
        )

        result = sim.scan_at(datetime(2025, 1, 10, tzinfo=UTC))
        with pytest.raises(AttributeError):
            result.added = ("B",)  # type: ignore[misc]
