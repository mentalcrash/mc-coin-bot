"""VolumeMatrix 단위 테스트."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.orchestrator.volume_matrix import VolumeMatrix, compute_volume_matrix, rank_at


# ── Fixtures ──────────────────────────────────────────────────────


def _make_1m_ohlcv(
    symbol: str,
    start: str,
    days: int,
    base_close: float = 100.0,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    """테스트용 1m OHLCV DataFrame 생성."""
    bars_per_day = 1440  # 24 * 60
    total_bars = days * bars_per_day
    index = pd.date_range(start, periods=total_bars, freq="1min", tz=UTC)
    rng = np.random.default_rng(hash(symbol) & 0xFFFFFFFF)
    close = base_close + rng.standard_normal(total_bars).cumsum() * 0.1
    close = np.maximum(close, 1.0)  # 음수 방지
    volume = base_volume + rng.uniform(0, 500, total_bars)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture()
def three_symbol_ohlcv() -> dict[str, pd.DataFrame]:
    """3개 심볼 14일 1m OHLCV."""
    return {
        "BTC/USDT": _make_1m_ohlcv("BTC", "2025-01-01", 14, base_close=40000, base_volume=5000),
        "ETH/USDT": _make_1m_ohlcv("ETH", "2025-01-01", 14, base_close=2500, base_volume=3000),
        "DOGE/USDT": _make_1m_ohlcv("DOGE", "2025-01-01", 14, base_close=0.1, base_volume=100000),
    }


# ── Tests: compute_volume_matrix ─────────────────────────────────


class TestComputeVolumeMatrix:
    def test_basic_structure(self, three_symbol_ohlcv: dict[str, pd.DataFrame]) -> None:
        matrix = compute_volume_matrix(three_symbol_ohlcv)
        assert isinstance(matrix, VolumeMatrix)
        assert set(matrix.symbols) == {"BTC/USDT", "ETH/USDT", "DOGE/USDT"}

    def test_daily_aggregation(self, three_symbol_ohlcv: dict[str, pd.DataFrame]) -> None:
        matrix = compute_volume_matrix(three_symbol_ohlcv)
        # 14일 데이터 → 14일 daily volume
        for sym in matrix.symbols:
            assert len(matrix.daily_volume[sym]) == 14

    def test_daily_volume_positive(self, three_symbol_ohlcv: dict[str, pd.DataFrame]) -> None:
        matrix = compute_volume_matrix(three_symbol_ohlcv)
        for sym in matrix.symbols:
            assert (matrix.daily_volume[sym] > 0).all()

    def test_empty_df_skipped(self) -> None:
        ohlcv = {
            "A": pd.DataFrame(columns=["open", "high", "low", "close", "volume"]),
            "B": _make_1m_ohlcv("B", "2025-01-01", 3, base_close=100, base_volume=100),
        }
        matrix = compute_volume_matrix(ohlcv)
        assert "A" not in matrix.symbols
        assert "B" in matrix.symbols

    def test_quote_volume_formula(self) -> None:
        """quote_volume = close * volume 검증."""
        index = pd.date_range("2025-01-01", periods=1440, freq="1min", tz=UTC)
        df = pd.DataFrame(
            {
                "open": [100.0] * 1440,
                "high": [101.0] * 1440,
                "low": [99.0] * 1440,
                "close": [100.0] * 1440,
                "volume": [10.0] * 1440,
            },
            index=index,
        )
        matrix = compute_volume_matrix({"SYM": df})
        expected_daily = 100.0 * 10.0 * 1440  # close * volume * bars_per_day
        assert abs(matrix.daily_volume["SYM"].iloc[0] - expected_daily) < 1e-6


# ── Tests: rank_at ────────────────────────────────────────────────


class TestRankAt:
    def test_ranking_order(self) -> None:
        """BTC > ETH > DOGE volume 순서 검증."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1D", tz=UTC)
        matrix = VolumeMatrix(
            daily_volume={
                "BTC": pd.Series([1000.0] * 10, index=dates),
                "ETH": pd.Series([500.0] * 10, index=dates),
                "DOGE": pd.Series([100.0] * 10, index=dates),
            }
        )
        result = rank_at(matrix, datetime(2025, 1, 10, tzinfo=UTC), top_n=3)
        assert result == ["BTC", "ETH", "DOGE"]

    def test_top_n_limit(self) -> None:
        """top_n=2 → 상위 2개만."""
        dates = pd.date_range("2025-01-01", periods=10, freq="1D", tz=UTC)
        matrix = VolumeMatrix(
            daily_volume={
                "A": pd.Series([1000.0] * 10, index=dates),
                "B": pd.Series([500.0] * 10, index=dates),
                "C": pd.Series([100.0] * 10, index=dates),
            }
        )
        result = rank_at(matrix, datetime(2025, 1, 10, tzinfo=UTC), top_n=2)
        assert len(result) == 2
        assert result == ["A", "B"]

    def test_rolling_window(self) -> None:
        """rolling window 내 데이터만 사용."""
        dates = pd.date_range("2025-01-01", periods=14, freq="1D", tz=UTC)
        # A: 처음 7일만 큰 거래량, 이후 작음
        vol_a = [1000.0] * 7 + [10.0] * 7
        # B: 처음 7일은 작고, 이후 7일 큼
        vol_b = [10.0] * 7 + [1000.0] * 7
        matrix = VolumeMatrix(
            daily_volume={
                "A": pd.Series(vol_a, index=dates),
                "B": pd.Series(vol_b, index=dates),
            }
        )
        # 7일차에서 볼 때 → A가 1위
        result_early = rank_at(matrix, datetime(2025, 1, 7, tzinfo=UTC), rolling_window_days=7, top_n=2)
        assert result_early[0] == "A"

        # 14일차에서 볼 때 → B가 1위
        result_late = rank_at(matrix, datetime(2025, 1, 14, tzinfo=UTC), rolling_window_days=7, top_n=2)
        assert result_late[0] == "B"

    def test_no_data_before_timestamp(self) -> None:
        """timestamp 이전에 데이터 없으면 빈 리스트."""
        dates = pd.date_range("2025-01-10", periods=5, freq="1D", tz=UTC)
        matrix = VolumeMatrix(
            daily_volume={"A": pd.Series([100.0] * 5, index=dates)}
        )
        result = rank_at(matrix, datetime(2025, 1, 5, tzinfo=UTC))
        assert result == []

    def test_single_symbol(self) -> None:
        """심볼 1개 → 해당 심볼만 반환."""
        dates = pd.date_range("2025-01-01", periods=7, freq="1D", tz=UTC)
        matrix = VolumeMatrix(
            daily_volume={"ONLY": pd.Series([100.0] * 7, index=dates)}
        )
        result = rank_at(matrix, datetime(2025, 1, 7, tzinfo=UTC))
        assert result == ["ONLY"]

    def test_with_full_ohlcv(self, three_symbol_ohlcv: dict[str, pd.DataFrame]) -> None:
        """compute_volume_matrix → rank_at 통합 검증."""
        matrix = compute_volume_matrix(three_symbol_ohlcv)
        result = rank_at(matrix, datetime(2025, 1, 14, tzinfo=UTC), top_n=3)
        assert len(result) == 3
        # BTC가 close*volume 기준으로 가장 클 것 (close=40000, vol=5000)
        assert result[0] == "BTC/USDT"
