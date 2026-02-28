"""BacktestEngine._validate_required_enrichments 테스트.

전략의 필수 enrichment 컬럼 커버리지 검증 로직을 테스트합니다.
- 필수 enrichment 없는 기본 전략 → 통과
- 필수 컬럼 누락 → ValueError
- NaN 비율 초과 → ValueError
- NaN 비율 이하 → 통과
- 에러 메시지에 remediation 명령 포함 확인
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.data.market_data import MarketDataSet
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NoEnrichmentStrategy(BaseStrategy):
    """required_enrichments가 비어 있는 기본 전략."""

    @property
    def name(self) -> str:
        return "no-enrichment"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        idx = df.index
        return StrategySignals(
            entries=pd.Series(False, index=idx),
            exits=pd.Series(False, index=idx),
            direction=pd.Series(0, index=idx, dtype=int),
            strength=pd.Series(0.0, index=idx),
        )


class _TflowEnrichmentStrategy(BaseStrategy):
    """tflow_ prefix enrichment을 요구하는 전략."""

    @property
    def name(self) -> str:
        return "tflow-strategy"

    @property
    def required_columns(self) -> list[str]:
        return ["close"]

    @property
    def required_enrichments(self) -> list[str]:
        return ["tflow_cvd", "tflow_vpin"]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        idx = df.index
        return StrategySignals(
            entries=pd.Series(False, index=idx),
            exits=pd.Series(False, index=idx),
            direction=pd.Series(0, index=idx, dtype=int),
            strength=pd.Series(0.0, index=idx),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_market_data(
    df: pd.DataFrame,
    symbol: str = "BTC/USDT",
) -> MarketDataSet:
    """테스트용 MarketDataSet 생성."""
    return MarketDataSet(
        symbol=symbol,
        timeframe="1D",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 9, tzinfo=UTC),
        ohlcv=df,
    )


@pytest.fixture
def base_df() -> pd.DataFrame:
    """100행 OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz=UTC)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 100))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1000, 5000, 100),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateRequiredEnrichments:
    """BacktestEngine._validate_required_enrichments 단위 테스트."""

    def test_no_required_enrichments_passes(self, base_df: pd.DataFrame) -> None:
        """필수 enrichment 없는 전략은 검증을 무조건 통과한다."""
        strategy = _NoEnrichmentStrategy()
        data = _make_market_data(base_df)

        # 예외 없이 통과해야 함
        BacktestEngine._validate_required_enrichments(strategy, data)

    def test_missing_columns_raises_value_error(self, base_df: pd.DataFrame) -> None:
        """필수 enrichment 컬럼이 DataFrame에 없으면 ValueError."""
        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(base_df)  # tflow_cvd, tflow_vpin 없음

        with pytest.raises(ValueError, match="missing from the data"):
            BacktestEngine._validate_required_enrichments(strategy, data)

    def test_missing_columns_error_contains_remediation(self, base_df: pd.DataFrame) -> None:
        """에러 메시지에 tflow_ prefix 기반 remediation 명령이 포함된다."""
        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(base_df)

        with pytest.raises(ValueError, match="uv run mcbot ingest trade-flow pipeline"):
            BacktestEngine._validate_required_enrichments(strategy, data)

    def test_high_nan_ratio_raises_value_error(self, base_df: pd.DataFrame) -> None:
        """NaN 비율이 20% 초과하면 ValueError."""
        df = base_df.copy()
        # 25% NaN (100행 중 25행)
        df["tflow_cvd"] = 1.0
        df.iloc[:25, df.columns.get_loc("tflow_cvd")] = np.nan
        df["tflow_vpin"] = 1.0

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        with pytest.raises(ValueError, match="excessive NaN"):
            BacktestEngine._validate_required_enrichments(strategy, data)

    def test_nan_ratio_at_boundary_passes(self, base_df: pd.DataFrame) -> None:
        """NaN 비율이 정확히 20%이면 통과 (> 20%만 실패)."""
        df = base_df.copy()
        # 정확히 20% NaN (100행 중 20행)
        df["tflow_cvd"] = 1.0
        df.iloc[:20, df.columns.get_loc("tflow_cvd")] = np.nan
        df["tflow_vpin"] = 1.0

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        # 20% 이하이므로 통과
        BacktestEngine._validate_required_enrichments(strategy, data)

    def test_nan_ratio_below_threshold_passes(self, base_df: pd.DataFrame) -> None:
        """NaN 비율이 20% 미만이면 통과."""
        df = base_df.copy()
        # 10% NaN (100행 중 10행)
        df["tflow_cvd"] = 1.0
        df.iloc[:10, df.columns.get_loc("tflow_cvd")] = np.nan
        df["tflow_vpin"] = 1.0

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        BacktestEngine._validate_required_enrichments(strategy, data)

    def test_nan_error_message_contains_column_ratios(self, base_df: pd.DataFrame) -> None:
        """NaN 에러 메시지에 컬럼별 NaN 비율이 포함된다."""
        df = base_df.copy()
        # tflow_cvd: 30% NaN, tflow_vpin: 50% NaN
        df["tflow_cvd"] = 1.0
        df.iloc[:30, df.columns.get_loc("tflow_cvd")] = np.nan
        df["tflow_vpin"] = 1.0
        df.iloc[:50, df.columns.get_loc("tflow_vpin")] = np.nan

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        with pytest.raises(ValueError, match=r"tflow_cvd=30%") as exc_info:
            BacktestEngine._validate_required_enrichments(strategy, data)

        # 두 컬럼 모두 에러에 포함되어야 함
        assert "tflow_vpin=50%" in str(exc_info.value)

    def test_empty_dataframe_with_enrichment_columns_passes(self) -> None:
        """빈 DataFrame이지만 컬럼이 있으면 통과 (n_rows == 0 early return)."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "tflow_cvd", "tflow_vpin"]
        )
        df.index = pd.DatetimeIndex([], tz=UTC)

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        BacktestEngine._validate_required_enrichments(strategy, data)

    def test_custom_max_nan_ratio(self, base_df: pd.DataFrame) -> None:
        """max_nan_ratio 커스텀 값으로 임계값을 조정할 수 있다."""
        df = base_df.copy()
        # 15% NaN — 기본 20%로는 통과하지만 10%로 설정하면 실패
        df["tflow_cvd"] = 1.0
        df.iloc[:15, df.columns.get_loc("tflow_cvd")] = np.nan
        df["tflow_vpin"] = 1.0

        strategy = _TflowEnrichmentStrategy()
        data = _make_market_data(df)

        # 기본 max_nan_ratio=0.2 → 통과
        BacktestEngine._validate_required_enrichments(strategy, data)

        # max_nan_ratio=0.1 → 실패
        with pytest.raises(ValueError, match="excessive NaN"):
            BacktestEngine._validate_required_enrichments(strategy, data, max_nan_ratio=0.1)
