"""StrategyEngine._check_live_enrichments 테스트.

라이브 모드에서 필수 enrichment 데이터의 무결성 검증 로직을 테스트합니다.
- required_enrichments가 빈 전략 → None (정상)
- 컬럼 누락 → 에러 메시지 반환
- 최신 bar에 NaN → 에러 메시지 반환
- 모든 컬럼 존재 + 유효 → None (정상)
"""

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from src.eda.strategy_engine import StrategyEngine
from src.strategy.base import BaseStrategy
from src.strategy.types import StrategySignals

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _NoEnrichmentStrategy(BaseStrategy):
    """required_enrichments가 빈 전략."""

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
    """tflow_ enrichment을 요구하는 전략."""

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


@pytest.fixture
def base_df() -> pd.DataFrame:
    """10행 OHLCV DataFrame (라이브 버퍼 시뮬레이션)."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz=UTC)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, 10))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.uniform(1000, 5000, 10),
        },
        index=dates,
    )


def _make_engine(strategy: BaseStrategy) -> StrategyEngine:
    """StrategyEngine 인스턴스 생성 (warmup 최소화)."""
    return StrategyEngine(strategy, warmup_periods=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckLiveEnrichments:
    """StrategyEngine._check_live_enrichments 단위 테스트."""

    def test_no_required_enrichments_returns_none(self, base_df: pd.DataFrame) -> None:
        """required_enrichments가 빈 전략 → None (정상)."""
        engine = _make_engine(_NoEnrichmentStrategy())
        result = engine._check_live_enrichments(base_df, "BTC/USDT")
        assert result is None

    def test_missing_column_returns_error_message(self, base_df: pd.DataFrame) -> None:
        """필수 enrichment 컬럼이 없으면 에러 메시지를 반환한다."""
        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(base_df, "BTC/USDT")

        assert result is not None
        assert "missing columns" in result
        assert "tflow_cvd" in result
        assert "tflow_vpin" in result

    def test_partial_missing_column(self, base_df: pd.DataFrame) -> None:
        """일부 컬럼만 누락되어도 에러 메시지를 반환한다."""
        df = base_df.copy()
        df["tflow_cvd"] = 1.0  # tflow_vpin만 누락

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is not None
        assert "missing columns" in result
        assert "tflow_vpin" in result
        # 존재하는 컬럼은 missing에 포함되면 안 됨
        assert "tflow_cvd" not in result

    def test_nan_in_latest_bar_returns_error_message(self, base_df: pd.DataFrame) -> None:
        """최신 bar(마지막 행)에 NaN이 있으면 에러 메시지를 반환한다."""
        df = base_df.copy()
        df["tflow_cvd"] = 1.0
        df["tflow_vpin"] = 1.0
        # 마지막 행만 NaN으로 설정
        df.iloc[-1, df.columns.get_loc("tflow_cvd")] = np.nan

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is not None
        assert "NaN in latest bar" in result
        assert "tflow_cvd" in result

    def test_nan_in_non_latest_bar_passes(self, base_df: pd.DataFrame) -> None:
        """과거 bar에 NaN이 있어도 최신 bar가 유효하면 통과."""
        df = base_df.copy()
        df["tflow_cvd"] = 1.0
        df["tflow_vpin"] = 1.0
        # 과거 행에만 NaN (최신 행은 정상)
        df.iloc[0, df.columns.get_loc("tflow_cvd")] = np.nan
        df.iloc[3, df.columns.get_loc("tflow_vpin")] = np.nan

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is None

    def test_all_columns_present_and_valid_returns_none(self, base_df: pd.DataFrame) -> None:
        """모든 필수 컬럼이 존재하고 최신 bar에 NaN이 없으면 None."""
        df = base_df.copy()
        df["tflow_cvd"] = 1.0
        df["tflow_vpin"] = 0.5

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is None

    def test_empty_dataframe_returns_none(self) -> None:
        """빈 DataFrame은 None을 반환 (df.empty early return)."""
        df = pd.DataFrame(columns=["close", "tflow_cvd", "tflow_vpin"])
        df.index = pd.DatetimeIndex([], tz=UTC)

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is None

    def test_multiple_nan_columns_in_latest_bar(self, base_df: pd.DataFrame) -> None:
        """최신 bar에서 여러 컬럼이 NaN이면 모두 에러에 포함된다."""
        df = base_df.copy()
        df["tflow_cvd"] = 1.0
        df["tflow_vpin"] = 1.0
        # 마지막 행 두 컬럼 모두 NaN
        df.iloc[-1, df.columns.get_loc("tflow_cvd")] = np.nan
        df.iloc[-1, df.columns.get_loc("tflow_vpin")] = np.nan

        engine = _make_engine(_TflowEnrichmentStrategy())
        result = engine._check_live_enrichments(df, "BTC/USDT")

        assert result is not None
        assert "tflow_cvd" in result
        assert "tflow_vpin" in result
