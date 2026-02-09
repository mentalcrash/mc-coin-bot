"""Tests for FundingCarryStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.funding_carry import FundingCarryConfig, FundingCarryStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_ohlcv_with_funding() -> pd.DataFrame:
    """샘플 OHLCV + funding_rate DataFrame 생성 (200일)."""
    np.random.seed(42)
    n = 200
    close = 50000.0 + np.cumsum(np.random.randn(n) * 300)
    funding_rate = np.random.randn(n) * 0.0003

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "funding_rate": funding_rate,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self) -> None:
        """'funding-carry'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "funding-carry" in available

    def test_get_strategy(self) -> None:
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("funding-carry")
        assert strategy_class == FundingCarryStrategy


class TestFundingCarryStrategy:
    """FundingCarryStrategy 테스트."""

    def test_properties(self) -> None:
        """전략 속성 테스트."""
        strategy = FundingCarryStrategy()

        assert strategy.name == "Funding Rate Carry"
        assert set(strategy.required_columns) == {
            "close",
            "high",
            "low",
            "volume",
            "funding_rate",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, FundingCarryConfig)

    def test_required_columns_includes_funding_rate(self) -> None:
        """required_columns에 funding_rate 포함 확인."""
        strategy = FundingCarryStrategy()
        assert "funding_rate" in strategy.required_columns

    def test_run_pipeline(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """전체 파이프라인 (run) 테스트."""
        strategy = FundingCarryStrategy()
        processed_df, signals = strategy.run(sample_ohlcv_with_funding)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_ohlcv_with_funding)
        assert len(signals.entries) == len(sample_ohlcv_with_funding)

        # 시그널 구조 확인
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        """from_params()로 전략 생성."""
        strategy = FundingCarryStrategy.from_params(
            lookback=5,
            zscore_window=60,
            vol_target=0.25,
        )

        assert strategy.config.lookback == 5
        assert strategy.config.zscore_window == 60
        assert strategy.config.vol_target == 0.25

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_ohlcv_with_funding)
        assert len(signals.entries) == len(sample_ohlcv_with_funding)

    def test_recommended_config(self) -> None:
        """recommended_config() 테스트."""
        config = FundingCarryStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        strategy = FundingCarryStrategy()
        warmup = strategy.warmup_periods()

        # max(90, 30) + 1 = 91
        assert warmup == 91

    def test_validate_missing_funding_rate(self) -> None:
        """funding_rate 컬럼 없는 데이터로 validate_input 실패."""
        strategy = FundingCarryStrategy()

        df = pd.DataFrame(
            {
                "open": [50000.0],
                "high": [50200.0],
                "low": [49800.0],
                "close": [50100.0],
                "volume": [1000000.0],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="D"),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_input(df)

    def test_get_startup_info(self) -> None:
        """get_startup_info() 테스트."""
        strategy = FundingCarryStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "lookback" in info
        assert "zscore_window" in info
        assert "vol_target" in info
        assert "entry_threshold" in info
        assert "mode" in info
        assert info["mode"] == "Long/Short"

    def test_params_property(self) -> None:
        """params 프로퍼티가 설정 딕셔너리 반환."""
        strategy = FundingCarryStrategy()
        params = strategy.params

        assert isinstance(params, dict)
        assert "lookback" in params
        assert "zscore_window" in params
        assert "vol_target" in params
        assert "entry_threshold" in params

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = FundingCarryConfig(
            lookback=5,
            zscore_window=120,
            vol_target=0.25,
            entry_threshold=0.0005,
        )
        strategy = FundingCarryStrategy(config)

        assert strategy.config.lookback == 5
        assert strategy.config.zscore_window == 120
        assert strategy.config.vol_target == 0.25
        assert strategy.config.entry_threshold == 0.0005
