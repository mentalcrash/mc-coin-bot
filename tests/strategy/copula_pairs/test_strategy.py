"""Tests for CopulaPairsStrategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.copula_pairs import CopulaPairsConfig, CopulaPairsStrategy
from src.strategy.types import Direction


@pytest.fixture
def sample_pairs_data() -> pd.DataFrame:
    """합성 cointegrated pair 데이터 생성."""
    np.random.seed(42)
    n = 200
    common_factor = np.cumsum(np.random.randn(n) * 200)
    close = 50000.0 + common_factor + np.random.randn(n) * 100
    pair_close = 3000.0 + common_factor * 0.06 + np.random.randn(n) * 50

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 100,
            "high": close + np.abs(np.random.randn(n) * 200),
            "low": close - np.abs(np.random.randn(n) * 200),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float) * 1000,
            "pair_close": pair_close,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


class TestRegistry:
    """Strategy Registry 테스트."""

    def test_registered(self) -> None:
        """'copula-pairs'로 Registry에 등록되었는지 확인."""
        available = list_strategies()
        assert "copula-pairs" in available

    def test_get_strategy(self) -> None:
        """get_strategy()로 전략 클래스 조회."""
        strategy_class = get_strategy("copula-pairs")
        assert strategy_class == CopulaPairsStrategy


class TestCopulaPairsStrategy:
    """CopulaPairsStrategy 테스트."""

    def test_properties(self) -> None:
        """전략 속성 테스트."""
        strategy = CopulaPairsStrategy()

        assert strategy.name == "Copula Pairs"
        assert set(strategy.required_columns) == {
            "close",
            "high",
            "low",
            "volume",
            "pair_close",
        }
        assert strategy.config is not None
        assert isinstance(strategy.config, CopulaPairsConfig)

    def test_run_pipeline(self, sample_pairs_data: pd.DataFrame) -> None:
        """전체 파이프라인 (run) 테스트."""
        strategy = CopulaPairsStrategy()
        processed_df, signals = strategy.run(sample_pairs_data)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_pairs_data)
        assert len(signals.entries) == len(sample_pairs_data)

        # 시그널 타입 검증
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool
        assert signals.direction.dtype == int
        assert set(signals.direction.unique()).issubset(
            {Direction.SHORT, Direction.NEUTRAL, Direction.LONG}
        )

    def test_from_params(self, sample_pairs_data: pd.DataFrame) -> None:
        """from_params()로 전략 생성."""
        strategy = CopulaPairsStrategy.from_params(
            formation_window=50,
            zscore_entry=1.5,
            zscore_exit=0.3,
            zscore_stop=2.5,
        )

        assert strategy.config.formation_window == 50
        assert strategy.config.zscore_entry == 1.5
        assert strategy.config.zscore_exit == 0.3
        assert strategy.config.zscore_stop == 2.5

        # 전체 파이프라인 정상 동작 확인
        _processed_df, signals = strategy.run(sample_pairs_data)
        assert len(signals.entries) == len(sample_pairs_data)

    def test_recommended_config(self) -> None:
        """recommended_config() 테스트."""
        config = CopulaPairsStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_get_startup_info(self) -> None:
        """get_startup_info() 테스트."""
        strategy = CopulaPairsStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "formation_window" in info
        assert "zscore_entry" in info
        assert "zscore_exit" in info
        assert "zscore_stop" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_warmup_periods(self) -> None:
        """warmup_periods() 테스트."""
        strategy = CopulaPairsStrategy()
        warmup = strategy.warmup_periods()

        # formation_window=63 + 1 = 64
        assert warmup == 64

    def test_validate_input(self, sample_pairs_data: pd.DataFrame) -> None:
        """validate_input 테스트 - 필수 컬럼 누락."""
        strategy = CopulaPairsStrategy()

        # pair_close 없이 호출하면 에러
        df_missing = sample_pairs_data.drop(columns=["pair_close"])
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_input(df_missing)

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = CopulaPairsConfig(
            formation_window=100,
            zscore_entry=1.5,
            zscore_exit=0.3,
            zscore_stop=2.5,
        )
        strategy = CopulaPairsStrategy(config)

        assert strategy.config.formation_window == 100
        assert strategy.config.zscore_entry == 1.5
