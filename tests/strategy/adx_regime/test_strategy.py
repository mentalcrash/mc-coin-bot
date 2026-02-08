"""Tests for ADXRegimeStrategy (Integration)."""

import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.adx_regime import ADXRegimeConfig, ADXRegimeStrategy


class TestRegistry:
    """전략 Registry 통합 테스트."""

    def test_strategy_registered(self) -> None:
        """adx-regime이 Registry에 등록됨."""
        available = list_strategies()
        assert "adx-regime" in available

    def test_get_strategy(self) -> None:
        """get_strategy로 클래스 조회."""
        strategy_class = get_strategy("adx-regime")
        assert strategy_class == ADXRegimeStrategy


class TestADXRegimeStrategy:
    """ADXRegimeStrategy 클래스 테스트."""

    def test_strategy_properties(self) -> None:
        """기본 속성 확인."""
        strategy = ADXRegimeStrategy()

        assert strategy.name == "ADX-Regime"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, ADXRegimeConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run() end-to-end 파이프라인."""
        strategy = ADXRegimeStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "adx" in processed_df.columns
        assert "vw_momentum" in processed_df.columns
        assert "z_score" in processed_df.columns
        assert "vol_scalar" in processed_df.columns

        # 시그널 구조 확인
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

        # direction 값 범위
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_from_params(self, sample_ohlcv: pd.DataFrame) -> None:
        """from_params()로 전략 생성 (parameter sweep 호환)."""
        strategy = ADXRegimeStrategy.from_params(
            adx_period=10,
            adx_low=12.0,
            adx_high=22.0,
            vol_target=0.40,
        )

        assert strategy.config.adx_period == 10
        assert strategy.config.adx_low == 12.0
        assert strategy.config.adx_high == 22.0
        assert strategy.config.vol_target == 0.40

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self) -> None:
        """recommended_config 값 확인."""
        config = ADXRegimeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 3.0

    def test_warmup_periods(self) -> None:
        """warmup_periods 반환값이 config와 일치."""
        strategy = ADXRegimeStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()

    def test_get_startup_info(self) -> None:
        """get_startup_info 키 확인."""
        strategy = ADXRegimeStrategy()
        info = strategy.get_startup_info()

        assert "adx_band" in info
        assert "mom_lookback" in info
        assert "mr_lookback" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """conservative() 팩토리 동작."""
        strategy = ADXRegimeStrategy.conservative()
        assert strategy.config.adx_period == 20
        assert strategy.config.vol_target == 0.15
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """aggressive() 팩토리 동작."""
        strategy = ADXRegimeStrategy.aggressive()
        assert strategy.config.adx_period == 10
        assert strategy.config.vol_target == 0.40
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_for_timeframe(self) -> None:
        """for_timeframe() 팩토리."""
        strategy = ADXRegimeStrategy.for_timeframe("4h")
        assert strategy.config.annualization_factor == 2190.0

    def test_validate_input_missing_columns(self) -> None:
        """필수 컬럼 누락 시 에러."""
        strategy = ADXRegimeStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)
