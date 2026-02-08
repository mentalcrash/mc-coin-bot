"""Tests for RSICrossoverStrategy (Integration)."""

import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.rsi_crossover import RSICrossoverConfig, RSICrossoverStrategy


class TestRegistry:
    """전략 Registry 통합 테스트."""

    def test_strategy_registered(self) -> None:
        """'rsi-crossover'가 Registry에 등록됨."""
        available = list_strategies()
        assert "rsi-crossover" in available

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스 조회."""
        strategy_class = get_strategy("rsi-crossover")
        assert strategy_class == RSICrossoverStrategy


class TestRSICrossoverStrategy:
    """RSICrossoverStrategy 클래스 테스트."""

    def test_strategy_properties(self) -> None:
        """기본 속성 확인."""
        strategy = RSICrossoverStrategy()

        assert strategy.name == "RSI-Crossover"
        assert set(strategy.required_columns) == {"open", "high", "low", "close", "volume"}
        assert isinstance(strategy.config, RSICrossoverConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run() end-to-end 파이프라인."""
        strategy = RSICrossoverStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "rsi" in processed_df.columns
        assert "vol_scalar" in processed_df.columns
        assert "atr" in processed_df.columns
        assert "drawdown" in processed_df.columns

        # 시그널 구조 확인
        assert len(signals.entries) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

        # direction 값 범위
        unique_dirs = set(signals.direction.unique())
        assert unique_dirs.issubset({-1, 0, 1})

    def test_from_params(self, sample_ohlcv: pd.DataFrame) -> None:
        """from_params()로 전략 생성 (parameter sweep 호환)."""
        strategy = RSICrossoverStrategy.from_params(
            rsi_period=10,
            vol_target=0.30,
            entry_oversold=30.0,
            entry_overbought=70.0,
        )

        assert strategy.config.rsi_period == 10
        assert strategy.config.vol_target == 0.30

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self) -> None:
        """recommended_config 값 확인."""
        config = RSICrossoverStrategy.recommended_config()

        assert config["execution_mode"] == "orders"
        assert config["max_leverage_cap"] == 1.5
        assert config["system_stop_loss"] == 0.08
        assert config["rebalance_threshold"] == 0.03
        assert config["use_trailing_stop"] is True
        assert config["trailing_stop_atr_multiplier"] == 1.5

    def test_warmup_periods(self) -> None:
        """warmup_periods 반환값이 config와 일치."""
        strategy = RSICrossoverStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()
        # 기본값: 14 + 30 + 2 = 46
        assert warmup == 46

    def test_for_timeframe(self) -> None:
        """for_timeframe() 팩토리."""
        strategy = RSICrossoverStrategy.for_timeframe("4h")
        assert strategy.config.annualization_factor == 2190.0

        strategy_1d = RSICrossoverStrategy.for_timeframe("1d")
        assert strategy_1d.config.annualization_factor == 365.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe()에 추가 파라미터 전달."""
        strategy = RSICrossoverStrategy.for_timeframe("4h", vol_target=0.35)
        assert strategy.config.annualization_factor == 2190.0
        assert strategy.config.vol_target == 0.35

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """conservative() 팩토리 동작."""
        strategy = RSICrossoverStrategy.conservative()
        assert strategy.config.entry_oversold == 25.0
        assert strategy.config.entry_overbought == 75.0
        assert strategy.config.vol_target == 0.15
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """aggressive() 팩토리 동작."""
        strategy = RSICrossoverStrategy.aggressive()
        assert strategy.config.entry_oversold == 35.0
        assert strategy.config.entry_overbought == 65.0
        assert strategy.config.vol_target == 0.35
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_get_startup_info(self) -> None:
        """get_startup_info 키 확인."""
        strategy = RSICrossoverStrategy()
        info = strategy.get_startup_info()

        assert "rsi_period" in info
        assert "entry" in info
        assert "exit" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_validate_input_missing_columns(self) -> None:
        """필수 컬럼 누락 시 에러."""
        strategy = RSICrossoverStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="4h"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_validate_input_empty_df(self) -> None:
        """빈 DataFrame 시 에러."""
        strategy = RSICrossoverStrategy()
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(df)

    def test_default_config(self) -> None:
        """기본 설정으로 생성된 전략의 config 확인."""
        strategy = RSICrossoverStrategy()
        assert strategy.config.rsi_period == 14
        assert strategy.config.entry_oversold == 30.0
        assert strategy.config.entry_overbought == 70.0

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = RSICrossoverConfig(rsi_period=10, vol_target=0.30)
        strategy = RSICrossoverStrategy(config)
        assert strategy.config.rsi_period == 10
        assert strategy.config.vol_target == 0.30

    def test_params_dict(self) -> None:
        """params 프로퍼티가 config의 model_dump와 일치."""
        strategy = RSICrossoverStrategy()
        params = strategy.params
        assert params["rsi_period"] == 14
        assert params["entry_oversold"] == 30.0
        assert params["entry_overbought"] == 70.0

    def test_repr(self) -> None:
        """__repr__ 문자열 확인."""
        strategy = RSICrossoverStrategy()
        repr_str = repr(strategy)
        assert "RSICrossoverStrategy" in repr_str
