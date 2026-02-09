"""Unit tests for XSMOMStrategy."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.base import BaseStrategy
from src.strategy.registry import get_strategy, list_strategies
from src.strategy.xsmom.config import XSMOMConfig
from src.strategy.xsmom.strategy import XSMOMStrategy


class TestStrategyRegistry:
    """전략 레지스트리 등록 테스트."""

    def test_strategy_registered(self) -> None:
        """'xsmom'이 레지스트리에 등록되어 있는지 확인."""
        assert "xsmom" in list_strategies()

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스를 반환하는지 확인."""
        strategy_cls = get_strategy("xsmom")
        assert strategy_cls is XSMOMStrategy

    def test_is_base_strategy_subclass(self) -> None:
        """BaseStrategy의 서브클래스인지 확인."""
        assert issubclass(XSMOMStrategy, BaseStrategy)


class TestStrategyProperties:
    """전략 프로퍼티 테스트."""

    def test_strategy_name(self) -> None:
        """name 프로퍼티가 'XSMOM'인지 확인."""
        strategy = XSMOMStrategy()
        assert strategy.name == "XSMOM"

    def test_required_columns(self) -> None:
        """required_columns가 close, high, low, volume인지 확인."""
        strategy = XSMOMStrategy()
        assert strategy.required_columns == ["close", "high", "low", "volume"]

    def test_config_property(self) -> None:
        """config 프로퍼티가 XSMOMConfig를 반환하는지 확인."""
        strategy = XSMOMStrategy()
        assert isinstance(strategy.config, XSMOMConfig)

    def test_config_custom(self) -> None:
        """커스텀 config가 올바르게 설정되는지 확인."""
        config = XSMOMConfig(lookback=30, holding_period=14, vol_target=0.25)
        strategy = XSMOMStrategy(config)
        assert strategy.config.lookback == 30
        assert strategy.config.holding_period == 14
        assert strategy.config.vol_target == 0.25

    def test_default_config(self) -> None:
        """config=None일 때 기본 설정으로 초기화."""
        strategy = XSMOMStrategy(config=None)
        assert strategy.config.lookback == 21
        assert strategy.config.holding_period == 7
        assert strategy.config.vol_target == 0.35


class TestRunPipeline:
    """전체 파이프라인(run) 테스트."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run()으로 전처리 + 시그널 생성 파이프라인 실행."""
        strategy = XSMOMStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 결과 검증
        assert "rolling_return" in processed_df.columns
        assert "realized_vol" in processed_df.columns
        assert "vol_scalar" in processed_df.columns
        assert "atr" in processed_df.columns

        # 시그널 검증
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_empty_df_raises(self) -> None:
        """빈 DataFrame에서 ValueError 발생."""
        strategy = XSMOMStrategy()
        empty_df = pd.DataFrame(
            columns=["close", "high", "low", "volume"],
        )
        empty_df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(empty_df)


class TestFromParams:
    """from_params() 팩토리 메서드 테스트."""

    def test_from_params(self) -> None:
        """파라미터 딕셔너리로 전략 생성."""
        strategy = XSMOMStrategy.from_params(
            lookback=30,
            holding_period=14,
            vol_target=0.25,
        )
        assert isinstance(strategy, XSMOMStrategy)
        assert strategy.config.lookback == 30
        assert strategy.config.holding_period == 14
        assert strategy.config.vol_target == 0.25

    def test_from_params_default(self) -> None:
        """파라미터 없이 기본값으로 생성."""
        strategy = XSMOMStrategy.from_params()
        assert strategy.config.lookback == 21
        assert strategy.config.vol_target == 0.35

    def test_from_params_invalid_raises(self) -> None:
        """유효하지 않은 파라미터에서 ValidationError 발생."""
        with pytest.raises(ValidationError):
            XSMOMStrategy.from_params(lookback=2)  # < min(5)


class TestRecommendedConfig:
    """recommended_config() 테스트."""

    def test_recommended_config(self) -> None:
        """권장 PM 설정이 올바른 키를 포함하는지 확인."""
        config = XSMOMStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "max_leverage_cap" in config
        assert "system_stop_loss" in config
        assert "rebalance_threshold" in config

    def test_recommended_config_values(self) -> None:
        """권장 PM 설정값 검증."""
        config = XSMOMStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.10


class TestStartupInfo:
    """get_startup_info() 테스트."""

    def test_startup_info_keys(self) -> None:
        """시작 패널 정보에 핵심 키가 포함되는지 확인."""
        strategy = XSMOMStrategy()
        info = strategy.get_startup_info()
        assert "lookback" in info
        assert "holding_period" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_startup_info_default_mode(self) -> None:
        """기본 FULL 모드에서 mode 값 확인."""
        strategy = XSMOMStrategy()
        info = strategy.get_startup_info()
        assert info["mode"] == "Long/Short"


class TestWarmupPeriods:
    """warmup_periods() 테스트."""

    def test_warmup_periods(self) -> None:
        """전략의 warmup_periods가 config와 일치하는지 확인."""
        strategy = XSMOMStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()

    def test_warmup_periods_default(self) -> None:
        """기본 설정의 warmup_periods = 31."""
        strategy = XSMOMStrategy()
        assert strategy.warmup_periods() == 31

    def test_warmup_periods_conservative(self) -> None:
        """보수적 설정의 warmup_periods."""
        strategy = XSMOMStrategy.conservative()
        expected = max(60, 60) + 1
        assert strategy.warmup_periods() == expected

    def test_warmup_periods_aggressive(self) -> None:
        """공격적 설정의 warmup_periods."""
        strategy = XSMOMStrategy.aggressive()
        expected = max(10, 15) + 1
        assert strategy.warmup_periods() == expected


class TestValidateInput:
    """validate_input() 테스트."""

    def test_validate_input_passes(self, sample_ohlcv: pd.DataFrame) -> None:
        """올바른 입력에서 에러 없음."""
        strategy = XSMOMStrategy()
        strategy.validate_input(sample_ohlcv)  # should not raise

    def test_validate_input_missing_column(self) -> None:
        """필수 컬럼 누락 시 ValueError 발생."""
        strategy = XSMOMStrategy()
        df = pd.DataFrame(
            {"close": [1, 2, 3]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_input(df)
