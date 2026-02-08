"""Unit tests for RiskMomStrategy."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.base import BaseStrategy
from src.strategy.registry import get_strategy, list_strategies
from src.strategy.risk_mom.config import RiskMomConfig
from src.strategy.risk_mom.strategy import RiskMomStrategy


class TestStrategyRegistry:
    """전략 레지스트리 등록 테스트."""

    def test_strategy_registered(self) -> None:
        """'risk-mom'이 레지스트리에 등록되어 있는지 확인."""
        assert "risk-mom" in list_strategies()

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스를 반환하는지 확인."""
        strategy_cls = get_strategy("risk-mom")
        assert strategy_cls is RiskMomStrategy

    def test_is_base_strategy_subclass(self) -> None:
        """BaseStrategy의 서브클래스인지 확인."""
        assert issubclass(RiskMomStrategy, BaseStrategy)


class TestStrategyProperties:
    """전략 프로퍼티 테스트."""

    def test_strategy_name(self) -> None:
        """name 프로퍼티가 'Risk-Mom'인지 확인."""
        strategy = RiskMomStrategy()
        assert strategy.name == "Risk-Mom"

    def test_required_columns(self) -> None:
        """required_columns가 OHLCV 전체인지 확인."""
        strategy = RiskMomStrategy()
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]

    def test_config_property(self) -> None:
        """config 프로퍼티가 RiskMomConfig를 반환하는지 확인."""
        strategy = RiskMomStrategy()
        assert isinstance(strategy.config, RiskMomConfig)

    def test_config_custom(self) -> None:
        """커스텀 config가 올바르게 설정되는지 확인."""
        config = RiskMomConfig(lookback=50, var_window=180)
        strategy = RiskMomStrategy(config)
        assert strategy.config.lookback == 50
        assert strategy.config.var_window == 180

    def test_default_config(self) -> None:
        """config=None일 때 기본 설정으로 초기화."""
        strategy = RiskMomStrategy(config=None)
        assert strategy.config.lookback == 30
        assert strategy.config.var_window == 126


class TestRunPipeline:
    """전체 파이프라인(run) 테스트."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run()으로 전처리 + 시그널 생성 파이프라인 실행."""
        strategy = RiskMomStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 결과 검증
        assert "bsc_scaling" in processed_df.columns
        assert "vw_momentum" in processed_df.columns
        assert "realized_var" in processed_df.columns

        # 시그널 검증
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_pipeline_with_conservative(self, sample_ohlcv: pd.DataFrame) -> None:
        """보수적 설정으로 파이프라인 실행."""
        strategy = RiskMomStrategy.conservative()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "bsc_scaling" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_pipeline_with_aggressive(self, sample_ohlcv: pd.DataFrame) -> None:
        """공격적 설정으로 파이프라인 실행."""
        strategy = RiskMomStrategy.aggressive()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "bsc_scaling" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_empty_df_raises(self) -> None:
        """빈 DataFrame에서 ValueError 발생."""
        strategy = RiskMomStrategy()
        empty_df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
        )
        empty_df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(empty_df)


class TestFromParams:
    """from_params() 팩토리 메서드 테스트."""

    def test_from_params(self) -> None:
        """파라미터 딕셔너리로 전략 생성."""
        strategy = RiskMomStrategy.from_params(
            lookback=20,
            var_window=100,
            vol_target=0.25,
        )
        assert isinstance(strategy, RiskMomStrategy)
        assert strategy.config.lookback == 20
        assert strategy.config.var_window == 100
        assert strategy.config.vol_target == 0.25

    def test_from_params_default(self) -> None:
        """파라미터 없이 기본값으로 생성."""
        strategy = RiskMomStrategy.from_params()
        assert strategy.config.lookback == 30
        assert strategy.config.var_window == 126

    def test_from_params_invalid_raises(self) -> None:
        """유효하지 않은 파라미터에서 ValidationError 발생."""
        with pytest.raises(ValidationError):
            RiskMomStrategy.from_params(lookback=1)  # < min(6)


class TestRecommendedConfig:
    """recommended_config() 테스트."""

    def test_recommended_config(self) -> None:
        """권장 PM 설정이 올바른 키를 포함하는지 확인."""
        config = RiskMomStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "max_leverage_cap" in config
        assert "system_stop_loss" in config
        assert "rebalance_threshold" in config

    def test_recommended_config_values(self) -> None:
        """권장 PM 설정값 검증."""
        config = RiskMomStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05


class TestWarmupPeriods:
    """warmup_periods() 테스트."""

    def test_warmup_periods(self) -> None:
        """전략의 warmup_periods가 config와 일치하는지 확인."""
        strategy = RiskMomStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()

    def test_warmup_periods_conservative(self) -> None:
        """보수적 설정의 warmup_periods."""
        strategy = RiskMomStrategy.conservative()
        expected = max(48, 180, 48) + 1  # var_window=180이 최대
        assert strategy.warmup_periods() == expected

    def test_warmup_periods_aggressive(self) -> None:
        """공격적 설정의 warmup_periods."""
        strategy = RiskMomStrategy.aggressive()
        expected = max(12, 63, 12) + 1  # var_window=63이 최대
        assert strategy.warmup_periods() == expected


class TestTimeframeFactory:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임 전략 생성."""
        strategy = RiskMomStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임 전략 생성."""
        strategy = RiskMomStrategy.for_timeframe("1h")
        assert strategy.config.annualization_factor == 8760.0


class TestStartupInfo:
    """get_startup_info() 테스트."""

    def test_startup_info_keys(self) -> None:
        """시작 패널 정보에 핵심 키가 포함되는지 확인."""
        strategy = RiskMomStrategy()
        info = strategy.get_startup_info()
        assert "lookback" in info
        assert "var_window" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_startup_info_hedge_mode(self) -> None:
        """HEDGE_ONLY 모드에서 hedge_strength 키 포함."""
        config = RiskMomConfig(short_mode=1)  # HEDGE_ONLY
        strategy = RiskMomStrategy(config)
        info = strategy.get_startup_info()
        assert "hedge_strength" in info

    def test_startup_info_full_mode_no_hedge_strength(self) -> None:
        """FULL 모드에서 hedge_strength 키 미포함."""
        config = RiskMomConfig(short_mode=2)  # FULL
        strategy = RiskMomStrategy(config)
        info = strategy.get_startup_info()
        assert "hedge_strength" not in info
