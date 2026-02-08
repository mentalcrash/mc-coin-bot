"""Unit tests for DonchianEnsembleStrategy."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.base import BaseStrategy
from src.strategy.donchian_ensemble.config import DonchianEnsembleConfig
from src.strategy.donchian_ensemble.strategy import DonchianEnsembleStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    """전략 레지스트리 등록 테스트."""

    def test_strategy_registered(self) -> None:
        """'donchian-ensemble'이 레지스트리에 등록되어 있는지 확인."""
        assert "donchian-ensemble" in list_strategies()

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스를 반환하는지 확인."""
        strategy_cls = get_strategy("donchian-ensemble")
        assert strategy_cls is DonchianEnsembleStrategy

    def test_is_base_strategy_subclass(self) -> None:
        """BaseStrategy의 서브클래스인지 확인."""
        assert issubclass(DonchianEnsembleStrategy, BaseStrategy)


class TestStrategyProperties:
    """전략 프로퍼티 테스트."""

    def test_strategy_name(self) -> None:
        """name 프로퍼티가 'Donchian-Ensemble'인지 확인."""
        strategy = DonchianEnsembleStrategy()
        assert strategy.name == "Donchian-Ensemble"

    def test_required_columns(self) -> None:
        """required_columns가 high, low, close인지 확인."""
        strategy = DonchianEnsembleStrategy()
        assert strategy.required_columns == ["high", "low", "close"]

    def test_config_property(self) -> None:
        """config 프로퍼티가 DonchianEnsembleConfig를 반환하는지 확인."""
        strategy = DonchianEnsembleStrategy()
        assert isinstance(strategy.config, DonchianEnsembleConfig)

    def test_config_custom(self) -> None:
        """커스텀 config가 올바르게 설정되는지 확인."""
        config = DonchianEnsembleConfig(lookbacks=(10, 20, 50), vol_target=0.30)
        strategy = DonchianEnsembleStrategy(config)
        assert strategy.config.lookbacks == (10, 20, 50)
        assert strategy.config.vol_target == 0.30

    def test_default_config(self) -> None:
        """config=None일 때 기본 설정으로 초기화."""
        strategy = DonchianEnsembleStrategy(config=None)
        assert strategy.config.lookbacks == (5, 10, 20, 30, 60, 90, 150, 250, 360)
        assert strategy.config.vol_target == 0.40


class TestRunPipeline:
    """전체 파이프라인(run) 테스트."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run()으로 전처리 + 시그널 생성 파이프라인 실행."""
        strategy = DonchianEnsembleStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 결과 검증
        assert "dc_upper_5" in processed_df.columns
        assert "dc_lower_360" in processed_df.columns
        assert "realized_vol" in processed_df.columns
        assert "vol_scalar" in processed_df.columns

        # 시그널 검증
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_pipeline_with_conservative(self, sample_ohlcv: pd.DataFrame) -> None:
        """보수적 설정으로 파이프라인 실행."""
        strategy = DonchianEnsembleStrategy.conservative()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "dc_upper_20" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_pipeline_with_aggressive(self, sample_ohlcv: pd.DataFrame) -> None:
        """공격적 설정으로 파이프라인 실행."""
        strategy = DonchianEnsembleStrategy.aggressive()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "dc_upper_5" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_empty_df_raises(self) -> None:
        """빈 DataFrame에서 ValueError 발생."""
        strategy = DonchianEnsembleStrategy()
        empty_df = pd.DataFrame(
            columns=["high", "low", "close"],
        )
        empty_df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(empty_df)


class TestFromParams:
    """from_params() 팩토리 메서드 테스트."""

    def test_from_params(self) -> None:
        """파라미터 딕셔너리로 전략 생성."""
        strategy = DonchianEnsembleStrategy.from_params(
            lookbacks=(10, 20, 50),
            vol_target=0.25,
        )
        assert isinstance(strategy, DonchianEnsembleStrategy)
        assert strategy.config.lookbacks == (10, 20, 50)
        assert strategy.config.vol_target == 0.25

    def test_from_params_default(self) -> None:
        """파라미터 없이 기본값으로 생성."""
        strategy = DonchianEnsembleStrategy.from_params()
        assert strategy.config.lookbacks == (5, 10, 20, 30, 60, 90, 150, 250, 360)
        assert strategy.config.vol_target == 0.40

    def test_from_params_invalid_raises(self) -> None:
        """유효하지 않은 파라미터에서 ValidationError 발생."""
        with pytest.raises(ValidationError):
            DonchianEnsembleStrategy.from_params(atr_period=1)  # < min(5)


class TestRecommendedConfig:
    """recommended_config() 테스트."""

    def test_recommended_config(self) -> None:
        """권장 PM 설정이 올바른 키를 포함하는지 확인."""
        config = DonchianEnsembleStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "max_leverage_cap" in config
        assert "system_stop_loss" in config
        assert "rebalance_threshold" in config

    def test_recommended_config_values(self) -> None:
        """권장 PM 설정값 검증."""
        config = DonchianEnsembleStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05


class TestWarmupPeriods:
    """warmup_periods() 테스트."""

    def test_warmup_periods(self) -> None:
        """전략의 warmup_periods가 config와 일치하는지 확인."""
        strategy = DonchianEnsembleStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()

    def test_warmup_periods_default(self) -> None:
        """기본 설정의 warmup_periods = 361."""
        strategy = DonchianEnsembleStrategy()
        assert strategy.warmup_periods() == 361

    def test_warmup_periods_conservative(self) -> None:
        """보수적 설정의 warmup_periods."""
        strategy = DonchianEnsembleStrategy.conservative()
        expected = max(20, 30, 60, 90, 150, 250, 360) + 1
        assert strategy.warmup_periods() == expected

    def test_warmup_periods_aggressive(self) -> None:
        """공격적 설정의 warmup_periods."""
        strategy = DonchianEnsembleStrategy.aggressive()
        expected = max(5, 10, 20, 30, 60) + 1
        assert strategy.warmup_periods() == expected


class TestTimeframeFactory:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임 전략 생성."""
        strategy = DonchianEnsembleStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임 전략 생성."""
        strategy = DonchianEnsembleStrategy.for_timeframe("1h")
        assert strategy.config.annualization_factor == 8760.0


class TestStartupInfo:
    """get_startup_info() 테스트."""

    def test_startup_info_keys(self) -> None:
        """시작 패널 정보에 핵심 키가 포함되는지 확인."""
        strategy = DonchianEnsembleStrategy()
        info = strategy.get_startup_info()
        assert "lookbacks" in info
        assert "num_channels" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_startup_info_disabled_mode(self) -> None:
        """DISABLED 모드에서 mode 값 확인."""
        strategy = DonchianEnsembleStrategy()
        info = strategy.get_startup_info()
        assert info["mode"] == "Long-Only"

    def test_startup_info_full_mode(self) -> None:
        """FULL 모드에서 mode 값 확인."""
        config = DonchianEnsembleConfig(short_mode=2)  # FULL
        strategy = DonchianEnsembleStrategy(config)
        info = strategy.get_startup_info()
        assert info["mode"] == "Long/Short"

    def test_startup_info_num_channels(self) -> None:
        """num_channels가 lookback 수와 일치하는지 확인."""
        strategy = DonchianEnsembleStrategy()
        info = strategy.get_startup_info()
        assert info["num_channels"] == "9"  # 기본 9개
