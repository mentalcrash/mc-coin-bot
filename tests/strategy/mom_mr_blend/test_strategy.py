"""Unit tests for MomMrBlendStrategy."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.base import BaseStrategy
from src.strategy.mom_mr_blend.config import MomMrBlendConfig
from src.strategy.mom_mr_blend.strategy import MomMrBlendStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestStrategyRegistry:
    """전략 레지스트리 등록 테스트."""

    def test_strategy_registered(self) -> None:
        """'mom-mr-blend'가 레지스트리에 등록되어 있는지 확인."""
        assert "mom-mr-blend" in list_strategies()

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스를 반환하는지 확인."""
        strategy_cls = get_strategy("mom-mr-blend")
        assert strategy_cls is MomMrBlendStrategy

    def test_is_base_strategy_subclass(self) -> None:
        """BaseStrategy의 서브클래스인지 확인."""
        assert issubclass(MomMrBlendStrategy, BaseStrategy)


class TestStrategyProperties:
    """전략 프로퍼티 테스트."""

    def test_strategy_name(self) -> None:
        """name 프로퍼티가 'Mom-MR-Blend'인지 확인."""
        strategy = MomMrBlendStrategy()
        assert strategy.name == "Mom-MR-Blend"

    def test_required_columns(self) -> None:
        """required_columns가 ['close']인지 확인."""
        strategy = MomMrBlendStrategy()
        assert strategy.required_columns == ["close"]

    def test_config_property(self) -> None:
        """config 프로퍼티가 MomMrBlendConfig를 반환하는지 확인."""
        strategy = MomMrBlendStrategy()
        assert isinstance(strategy.config, MomMrBlendConfig)

    def test_config_custom(self) -> None:
        """커스텀 config가 올바르게 설정되는지 확인."""
        config = MomMrBlendConfig(mom_lookback=42, mr_lookback=21)
        strategy = MomMrBlendStrategy(config)
        assert strategy.config.mom_lookback == 42
        assert strategy.config.mr_lookback == 21

    def test_default_config(self) -> None:
        """config=None일 때 기본 설정으로 초기화."""
        strategy = MomMrBlendStrategy(config=None)
        assert strategy.config.mom_lookback == 28
        assert strategy.config.mr_lookback == 14


class TestRunPipeline:
    """전체 파이프라인(run) 테스트."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run()으로 전처리 + 시그널 생성 파이프라인 실행."""
        strategy = MomMrBlendStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 결과 검증
        assert "mom_zscore" in processed_df.columns
        assert "mr_zscore" in processed_df.columns
        assert "vol_scalar" in processed_df.columns

        # 시그널 검증
        assert len(signals.entries) == len(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_run_pipeline_with_conservative(self, sample_ohlcv: pd.DataFrame) -> None:
        """보수적 설정으로 파이프라인 실행."""
        strategy = MomMrBlendStrategy.conservative()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "mom_zscore" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_pipeline_with_aggressive(self, sample_ohlcv: pd.DataFrame) -> None:
        """공격적 설정으로 파이프라인 실행."""
        strategy = MomMrBlendStrategy.aggressive()
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "mom_zscore" in processed_df.columns
        assert len(signals.entries) == len(sample_ohlcv)

    def test_run_empty_df_raises(self) -> None:
        """빈 DataFrame에서 ValueError 발생."""
        strategy = MomMrBlendStrategy()
        empty_df = pd.DataFrame(
            columns=["close"],
        )
        empty_df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(empty_df)


class TestFromParams:
    """from_params() 팩토리 메서드 테스트."""

    def test_from_params(self) -> None:
        """파라미터 딕셔너리로 전략 생성."""
        strategy = MomMrBlendStrategy.from_params(
            mom_lookback=42,
            mr_lookback=21,
            vol_target=0.30,
        )
        assert isinstance(strategy, MomMrBlendStrategy)
        assert strategy.config.mom_lookback == 42
        assert strategy.config.mr_lookback == 21
        assert strategy.config.vol_target == 0.30

    def test_from_params_default(self) -> None:
        """파라미터 없이 기본값으로 생성."""
        strategy = MomMrBlendStrategy.from_params()
        assert strategy.config.mom_lookback == 28
        assert strategy.config.mr_lookback == 14

    def test_from_params_invalid_raises(self) -> None:
        """유효하지 않은 파라미터에서 ValidationError 발생."""
        with pytest.raises(ValidationError):
            MomMrBlendStrategy.from_params(mom_lookback=1)  # < min(5)


class TestRecommendedConfig:
    """recommended_config() 테스트."""

    def test_recommended_config(self) -> None:
        """권장 PM 설정이 올바른 키를 포함하는지 확인."""
        config = MomMrBlendStrategy.recommended_config()
        assert isinstance(config, dict)
        assert "max_leverage_cap" in config
        assert "system_stop_loss" in config
        assert "rebalance_threshold" in config

    def test_recommended_config_values(self) -> None:
        """권장 PM 설정값 검증."""
        config = MomMrBlendStrategy.recommended_config()
        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05


class TestWarmupPeriods:
    """warmup_periods() 테스트."""

    def test_warmup_periods(self) -> None:
        """전략의 warmup_periods가 config와 일치하는지 확인."""
        strategy = MomMrBlendStrategy()
        assert strategy.warmup_periods() == strategy.config.warmup_periods()

    def test_warmup_periods_default(self) -> None:
        """기본 설정의 warmup_periods."""
        strategy = MomMrBlendStrategy()
        expected = max(28 + 90, 14 + 90, 20) + 1
        assert strategy.warmup_periods() == expected

    def test_warmup_periods_conservative(self) -> None:
        """보수적 설정의 warmup_periods."""
        strategy = MomMrBlendStrategy.conservative()
        cfg = strategy.config
        expected = (
            max(
                cfg.mom_lookback + cfg.mom_z_window,
                cfg.mr_lookback + cfg.mr_z_window,
                cfg.vol_window,
            )
            + 1
        )
        assert strategy.warmup_periods() == expected


class TestTimeframeFactory:
    """for_timeframe() 팩토리 메서드 테스트."""

    def test_for_timeframe_1d(self) -> None:
        """1d 타임프레임 전략 생성."""
        strategy = MomMrBlendStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

    def test_for_timeframe_1h(self) -> None:
        """1h 타임프레임 전략 생성."""
        strategy = MomMrBlendStrategy.for_timeframe("1h")
        assert strategy.config.annualization_factor == 8760.0


class TestStartupInfo:
    """get_startup_info() 테스트."""

    def test_startup_info_keys(self) -> None:
        """시작 패널 정보에 핵심 키가 포함되는지 확인."""
        strategy = MomMrBlendStrategy()
        info = strategy.get_startup_info()
        assert "mom_lookback" in info
        assert "mr_lookback" in info
        assert "blend_weights" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_startup_info_values(self) -> None:
        """시작 패널 정보 값 검증."""
        strategy = MomMrBlendStrategy()
        info = strategy.get_startup_info()
        assert info["mom_lookback"] == "28d"
        assert info["mr_lookback"] == "14d"
        assert "Mom=50%" in info["blend_weights"]
        assert "MR=50%" in info["blend_weights"]
        assert info["mode"] == "Long-Only"
