"""Tests for MaxMinStrategy."""

import pandas as pd
import pytest
from pydantic import ValidationError

from src.strategy.max_min.config import MaxMinConfig
from src.strategy.max_min.strategy import MaxMinStrategy
from src.strategy.registry import get_strategy, list_strategies


class TestMaxMinStrategyRegistration:
    """전략 등록 테스트."""

    def test_strategy_registered(self) -> None:
        """'max-min'이 전략 레지스트리에 등록되어 있음."""
        assert "max-min" in list_strategies()

    def test_get_strategy(self) -> None:
        """get_strategy('max-min')로 MaxMinStrategy 클래스 반환."""
        strategy_cls = get_strategy("max-min")
        assert strategy_cls is MaxMinStrategy


class TestMaxMinStrategyProperties:
    """전략 속성 테스트."""

    def test_strategy_properties(self) -> None:
        """name, required_columns, config type 확인."""
        strategy = MaxMinStrategy()

        assert strategy.name == "MAX-MIN"
        assert strategy.required_columns == ["open", "high", "low", "close", "volume"]
        assert isinstance(strategy.config, MaxMinConfig)

    def test_warmup_periods(self) -> None:
        """warmup_periods > 0."""
        strategy = MaxMinStrategy()
        assert strategy.warmup_periods() > 0

        # config의 warmup_periods와 동일
        assert strategy.warmup_periods() == strategy.config.warmup_periods()

    def test_default_config(self) -> None:
        """config=None이면 기본 MaxMinConfig 사용."""
        strategy = MaxMinStrategy()
        config = strategy.config

        assert config.lookback == 10
        assert config.max_weight == 0.5
        assert config.min_weight == 0.5

    def test_custom_config(self) -> None:
        """커스텀 설정으로 생성."""
        config = MaxMinConfig(lookback=15, max_weight=0.7, min_weight=0.3)
        strategy = MaxMinStrategy(config)

        assert strategy.config.lookback == 15
        assert strategy.config.max_weight == 0.7
        assert strategy.config.min_weight == 0.3


class TestMaxMinStrategyExecution:
    """전략 실행 테스트."""

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """전체 파이프라인 실행 (preprocess -> generate_signals)."""
        strategy = MaxMinStrategy()
        processed = strategy.preprocess(sample_ohlcv)
        signals = strategy.generate_signals(processed)

        # 출력 타입 확인
        assert isinstance(processed, pd.DataFrame)
        assert isinstance(signals.entries, pd.Series)
        assert isinstance(signals.exits, pd.Series)
        assert isinstance(signals.direction, pd.Series)
        assert isinstance(signals.strength, pd.Series)

        # 길이 일치
        assert len(processed) == len(sample_ohlcv)
        assert len(signals.entries) == len(sample_ohlcv)

        # entries/exits are bool
        assert signals.entries.dtype == bool
        assert signals.exits.dtype == bool

    def test_from_params(self) -> None:
        """from_params로 전략 생성."""
        strategy = MaxMinStrategy.from_params(
            lookback=15,
            max_weight=0.6,
            min_weight=0.4,
            vol_target=0.25,
        )

        assert isinstance(strategy, MaxMinStrategy)
        assert strategy.config.lookback == 15
        assert strategy.config.max_weight == 0.6
        assert strategy.config.min_weight == 0.4
        assert strategy.config.vol_target == 0.25

    def test_from_params_invalid(self) -> None:
        """from_params에 유효하지 않은 파라미터면 에러."""
        with pytest.raises(ValidationError):
            MaxMinStrategy.from_params(max_weight=0.5, min_weight=0.3)  # sum != 1.0

    def test_recommended_config(self) -> None:
        """recommended_config()가 기대 키를 포함한 dict 반환."""
        rec = MaxMinStrategy.recommended_config()

        assert isinstance(rec, dict)
        assert "max_leverage_cap" in rec
        assert "system_stop_loss" in rec
        assert "rebalance_threshold" in rec
        assert rec["max_leverage_cap"] == 2.0

    def test_for_timeframe(self) -> None:
        """for_timeframe으로 타임프레임에 맞는 전략 생성."""
        strategy = MaxMinStrategy.for_timeframe("4h")

        assert isinstance(strategy, MaxMinStrategy)
        assert strategy.config.annualization_factor == 2190.0
        assert strategy.config.lookback == 12

    def test_for_timeframe_with_override(self) -> None:
        """for_timeframe에 kwargs 오버라이드."""
        strategy = MaxMinStrategy.for_timeframe("1d", vol_target=0.40)

        assert strategy.config.annualization_factor == 365.0
        assert strategy.config.vol_target == 0.40

    def test_conservative(self) -> None:
        """conservative() 팩토리 메서드."""
        strategy = MaxMinStrategy.conservative()
        assert isinstance(strategy, MaxMinStrategy)
        assert strategy.config.lookback == 20
        assert strategy.config.vol_target == 0.15

    def test_aggressive(self) -> None:
        """aggressive() 팩토리 메서드."""
        strategy = MaxMinStrategy.aggressive()
        assert isinstance(strategy, MaxMinStrategy)
        assert strategy.config.lookback == 5
        assert strategy.config.vol_target == 0.40

    def test_get_startup_info(self) -> None:
        """get_startup_info()가 핵심 파라미터 dict 반환."""
        strategy = MaxMinStrategy()
        info = strategy.get_startup_info()

        assert isinstance(info, dict)
        assert "lookback" in info
        assert "max_weight" in info
        assert "min_weight" in info
        assert "vol_target" in info
        assert "mode" in info
