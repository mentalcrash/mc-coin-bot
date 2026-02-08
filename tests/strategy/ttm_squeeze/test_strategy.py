"""Tests for TtmSqueezeStrategy (Integration)."""

import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.ttm_squeeze import TtmSqueezeConfig, TtmSqueezeStrategy


class TestRegistry:
    """전략 Registry 통합 테스트."""

    def test_strategy_registered(self) -> None:
        """'ttm-squeeze'가 Registry에 등록됨."""
        available = list_strategies()
        assert "ttm-squeeze" in available

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스 조회."""
        strategy_class = get_strategy("ttm-squeeze")
        assert strategy_class == TtmSqueezeStrategy


class TestTtmSqueezeStrategy:
    """TtmSqueezeStrategy 클래스 테스트."""

    def test_strategy_properties(self) -> None:
        """기본 속성 확인."""
        strategy = TtmSqueezeStrategy()

        assert strategy.name == "TTM-Squeeze"
        assert set(strategy.required_columns) == {"open", "high", "low", "close"}
        assert isinstance(strategy.config, TtmSqueezeConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run() end-to-end 파이프라인."""
        strategy = TtmSqueezeStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "bb_upper" in processed_df.columns
        assert "bb_lower" in processed_df.columns
        assert "kc_upper" in processed_df.columns
        assert "kc_lower" in processed_df.columns
        assert "squeeze_on" in processed_df.columns
        assert "momentum" in processed_df.columns
        assert "exit_sma" in processed_df.columns
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
        strategy = TtmSqueezeStrategy.from_params(
            bb_period=15,
            kc_mult=2.0,
            vol_target=0.30,
        )

        assert strategy.config.bb_period == 15
        assert strategy.config.kc_mult == 2.0
        assert strategy.config.vol_target == 0.30

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self) -> None:
        """recommended_config 값 확인."""
        config = TtmSqueezeStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05

    def test_warmup_periods(self) -> None:
        """warmup_periods 반환값이 config와 일치."""
        strategy = TtmSqueezeStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()
        # 기본값: max(20, 20, 20, 21, 20) + 1 = 22
        assert warmup == 22

    def test_for_timeframe(self) -> None:
        """for_timeframe() 팩토리."""
        strategy = TtmSqueezeStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

        strategy_4h = TtmSqueezeStrategy.for_timeframe("4h")
        assert strategy_4h.config.annualization_factor == 2190.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe()에 추가 파라미터 전달."""
        strategy = TtmSqueezeStrategy.for_timeframe("4h", vol_target=0.35)
        assert strategy.config.annualization_factor == 2190.0
        assert strategy.config.vol_target == 0.35

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """conservative() 팩토리 동작."""
        strategy = TtmSqueezeStrategy.conservative()
        assert strategy.config.bb_std == 2.0
        assert strategy.config.kc_mult == 2.0
        assert strategy.config.vol_target == 0.30
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """aggressive() 팩토리 동작."""
        strategy = TtmSqueezeStrategy.aggressive()
        assert strategy.config.bb_std == 1.5
        assert strategy.config.kc_mult == 1.0
        assert strategy.config.vol_target == 0.50
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_get_startup_info(self) -> None:
        """get_startup_info 키 확인."""
        strategy = TtmSqueezeStrategy()
        info = strategy.get_startup_info()

        assert "bb" in info
        assert "kc" in info
        assert "momentum" in info
        assert "exit_sma" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_validate_input_missing_columns(self) -> None:
        """필수 컬럼 누락 시 에러."""
        strategy = TtmSqueezeStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_validate_input_empty_df(self) -> None:
        """빈 DataFrame 시 에러."""
        strategy = TtmSqueezeStrategy()
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
        )
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(df)

    def test_default_config(self) -> None:
        """기본 설정으로 생성된 전략의 config 확인."""
        strategy = TtmSqueezeStrategy()
        assert strategy.config.bb_period == 20
        assert strategy.config.bb_std == 2.0
        assert strategy.config.kc_mult == 1.5

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = TtmSqueezeConfig(bb_period=15, kc_mult=2.0)
        strategy = TtmSqueezeStrategy(config)
        assert strategy.config.bb_period == 15
        assert strategy.config.kc_mult == 2.0

    def test_params_dict(self) -> None:
        """params 프로퍼티가 config의 model_dump와 일치."""
        strategy = TtmSqueezeStrategy()
        params = strategy.params
        assert params["bb_period"] == 20
        assert params["bb_std"] == 2.0
        assert params["kc_mult"] == 1.5
        assert params["mom_period"] == 20

    def test_repr(self) -> None:
        """__repr__ 문자열 확인."""
        strategy = TtmSqueezeStrategy()
        repr_str = repr(strategy)
        assert "TtmSqueezeStrategy" in repr_str
