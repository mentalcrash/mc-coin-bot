"""Tests for MtfMacdStrategy (Integration)."""

import pandas as pd
import pytest

from src.strategy import get_strategy, list_strategies
from src.strategy.mtf_macd import MtfMacdConfig, MtfMacdStrategy


class TestRegistry:
    """전략 Registry 통합 테스트."""

    def test_strategy_registered(self) -> None:
        """'mtf-macd'가 Registry에 등록됨."""
        available = list_strategies()
        assert "mtf-macd" in available

    def test_get_strategy(self) -> None:
        """get_strategy로 올바른 클래스 조회."""
        strategy_class = get_strategy("mtf-macd")
        assert strategy_class == MtfMacdStrategy


class TestMtfMacdStrategy:
    """MtfMacdStrategy 클래스 테스트."""

    def test_strategy_properties(self) -> None:
        """기본 속성 확인."""
        strategy = MtfMacdStrategy()

        assert strategy.name == "MTF-MACD"
        assert set(strategy.required_columns) == {"open", "high", "low", "close"}
        assert isinstance(strategy.config, MtfMacdConfig)

    def test_run_pipeline(self, sample_ohlcv: pd.DataFrame) -> None:
        """run() end-to-end 파이프라인."""
        strategy = MtfMacdStrategy()
        processed_df, signals = strategy.run(sample_ohlcv)

        # 전처리 컬럼 확인
        assert "macd_line" in processed_df.columns
        assert "signal_line" in processed_df.columns
        assert "macd_histogram" in processed_df.columns
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
        strategy = MtfMacdStrategy.from_params(
            fast_period=8,
            slow_period=17,
            signal_period=5,
            vol_target=0.30,
        )

        assert strategy.config.fast_period == 8
        assert strategy.config.slow_period == 17
        assert strategy.config.signal_period == 5
        assert strategy.config.vol_target == 0.30

        # 실행 가능 확인
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_recommended_config(self) -> None:
        """recommended_config 값 확인."""
        config = MtfMacdStrategy.recommended_config()

        assert config["max_leverage_cap"] == 2.0
        assert config["system_stop_loss"] == 0.10
        assert config["rebalance_threshold"] == 0.05

    def test_warmup_periods(self) -> None:
        """warmup_periods 반환값이 config와 일치."""
        strategy = MtfMacdStrategy()
        warmup = strategy.warmup_periods()
        assert warmup > 0
        assert warmup == strategy.config.warmup_periods()
        # 기본값: 26 + 9 + 1 = 36
        assert warmup == 36

    def test_for_timeframe(self) -> None:
        """for_timeframe() 팩토리."""
        strategy = MtfMacdStrategy.for_timeframe("1d")
        assert strategy.config.annualization_factor == 365.0

        strategy_1h = MtfMacdStrategy.for_timeframe("1h")
        assert strategy_1h.config.annualization_factor == 8760.0

    def test_for_timeframe_with_overrides(self) -> None:
        """for_timeframe()에 추가 파라미터 전달."""
        strategy = MtfMacdStrategy.for_timeframe("1d", vol_target=0.35)
        assert strategy.config.annualization_factor == 365.0
        assert strategy.config.vol_target == 0.35

    def test_conservative_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """conservative() 팩토리 동작."""
        strategy = MtfMacdStrategy.conservative()
        assert strategy.config.fast_period == 12
        assert strategy.config.slow_period == 26
        assert strategy.config.vol_target == 0.30
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_aggressive_preset(self, sample_ohlcv: pd.DataFrame) -> None:
        """aggressive() 팩토리 동작."""
        strategy = MtfMacdStrategy.aggressive()
        assert strategy.config.fast_period == 8
        assert strategy.config.slow_period == 17
        assert strategy.config.vol_target == 0.50
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)

    def test_get_startup_info(self) -> None:
        """get_startup_info 키 확인."""
        strategy = MtfMacdStrategy()
        info = strategy.get_startup_info()

        assert "macd" in info
        assert "vol_target" in info
        assert "mode" in info

    def test_validate_input_missing_columns(self) -> None:
        """필수 컬럼 누락 시 에러."""
        strategy = MtfMacdStrategy()
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0]},
            index=pd.date_range("2024-01-01", periods=3, freq="1D"),
        )
        with pytest.raises(ValueError):
            strategy.run(df)

    def test_validate_input_empty_df(self) -> None:
        """빈 DataFrame 시 에러."""
        strategy = MtfMacdStrategy()
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
        )
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            strategy.run(df)

    def test_default_config(self) -> None:
        """기본 설정으로 생성된 전략의 config 확인."""
        strategy = MtfMacdStrategy()
        assert strategy.config.fast_period == 12
        assert strategy.config.slow_period == 26
        assert strategy.config.signal_period == 9

    def test_custom_config(self) -> None:
        """커스텀 설정으로 전략 생성."""
        config = MtfMacdConfig(fast_period=8, slow_period=17, vol_target=0.30)
        strategy = MtfMacdStrategy(config)
        assert strategy.config.fast_period == 8
        assert strategy.config.vol_target == 0.30

    def test_params_dict(self) -> None:
        """params 프로퍼티가 config의 model_dump와 일치."""
        strategy = MtfMacdStrategy()
        params = strategy.params
        assert params["fast_period"] == 12
        assert params["slow_period"] == 26
        assert params["signal_period"] == 9

    def test_repr(self) -> None:
        """__repr__ 문자열 확인."""
        strategy = MtfMacdStrategy()
        repr_str = repr(strategy)
        assert "MtfMacdStrategy" in repr_str
