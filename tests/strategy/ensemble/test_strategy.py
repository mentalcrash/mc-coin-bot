"""Unit tests for EnsembleStrategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.strategy.ensemble.config import (
    AggregationMethod,
    EnsembleConfig,
    ShortMode,
    SubStrategySpec,
)
from src.strategy.ensemble.strategy import EnsembleStrategy
from src.strategy.registry import get_strategy, is_registered

if TYPE_CHECKING:
    import pandas as pd


class TestEnsembleStrategyRegistration:
    """Registry 등록 확인."""

    def test_registered_as_ensemble(self) -> None:
        assert is_registered("ensemble")

    def test_get_strategy_returns_class(self) -> None:
        cls = get_strategy("ensemble")
        assert cls is EnsembleStrategy


class TestEnsembleStrategyInit:
    """초기화 검증."""

    def test_requires_config(self) -> None:
        with pytest.raises(ValueError, match="EnsembleConfig is required"):
            EnsembleStrategy(config=None)

    def test_creates_sub_strategies(self) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
        )
        strategy = EnsembleStrategy(config)
        assert len(strategy._sub_strategies) == 2
        assert strategy._strategy_names == ["tsmom", "donchian-ensemble"]

    def test_properties(self) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
        )
        strategy = EnsembleStrategy(config)
        assert strategy.name == "Ensemble"
        assert "close" in strategy.required_columns
        assert strategy.config is config


class TestEnsembleFromParams:
    """from_params 변환 검증."""

    def test_from_params_basic(self) -> None:
        strategy = EnsembleStrategy.from_params(
            strategies=[
                {"name": "tsmom", "params": {"lookback": 30}},
                {"name": "donchian-ensemble"},
            ],
            aggregation="equal_weight",
        )
        assert isinstance(strategy, EnsembleStrategy)
        assert len(strategy._sub_strategies) == 2

    def test_from_params_with_aggregation(self) -> None:
        strategy = EnsembleStrategy.from_params(
            strategies=[
                {"name": "tsmom"},
                {"name": "donchian-ensemble"},
            ],
            aggregation="inverse_volatility",
            vol_lookback=30,
        )
        assert strategy.config.aggregation == AggregationMethod.INVERSE_VOLATILITY
        assert strategy.config.vol_lookback == 30

    def test_from_params_with_weights(self) -> None:
        strategy = EnsembleStrategy.from_params(
            strategies=[
                {"name": "tsmom", "weight": 2.0},
                {"name": "donchian-ensemble", "weight": 1.0},
            ],
        )
        assert strategy._weights["tsmom"] == 2.0
        assert strategy._weights["donchian-ensemble"] == 1.0


class TestEnsembleRun:
    """run() 통합 테스트."""

    def test_run_produces_signals(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
        )
        strategy = EnsembleStrategy(config)
        processed_df, signals = strategy.run(sample_ohlcv)

        assert "vol_scalar" in processed_df.columns
        assert len(signals.direction) == len(sample_ohlcv)
        assert len(signals.strength) == len(sample_ohlcv)
        assert set(signals.direction.unique()).issubset({-1, 0, 1})

    def test_run_with_inverse_vol(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
            aggregation=AggregationMethod.INVERSE_VOLATILITY,
            vol_lookback=20,
        )
        strategy = EnsembleStrategy(config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)

    def test_run_with_majority_vote(self, sample_ohlcv: pd.DataFrame) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
            aggregation=AggregationMethod.MAJORITY_VOTE,
        )
        strategy = EnsembleStrategy(config)
        _, signals = strategy.run(sample_ohlcv)
        assert len(signals.direction) == len(sample_ohlcv)


class TestEnsembleStartupInfo:
    """get_startup_info 검증."""

    def test_startup_info(self) -> None:
        config = EnsembleConfig(
            strategies=(
                SubStrategySpec(name="tsmom"),
                SubStrategySpec(name="donchian-ensemble"),
            ),
            aggregation=AggregationMethod.EQUAL_WEIGHT,
            short_mode=ShortMode.DISABLED,
        )
        strategy = EnsembleStrategy(config)
        info = strategy.get_startup_info()

        assert info["num_strategies"] == "2"
        assert info["aggregation"] == "equal_weight"
        assert info["mode"] == "Long-Only"
        assert "tsmom" in info["strategies"]
