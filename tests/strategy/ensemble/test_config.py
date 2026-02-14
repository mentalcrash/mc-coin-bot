"""Unit tests for Ensemble Strategy config."""

import pytest
from pydantic import ValidationError

from src.strategy.ensemble.config import (
    AggregationMethod,
    EnsembleConfig,
    ShortMode,
    SubStrategySpec,
)


def _two_specs() -> tuple[SubStrategySpec, ...]:
    """최소 2개 SubStrategySpec 반환."""
    return (
        SubStrategySpec(name="tsmom", params={"lookback": 30}),
        SubStrategySpec(name="donchian-ensemble", params={"lookbacks": [20, 60]}),
    )


class TestSubStrategySpec:
    """SubStrategySpec 검증."""

    def test_default_weight(self) -> None:
        spec = SubStrategySpec(name="tsmom")
        assert spec.weight == 1.0
        assert spec.params == {}

    def test_custom_params(self) -> None:
        spec = SubStrategySpec(name="tsmom", params={"lookback": 30}, weight=2.0)
        assert spec.params["lookback"] == 30
        assert spec.weight == 2.0

    def test_frozen(self) -> None:
        spec = SubStrategySpec(name="tsmom")
        with pytest.raises(ValidationError):
            spec.name = "other"  # type: ignore[misc]

    def test_weight_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="greater than 0"):
            SubStrategySpec(name="tsmom", weight=0.0)


class TestEnsembleConfig:
    """EnsembleConfig 검증."""

    def test_defaults(self) -> None:
        cfg = EnsembleConfig(strategies=_two_specs())
        assert cfg.aggregation == AggregationMethod.EQUAL_WEIGHT
        assert cfg.vol_target == 0.35
        assert cfg.short_mode == ShortMode.DISABLED
        assert cfg.vol_lookback == 63
        assert cfg.momentum_lookback == 126
        assert cfg.top_n == 3
        assert cfg.min_agreement == 0.5

    def test_frozen(self) -> None:
        cfg = EnsembleConfig(strategies=_two_specs())
        with pytest.raises(ValidationError):
            cfg.vol_target = 0.5  # type: ignore[misc]

    def test_min_two_strategies(self) -> None:
        with pytest.raises(ValidationError, match="too_short"):
            EnsembleConfig(strategies=(SubStrategySpec(name="tsmom"),))

    def test_top_n_exceeds_strategies(self) -> None:
        with pytest.raises(ValidationError, match="top_n"):
            EnsembleConfig(
                strategies=_two_specs(),
                aggregation=AggregationMethod.STRATEGY_MOMENTUM,
                top_n=5,
            )

    def test_custom_aggregation(self) -> None:
        cfg = EnsembleConfig(
            strategies=_two_specs(),
            aggregation=AggregationMethod.INVERSE_VOLATILITY,
            vol_lookback=30,
        )
        assert cfg.aggregation == AggregationMethod.INVERSE_VOLATILITY
        assert cfg.vol_lookback == 30

    def test_warmup_equal_weight(self) -> None:
        cfg = EnsembleConfig(strategies=_two_specs())
        assert cfg.warmup_periods() == cfg.vol_window + 1

    def test_warmup_inverse_vol(self) -> None:
        cfg = EnsembleConfig(
            strategies=_two_specs(),
            aggregation=AggregationMethod.INVERSE_VOLATILITY,
        )
        assert cfg.warmup_periods() == cfg.vol_lookback + cfg.vol_window + 1

    def test_warmup_strategy_momentum(self) -> None:
        cfg = EnsembleConfig(
            strategies=_two_specs(),
            aggregation=AggregationMethod.STRATEGY_MOMENTUM,
            top_n=2,
        )
        assert cfg.warmup_periods() == cfg.momentum_lookback + cfg.vol_window + 1
