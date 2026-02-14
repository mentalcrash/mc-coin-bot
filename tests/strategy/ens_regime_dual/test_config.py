"""Tests for Regime-Adaptive Dual-Alpha Ensemble config."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.strategy.ens_regime_dual.config import EnsRegimeDualConfig
from src.strategy.ensemble.config import AggregationMethod, ShortMode


class TestEnsRegimeDualConfig:
    def test_default_values(self) -> None:
        config = EnsRegimeDualConfig()
        assert config.aggregation == AggregationMethod.INVERSE_VOLATILITY
        assert config.vol_lookback == 63
        assert config.vol_target == 0.35
        assert config.annualization_factor == 365.0
        assert config.short_mode == ShortMode.FULL
        assert config.ctrend_weight == 1.0
        assert config.regime_mf_mr_weight == 1.0

    def test_frozen(self) -> None:
        config = EnsRegimeDualConfig()
        with pytest.raises(ValidationError):
            config.vol_target = 0.5  # type: ignore[misc]

    def test_aggregation_methods(self) -> None:
        for method in AggregationMethod:
            if method == AggregationMethod.STRATEGY_MOMENTUM:
                config = EnsRegimeDualConfig(aggregation=method, top_n=2)
            else:
                config = EnsRegimeDualConfig(aggregation=method)
            assert config.aggregation == method

    def test_strategy_momentum_top_n_validation(self) -> None:
        """top_n > 2 (number of sub-strategies) should fail for strategy_momentum."""
        with pytest.raises(ValidationError):
            EnsRegimeDualConfig(
                aggregation=AggregationMethod.STRATEGY_MOMENTUM,
                top_n=3,
            )

    def test_warmup_periods(self) -> None:
        config = EnsRegimeDualConfig()
        assert config.warmup_periods() > 0

    def test_warmup_with_inverse_vol(self) -> None:
        config = EnsRegimeDualConfig(
            aggregation=AggregationMethod.INVERSE_VOLATILITY,
            vol_lookback=63,
        )
        assert config.warmup_periods() >= 63

    def test_custom_weights(self) -> None:
        config = EnsRegimeDualConfig(ctrend_weight=2.0, regime_mf_mr_weight=0.5)
        assert config.ctrend_weight == 2.0
        assert config.regime_mf_mr_weight == 0.5

    def test_short_mode(self) -> None:
        config = EnsRegimeDualConfig(short_mode=ShortMode.DISABLED)
        assert config.short_mode == ShortMode.DISABLED
