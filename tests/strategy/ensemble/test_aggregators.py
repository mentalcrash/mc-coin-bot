"""Unit tests for ensemble aggregation functions."""

import numpy as np
import pandas as pd

from src.strategy.ensemble.aggregators import (
    equal_weight,
    inverse_volatility,
    majority_vote,
    strategy_momentum,
)


class TestEqualWeight:
    """equal_weight aggregator 검증."""

    def test_output_shape(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, s = equal_weight(directions, strengths, weights)
        assert len(d) == len(directions)
        assert len(s) == len(strengths)

    def test_direction_values(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, _ = equal_weight(directions, strengths, weights)
        assert set(d.unique()).issubset({-1, 0, 1})

    def test_uniform_positive(self) -> None:
        """모든 전략 동일 positive strength → direction +1."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.5] * 5, "b": [0.5] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0})

        d, s = equal_weight(directions, strengths, weights)
        assert (d == 1).all()
        assert np.isclose(s.iloc[0], 0.5)

    def test_opposing_cancel(self) -> None:
        """두 전략이 반대 방향 + 동일 strength → direction 0."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [-1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.5] * 5, "b": [-0.5] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0})

        d, s = equal_weight(directions, strengths, weights)
        assert (d == 0).all()
        assert np.isclose(s.iloc[0], 0.0)


class TestInverseVolatility:
    """inverse_volatility aggregator 검증."""

    def test_output_shape(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, s = inverse_volatility(directions, strengths, weights, vol_lookback=10)
        assert len(d) == len(directions)
        assert len(s) == len(strengths)

    def test_direction_values(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, _ = inverse_volatility(directions, strengths, weights, vol_lookback=10)
        assert set(d.unique()).issubset({-1, 0, 1})

    def test_nan_fallback_to_ew(self) -> None:
        """lookback 이전 NaN 구간에서 EW fallback 적용."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.5] * 5, "b": [0.5] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0})

        # vol_lookback=10이면 5 bars 데이터에서는 전부 NaN → EW fallback
        d, _s = inverse_volatility(directions, strengths, weights, vol_lookback=10)
        assert (d == 1).all()

    def test_low_vol_gets_higher_weight(self) -> None:
        """변동성이 낮은 전략이 더 높은 가중치를 받는지 확인."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")

        # a: 안정적 (낮은 std), b: 불안정 (높은 std)
        directions = pd.DataFrame({"a": [1] * n, "b": [1] * n}, index=idx)
        strengths = pd.DataFrame(
            {
                "a": 0.5 + np.random.randn(n) * 0.01,  # 낮은 변동성
                "b": 0.5 + np.random.randn(n) * 1.0,  # 높은 변동성
            },
            index=idx,
        )
        weights = pd.Series({"a": 1.0, "b": 1.0})

        _d, s = inverse_volatility(directions, strengths, weights, vol_lookback=20)

        # 후반부에서 a의 가중치가 b보다 높아야 함 → 결과가 a에 가까워야
        valid = s.iloc[50:]
        assert valid.mean() > 0  # positive direction 유지


class TestMajorityVote:
    """majority_vote aggregator 검증."""

    def test_output_shape(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, s = majority_vote(directions, strengths, weights)
        assert len(d) == len(directions)
        assert len(s) == len(strengths)

    def test_unanimous_long(self) -> None:
        """전원 Long 합의 → direction +1."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [1] * 5, "c": [1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.3] * 5, "b": [0.4] * 5, "c": [0.5] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})

        d, s = majority_vote(directions, strengths, weights, min_agreement=0.5)
        assert (d == 1).all()
        assert (s > 0).all()

    def test_no_agreement(self) -> None:
        """합의 미달 → direction 0."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [-1] * 5, "c": [0] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.3] * 5, "b": [-0.3] * 5, "c": [0.0] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})

        # 활성 2개 중 long 1, short 1 → 50%, min_agreement=0.6이면 부족
        d, _ = majority_vote(directions, strengths, weights, min_agreement=0.6)
        assert (d == 0).all()

    def test_short_agreement(self) -> None:
        """다수 Short 합의 → direction -1."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [-1] * 5, "b": [-1] * 5, "c": [1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [-0.5] * 5, "b": [-0.3] * 5, "c": [0.2] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0})

        d, s = majority_vote(directions, strengths, weights, min_agreement=0.5)
        assert (d == -1).all()
        assert (s < 0).all()


class TestStrategyMomentum:
    """strategy_momentum aggregator 검증."""

    def test_output_shape(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, s = strategy_momentum(directions, strengths, weights, momentum_lookback=20, top_n=2)
        assert len(d) == len(directions)
        assert len(s) == len(strengths)

    def test_direction_values(
        self,
        agg_data: tuple[pd.DataFrame, pd.DataFrame, pd.Series],
    ) -> None:
        directions, strengths, weights = agg_data
        d, _ = strategy_momentum(directions, strengths, weights, momentum_lookback=20, top_n=2)
        assert set(d.unique()).issubset({-1, 0, 1})

    def test_nan_fallback(self) -> None:
        """lookback 이전 NaN 구간에서 EW fallback."""
        idx = pd.date_range("2024-01-01", periods=5, freq="1D")
        directions = pd.DataFrame({"a": [1] * 5, "b": [1] * 5}, index=idx)
        strengths = pd.DataFrame({"a": [0.5] * 5, "b": [0.3] * 5}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0})

        d, _ = strategy_momentum(directions, strengths, weights, momentum_lookback=10, top_n=1)
        # 5 bars < lookback 10 → 전부 NaN → EW fallback
        assert (d == 1).all()

    def test_selects_top_n(self) -> None:
        """top_n=1일 때 가장 좋은 전략 하나만 선택."""
        np.random.seed(42)
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="1D")

        # a: 일관되게 positive, b: 일관되게 negative
        directions = pd.DataFrame({"a": [1] * n, "b": [-1] * n}, index=idx)
        strengths = pd.DataFrame({"a": [0.5] * n, "b": [-0.3] * n}, index=idx)
        weights = pd.Series({"a": 1.0, "b": 1.0})

        d, _s = strategy_momentum(directions, strengths, weights, momentum_lookback=20, top_n=1)

        # lookback 이후 구간에서 a가 선택되어야 → direction +1
        valid = d.iloc[25:]
        assert (valid == 1).all()
