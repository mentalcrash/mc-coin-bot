"""Unit tests for Beta Attribution module.

Rules Applied:
    - #17 Testing Standards: Pytest, parametrize
    - #12 Data Engineering: Vectorized test data
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.beta_attribution import (
    calculate_beta_attribution,
    calculate_hypothetical_returns,
    calculate_overall_beta,
    calculate_rolling_beta,
    calculate_rolling_beta_attribution,
    summarize_suppression_impact,
)
from src.models.backtest import BetaAttributionResult


class TestCalculateOverallBeta:
    """calculate_overall_beta 함수 테스트."""

    def test_perfect_correlation(self) -> None:
        """완벽한 상관관계 (Beta = 1) 테스트."""
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        strategy = benchmark.copy()  # 동일한 수익률

        beta = calculate_overall_beta(strategy, benchmark)
        assert beta == pytest.approx(1.0, abs=0.01)

    def test_double_exposure(self) -> None:
        """2배 노출 (Beta = 2) 테스트."""
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        strategy = benchmark * 2  # 2배 수익률

        beta = calculate_overall_beta(strategy, benchmark)
        assert beta == pytest.approx(2.0, abs=0.01)

    def test_zero_exposure(self) -> None:
        """무상관 (Beta ≈ 0) 테스트."""
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        strategy = pd.Series(np.random.randn(n) * 0.01)  # 독립적

        beta = calculate_overall_beta(strategy, benchmark)
        # 독립적이므로 0에 가까움
        assert abs(beta) < 0.5

    def test_empty_series(self) -> None:
        """빈 시리즈 테스트."""
        benchmark = pd.Series([], dtype=float)
        strategy = pd.Series([], dtype=float)

        beta = calculate_overall_beta(strategy, benchmark)
        assert beta == 0.0


class TestCalculateRollingBeta:
    """calculate_rolling_beta 함수 테스트."""

    def test_rolling_beta_length(self) -> None:
        """Rolling Beta 시리즈 길이 테스트."""
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        strategy = benchmark * 1.5

        rolling = calculate_rolling_beta(strategy, benchmark, window=20)

        assert len(rolling) == n
        # 초기 윈도우는 NaN (min_periods = window // 2 = 10이므로 9개 NaN)
        assert rolling.isna().sum() >= 9

    def test_rolling_beta_values(self) -> None:
        """Rolling Beta 값 범위 테스트."""
        np.random.seed(42)
        n = 100
        benchmark = pd.Series(np.random.randn(n) * 0.01)
        strategy = benchmark * 1.5

        rolling = calculate_rolling_beta(strategy, benchmark, window=20)
        valid_values = rolling.dropna()

        # 1.5 레버리지이므로 beta ≈ 1.5
        assert valid_values.mean() == pytest.approx(1.5, abs=0.1)


class TestCalculateHypotheticalReturns:
    """calculate_hypothetical_returns 함수 테스트."""

    def test_return_columns(self, sample_diagnostics_df: pd.DataFrame) -> None:
        """반환 DataFrame 컬럼 테스트."""
        # 벤치마크 수익률 생성
        n = len(sample_diagnostics_df)
        benchmark_returns = pd.Series(
            np.random.randn(n) * 0.01,
            index=sample_diagnostics_df.index,
        )

        returns_df = calculate_hypothetical_returns(
            sample_diagnostics_df, benchmark_returns
        )

        expected_columns = [
            "potential_return",
            "return_after_trend",
            "return_after_deadband",
            "actual_return",
            "benchmark_return",
        ]
        for col in expected_columns:
            assert col in returns_df.columns


class TestCalculateBetaAttribution:
    """calculate_beta_attribution 함수 테스트."""

    def test_result_type(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """반환 타입 테스트."""
        result = calculate_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        assert isinstance(result, BetaAttributionResult)

    def test_beta_values_consistency(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Beta 값 일관성 테스트."""
        result = calculate_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        # potential_beta >= realized_beta (필터로 beta가 감소)
        # NOTE: 실제로는 vol scaling으로 증가할 수도 있음
        # 여기서는 값이 유효한지만 확인
        assert not np.isnan(result.potential_beta)
        assert not np.isnan(result.realized_beta)

    def test_total_beta_loss(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """총 Beta 손실 계산 테스트."""
        result = calculate_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        # total_beta_loss = potential - realized
        expected_loss = result.potential_beta - result.realized_beta
        assert result.total_beta_loss == pytest.approx(expected_loss, abs=0.001)

    def test_beta_retention_ratio(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Beta 보존 비율 테스트."""
        result = calculate_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        if result.potential_beta != 0:
            expected_ratio = result.realized_beta / result.potential_beta
            assert result.beta_retention_ratio == pytest.approx(
                expected_ratio, abs=0.001
            )
        else:
            assert result.beta_retention_ratio == 0.0

    def test_summary_method(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """summary 메서드 테스트."""
        result = calculate_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        summary = result.summary()

        assert "potential_beta" in summary
        assert "realized_beta" in summary
        assert "beta_retention" in summary


class TestCalculateRollingBetaAttribution:
    """calculate_rolling_beta_attribution 함수 테스트."""

    def test_return_columns(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """반환 DataFrame 컬럼 테스트."""
        rolling_df = calculate_rolling_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        expected_columns = [
            "potential_beta",
            "beta_after_trend_filter",
            "beta_after_deadband",
            "realized_beta",
            "lost_to_trend_filter",
            "lost_to_deadband",
            "lost_to_vol_scaling",
        ]
        for col in expected_columns:
            assert col in rolling_df.columns

    def test_rolling_length(
        self,
        sample_diagnostics_df: pd.DataFrame,
        sample_benchmark_returns: pd.Series,  # type: ignore[type-arg]
    ) -> None:
        """Rolling DataFrame 길이 테스트."""
        rolling_df = calculate_rolling_beta_attribution(
            sample_diagnostics_df,
            sample_benchmark_returns,
            window=20,
        )

        # 공통 인덱스 길이
        common_idx = sample_diagnostics_df.index.intersection(
            sample_benchmark_returns.index
        )
        assert len(rolling_df) <= len(common_idx)


class TestSummarizeSuppressionImpact:
    """summarize_suppression_impact 함수 테스트."""

    def test_empty_dataframe(self) -> None:
        """빈 DataFrame 테스트."""
        empty_df = pd.DataFrame()
        result = summarize_suppression_impact(empty_df)
        assert result == {}

    def test_with_sample_data(self, sample_diagnostics_df: pd.DataFrame) -> None:
        """샘플 데이터 테스트."""
        result = summarize_suppression_impact(sample_diagnostics_df)

        # 결과가 dict 형태
        assert isinstance(result, dict)

        # 각 원인별 통계 확인
        for stats in result.values():
            assert "count" in stats
            assert "percentage" in stats
            assert "avg_potential_weight" in stats
            assert stats["count"] >= 0
            assert 0 <= stats["percentage"] <= 100
