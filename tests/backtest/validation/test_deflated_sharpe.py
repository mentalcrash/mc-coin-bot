"""Tests for Deflated Sharpe Ratio."""

from src.backtest.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)


class TestExpectedMaxSharpe:
    """Tests for expected_max_sharpe()."""

    def test_single_trial_returns_zero(self) -> None:
        """Single trial should return 0."""
        assert expected_max_sharpe(1) == 0.0

    def test_more_trials_higher_expected_max(self) -> None:
        """More trials should give higher expected max Sharpe."""
        e10 = expected_max_sharpe(10)
        e100 = expected_max_sharpe(100)
        e1000 = expected_max_sharpe(1000)

        assert e10 > 0
        assert e100 > e10
        assert e1000 > e100

    def test_non_normal_correction(self) -> None:
        """Non-zero skewness/kurtosis should change the result."""
        normal = expected_max_sharpe(100)
        skewed = expected_max_sharpe(100, skewness=-0.5)

        assert normal != skewed

    def test_reasonable_range(self) -> None:
        """Expected max Sharpe for 100 trials should be in reasonable range."""
        e_max = expected_max_sharpe(100)
        assert 1.0 < e_max < 4.0


class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio()."""

    def test_high_sharpe_low_trials_passes(self) -> None:
        """High Sharpe with few trials should get high DSR."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=5,
            n_observations=252,
        )
        assert dsr > 0.5

    def test_low_sharpe_high_trials_fails(self) -> None:
        """Low Sharpe with many trials should get low DSR."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=0.5,
            n_trials=1000,
            n_observations=252,
        )
        assert dsr < 0.5

    def test_more_trials_lower_dsr(self) -> None:
        """More trials should decrease DSR for same observed Sharpe."""
        dsr_low = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=10,
            n_observations=252,
        )
        dsr_high = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=1000,
            n_observations=252,
        )
        assert dsr_low > dsr_high

    def test_edge_case_zero_trials(self) -> None:
        """Zero or negative trials should return 0."""
        assert deflated_sharpe_ratio(1.0, n_trials=0, n_observations=252) == 0.0

    def test_edge_case_few_observations(self) -> None:
        """Very few observations should return 0."""
        assert deflated_sharpe_ratio(1.0, n_trials=10, n_observations=1) == 0.0

    def test_returns_between_zero_and_one(self) -> None:
        """DSR should be a probability between 0 and 1."""
        dsr = deflated_sharpe_ratio(
            observed_sharpe=1.0,
            n_trials=50,
            n_observations=252,
        )
        assert 0.0 <= dsr <= 1.0


class TestProbabilisticSharpeRatio:
    """Tests for probabilistic_sharpe_ratio()."""

    def test_sharpe_above_benchmark(self) -> None:
        """Sharpe above benchmark should give PSR > 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_observations=252,
        )
        assert psr > 0.5

    def test_sharpe_below_benchmark(self) -> None:
        """Sharpe below benchmark should give PSR < 0.5."""
        psr = probabilistic_sharpe_ratio(
            observed_sharpe=0.0,
            benchmark_sharpe=1.0,
            n_observations=252,
        )
        assert psr < 0.5

    def test_more_observations_more_precise(self) -> None:
        """More observations should push PSR further from 0.5."""
        psr_few = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_observations=30,
        )
        psr_many = probabilistic_sharpe_ratio(
            observed_sharpe=1.0,
            benchmark_sharpe=0.0,
            n_observations=1000,
        )
        # Both above 0.5, but more data should be more confident
        assert psr_many > psr_few

    def test_few_observations_returns_zero(self) -> None:
        """Very few observations should return 0."""
        assert probabilistic_sharpe_ratio(1.0, 0.0, n_observations=1) == 0.0
