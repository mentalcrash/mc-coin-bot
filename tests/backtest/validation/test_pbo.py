"""Tests for Probability of Backtest Overfitting."""

import pytest

from src.backtest.validation.pbo import calculate_pbo, calculate_pbo_logit


class TestCalculatePBO:
    """Tests for calculate_pbo()."""

    def test_consistent_performance_low_pbo(self) -> None:
        """When IS and OOS rank similarly, PBO should be low."""
        # IS and OOS both increase together
        is_sharpes = [0.5, 1.0, 1.5, 2.0, 2.5]
        oos_sharpes = [0.4, 0.9, 1.3, 1.8, 2.2]

        pbo = calculate_pbo(is_sharpes, oos_sharpes)
        assert pbo < 0.5

    def test_overfit_performance_high_pbo(self) -> None:
        """When IS best is OOS worst, PBO should be high."""
        # IS: fold 4 is best; OOS: fold 4 is worst
        is_sharpes = [0.5, 1.0, 1.5, 2.0, 3.0]
        oos_sharpes = [2.5, 2.0, 1.5, 1.0, 0.1]

        pbo = calculate_pbo(is_sharpes, oos_sharpes)
        assert pbo > 0.3

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched IS/OOS lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            calculate_pbo([1.0, 2.0], [1.0])

    def test_too_few_folds_raises(self) -> None:
        """Less than 2 folds should raise."""
        with pytest.raises(ValueError, match="at least"):
            calculate_pbo([1.0], [1.0])

    def test_returns_between_zero_and_one(self) -> None:
        """PBO should be between 0 and 1."""
        is_sharpes = [1.0, 2.0, 3.0, 4.0]
        oos_sharpes = [4.0, 3.0, 2.0, 1.0]

        pbo = calculate_pbo(is_sharpes, oos_sharpes)
        assert 0.0 <= pbo <= 1.0


class TestCalculatePBOLogit:
    """Tests for calculate_pbo_logit()."""

    def test_is_best_also_oos_best(self) -> None:
        """If IS best is also OOS best, PBO logit should be low."""
        is_sharpes = [0.5, 1.0, 1.5, 2.0, 3.0]
        oos_sharpes = [0.4, 0.9, 1.3, 1.8, 2.8]

        pbo = calculate_pbo_logit(is_sharpes, oos_sharpes)
        assert pbo < 0.5

    def test_is_best_is_oos_worst(self) -> None:
        """If IS best is OOS worst, PBO logit should be high."""
        # IS best is index 4 (3.0), OOS index 4 is worst (0.1)
        is_sharpes = [0.5, 1.0, 1.5, 2.0, 3.0]
        oos_sharpes = [2.5, 2.0, 1.5, 1.0, 0.1]

        pbo = calculate_pbo_logit(is_sharpes, oos_sharpes)
        assert pbo > 0.5

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched IS/OOS lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            calculate_pbo_logit([1.0, 2.0], [1.0])

    def test_returns_between_zero_and_one(self) -> None:
        """PBO logit should be between 0 and 1."""
        is_sharpes = [1.0, 2.0, 3.0]
        oos_sharpes = [3.0, 2.0, 1.0]

        pbo = calculate_pbo_logit(is_sharpes, oos_sharpes)
        assert 0.0 <= pbo <= 1.0
