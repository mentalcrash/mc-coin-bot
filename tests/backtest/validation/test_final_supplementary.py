"""Tests for P6B Final Validation — supplementary 2/4 logic.

Validates the _run_final() verdict using mock FoldResult/MonteCarloResult,
without requiring vectorbt or real backtest data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.backtest.validation.models import (
    FINAL_ALT_MAX_PBO,
    FINAL_MAX_PBO,
    FINAL_MIN_OOS_SHARPE,
    FINAL_MIN_SUPPLEMENTARY_PASS,
    FoldResult,
    MonteCarloResult,
    SplitInfo,
    ValidationResult,
)


def _make_split(fold_id: int = 0) -> SplitInfo:
    """Minimal SplitInfo for testing."""
    return SplitInfo(
        fold_id=fold_id,
        train_start=datetime(2022, 1, 1, tzinfo=UTC),
        train_end=datetime(2023, 12, 31, tzinfo=UTC),
        test_start=datetime(2024, 1, 1, tzinfo=UTC),
        test_end=datetime(2024, 12, 31, tzinfo=UTC),
        train_periods=730,
        test_periods=365,
    )


def _make_fold(
    fold_id: int,
    train_sharpe: float,
    test_sharpe: float,
) -> FoldResult:
    return FoldResult(
        fold_id=fold_id,
        split=_make_split(fold_id),
        train_sharpe=train_sharpe,
        test_sharpe=test_sharpe,
        train_return=10.0,
        test_return=5.0,
        train_max_drawdown=10.0,
        test_max_drawdown=15.0,
        train_trades=50,
        test_trades=30,
    )


def _make_mc(p_value: float = 0.01) -> MonteCarloResult:
    return MonteCarloResult(
        n_simulations=1000,
        sharpe_mean=0.5,
        sharpe_std=0.1,
        sharpe_percentiles={5: 0.3, 25: 0.4, 50: 0.5, 75: 0.6, 95: 0.7},
        sharpe_ci_lower=0.3,
        sharpe_ci_upper=0.7,
        p_value=p_value,
    )


def _run_final_verdict(
    fold_results: list[FoldResult],
    monte_carlo: MonteCarloResult,
    pbo: float,
    dsr: float,
) -> ValidationResult:
    """Run _run_final with mocked internals, returning the ValidationResult."""
    from src.backtest.validation.validator import TieredValidator

    mock_engine = MagicMock()
    validator = TieredValidator(engine=mock_engine)

    # Mock split_cpcv to yield pre-built fold results
    mock_splits = []
    for fr in fold_results:
        train_result = MagicMock()
        train_result.metrics.sharpe_ratio = fr.train_sharpe
        train_result.metrics.total_return = fr.train_return
        train_result.metrics.max_drawdown = fr.train_max_drawdown
        train_result.metrics.total_trades = fr.train_trades

        test_result = MagicMock()
        test_result.metrics.sharpe_ratio = fr.test_sharpe
        test_result.metrics.total_return = fr.test_return
        test_result.metrics.max_drawdown = fr.test_max_drawdown
        test_result.metrics.total_trades = fr.test_trades

        mock_splits.append((MagicMock(), MagicMock(), fr.split))
        mock_engine.run.side_effect = _interleave_results(
            [(train_result, test_result) for fr in fold_results]
        )

    # Re-setup side_effect properly
    results_flat: list[MagicMock] = []
    for fr in fold_results:
        train_r = MagicMock()
        train_r.metrics.sharpe_ratio = fr.train_sharpe
        train_r.metrics.total_return = fr.train_return
        train_r.metrics.max_drawdown = fr.train_max_drawdown
        train_r.metrics.total_trades = fr.train_trades

        test_r = MagicMock()
        test_r.metrics.sharpe_ratio = fr.test_sharpe
        test_r.metrics.total_return = fr.test_return
        test_r.metrics.max_drawdown = fr.test_max_drawdown
        test_r.metrics.total_trades = fr.test_trades

        results_flat.extend([train_r, test_r])

    mock_engine.run.side_effect = results_flat

    with (
        patch(
            "src.backtest.validation.validator.split_cpcv",
            return_value=mock_splits,
        ),
        patch(
            "src.backtest.validation.pbo.calculate_pbo",
            return_value=pbo,
        ),
        patch(
            "src.backtest.validation.deflated_sharpe.deflated_sharpe_ratio",
            return_value=dsr,
        ),
        patch.object(
            validator,
            "_run_monte_carlo",
            return_value=monte_carlo,
        ),
    ):
        mock_data = MagicMock()
        mock_strategy = MagicMock()
        mock_portfolio = MagicMock()

        return validator._run_final(
            data=mock_data,
            strategy=mock_strategy,
            portfolio=mock_portfolio,
        )


def _interleave_results(
    pairs: list[tuple[MagicMock, MagicMock]],
) -> list[MagicMock]:
    flat: list[MagicMock] = []
    for train, test in pairs:
        flat.extend([train, test])
    return flat


# =============================================================================
# Test cases
# =============================================================================


class TestFinalSupplementary:
    """P6B supplementary 2/4 verdict tests."""

    def test_all_4_supplementary_pass(self) -> None:
        """보충 4/4 통과 → PASS."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(5)]
        mc = _make_mc(p_value=0.01)
        result = _run_final_verdict(folds, mc, pbo=0.2, dsr=0.7)
        assert result.passed is True

    def test_2_of_4_supplementary_pass(self) -> None:
        """보충 2/4 통과 → PASS (decay + mc OK, pbo + dsr FAIL)."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(5)]
        mc = _make_mc(p_value=0.05)  # mc PASS (< 0.10)
        # pbo=0.9 → Path A fail (>0.40), Path B fail (>0.80)
        # dsr=0.3 → fail (< 0.5)
        # decay: (1.0-0.6)/1.0 = 0.4 <= 0.5 → PASS
        result = _run_final_verdict(folds, mc, pbo=0.9, dsr=0.3)
        assert result.passed is True

    def test_1_of_4_supplementary_fail(self) -> None:
        """보충 1/4 통과 → FAIL."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(5)]
        mc = _make_mc(p_value=0.20)  # mc FAIL (> 0.10)
        # pbo=0.9 → FAIL
        # dsr=0.3 → FAIL
        # decay: 0.4 <= 0.5 → PASS (only 1 pass)
        result = _run_final_verdict(folds, mc, pbo=0.9, dsr=0.3)
        assert result.passed is False
        assert any("Supplementary" in r for r in result.failure_reasons)

    def test_oos_sharpe_fail_overrides_supplementary(self) -> None:
        """OOS Sharpe < 0.3 → 보충 4/4여도 FAIL."""
        folds = [_make_fold(i, 1.0, 0.1) for i in range(5)]
        mc = _make_mc(p_value=0.01)
        result = _run_final_verdict(folds, mc, pbo=0.1, dsr=0.8)
        assert result.passed is False
        assert any("OOS Sharpe" in r for r in result.failure_reasons)

    def test_pbo_path_a_pass(self) -> None:
        """PBO Path A: PBO < 40% → PBO supplementary PASS."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(5)]
        mc = _make_mc(p_value=0.20)  # mc FAIL
        # pbo=0.3 < 0.40 → Path A PASS
        # dsr=0.3 → FAIL
        # decay=0.4 → PASS
        # supplementary: decay + pbo = 2/4 → PASS
        result = _run_final_verdict(folds, mc, pbo=0.3, dsr=0.3)
        assert result.passed is True

    def test_pbo_path_b_pass(self) -> None:
        """PBO Path B: PBO 60% < 80%, all fold OOS > 0, MC p < 0.05 → PBO PASS."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(5)]  # all OOS > 0
        mc = _make_mc(p_value=0.02)  # < 0.05 for Path B
        # pbo=0.60 → Path A FAIL (>0.40), Path B: 0.60<0.80 + all OOS>0 + p<0.05 → PASS
        # dsr=0.3 → FAIL
        # decay=0.4 → PASS
        # mc=0.02 → PASS
        # supplementary: decay + pbo + mc = 3/4 → PASS
        result = _run_final_verdict(folds, mc, pbo=0.6, dsr=0.3)
        assert result.passed is True

    def test_pbo_path_b_fail_negative_fold(self) -> None:
        """PBO Path B FAIL: PBO 70%, one fold OOS < 0."""
        folds = [_make_fold(i, 1.0, 0.6) for i in range(4)]
        folds.append(_make_fold(4, 1.0, -0.1))  # negative OOS
        mc = _make_mc(p_value=0.02)
        # pbo=0.70 → Path A FAIL (>0.40), Path B: all fold OOS>0 FAILS → PBO FAIL
        # dsr=0.3 → FAIL
        # decay: avg_test=0.6*4/5 + (-0.1)/5 = 0.46; avg_train=1.0; decay=(1-0.46)/1=0.54 > 0.5 → FAIL
        # mc=0.02 → PASS
        # supplementary: mc only = 1/4 → FAIL
        result = _run_final_verdict(folds, mc, pbo=0.7, dsr=0.3)
        assert result.passed is False


class TestSupplementaryConstants:
    """Verify constant values are sensible."""

    def test_min_supplementary_pass(self) -> None:
        assert FINAL_MIN_SUPPLEMENTARY_PASS == 2

    def test_pbo_thresholds(self) -> None:
        assert FINAL_MAX_PBO < FINAL_ALT_MAX_PBO
        assert pytest.approx(0.40) == FINAL_MAX_PBO
        assert pytest.approx(0.80) == FINAL_ALT_MAX_PBO

    def test_oos_sharpe_threshold(self) -> None:
        assert pytest.approx(0.3) == FINAL_MIN_OOS_SHARPE
