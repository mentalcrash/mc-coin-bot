"""Validation report generation.

검증 결과를 텍스트 포맷으로 요약합니다.

Rules Applied:
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.backtest.validation.models import ValidationResult


def generate_validation_report(
    result: ValidationResult,
    output_path: Path | None = None,
) -> str:
    """검증 결과 요약 리포트 생성.

    Args:
        result: ValidationResult 인스턴스
        output_path: 파일 출력 경로 (None이면 문자열만 반환)

    Returns:
        리포트 텍스트
    """
    lines: list[str] = []

    # Header
    lines.append("=" * 60)
    lines.append(f"  Validation Report: {result.level.value.upper()}")
    lines.append(f"  Verdict: {result.verdict}")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    lines.append("--- Summary ---")
    lines.append(f"  Total Folds:        {result.total_folds}")
    lines.append(f"  Avg Train Sharpe:   {result.avg_train_sharpe:.3f}")
    lines.append(f"  Avg Test Sharpe:    {result.avg_test_sharpe:.3f}")
    lines.append(f"  Sharpe Decay:       {result.avg_sharpe_decay:.1%}")
    lines.append(f"  Consistency:        {result.consistency_ratio:.1%}")
    lines.append(f"  Overfit Prob:       {result.overfit_probability:.1%}")
    lines.append(f"  Sharpe Stability:   {result.sharpe_stability:.3f}")
    lines.append(f"  Computation Time:   {result.computation_time_seconds:.1f}s")
    lines.append("")

    # Fold Details
    if result.fold_results:
        lines.append("--- Fold Results ---")
        lines.append(
            f"  {'Fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>11}  {'Decay':>8}  {'Consistent':>10}"
        )
        lines.append("  " + "-" * 50)

        for fold in result.fold_results:
            consistent_str = "Yes" if fold.is_consistent else "No"
            lines.append(
                f"  {fold.fold_id:>4}  {fold.train_sharpe:>10.3f}  {fold.test_sharpe:>11.3f}  {fold.sharpe_decay:>7.1%}  {consistent_str:>10}"
            )
        lines.append("")

    # Monte Carlo
    if result.monte_carlo is not None:
        mc = result.monte_carlo
        lines.append("--- Monte Carlo ---")
        lines.append(f"  Simulations:   {mc.n_simulations}")
        lines.append(f"  Sharpe Mean:   {mc.sharpe_mean:.3f}")
        lines.append(f"  Sharpe Std:    {mc.sharpe_std:.3f}")
        lines.append(f"  95% CI:        [{mc.sharpe_ci_lower:.3f}, {mc.sharpe_ci_upper:.3f}]")
        lines.append(f"  P-value:       {mc.p_value:.4f}")
        lines.append(f"  Significant:   {'Yes' if mc.is_significant else 'No'}")
        lines.append("")

    # Failure Reasons
    if result.failure_reasons:
        lines.append("--- Failure Reasons ---")
        lines.extend(f"  - {reason}" for reason in result.failure_reasons)
        lines.append("")

    # Passed
    if result.passed and not result.failure_reasons:
        lines.append("--- Result: PASSED ---")
        lines.append("")

    report_text = "\n".join(lines)

    if output_path is not None:
        output_path.write_text(report_text, encoding="utf-8")

    return report_text
