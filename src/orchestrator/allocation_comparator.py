"""Allocation Comparator — 배분 방법 비교 엔진.

동일 데이터로 4가지 배분 방법(EW, InvVol, Risk Parity, Adaptive Kelly)을
비교 백테스트하여 최적 방법을 선택할 수 있는 도구입니다.

Rules Applied:
    - #10 Python Standards: Modern typing, named constants
    - Zero-Tolerance Lint Policy
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.orchestrator.models import AllocationMethod

if TYPE_CHECKING:
    from src.data.market_data import MultiSymbolData
    from src.orchestrator.config import OrchestratorConfig


# ── Result Model ──────────────────────────────────────────────────


@dataclass
class AllocationComparisonResult:
    """배분 방법 비교 결과.

    Attributes:
        method: 배분 방법명
        sharpe: Sharpe Ratio
        mdd: Max Drawdown (%)
        calmar: Calmar Ratio
        cagr: CAGR (%)
        total_return: Total Return (%)
    """

    method: str
    sharpe: float
    mdd: float
    calmar: float
    cagr: float
    total_return: float


# ── Comparator ────────────────────────────────────────────────────


def compare_allocations(
    orch_config: OrchestratorConfig,
    data: MultiSymbolData,
    target_tf: str,
    initial_capital: float = 10_000.0,
) -> list[AllocationComparisonResult]:
    """4가지 배분 방법으로 백테스트하여 결과를 비교합니다.

    Args:
        orch_config: OrchestratorConfig (base config)
        data: MultiSymbolData (1m)
        target_tf: 집계 목표 TF
        initial_capital: 초기 자본

    Returns:
        AllocationComparisonResult 리스트 (4개, Sharpe 내림차순 정렬)
    """
    from src.eda.orchestrated_runner import OrchestratedRunner

    methods = list(AllocationMethod)
    results: list[AllocationComparisonResult] = []

    for method in methods:
        logger.info("Running backtest with allocation_method={}", method.value)

        # Config variant 생성 (frozen이므로 model_copy 사용)
        variant = orch_config.model_copy(update={"allocation_method": method})

        runner = OrchestratedRunner.backtest(
            orchestrator_config=variant,
            data=data,
            target_timeframe=target_tf,
            initial_capital=initial_capital,
        )
        result = asyncio.run(runner.run())

        metrics = result.portfolio_metrics
        calmar = abs(metrics.cagr / metrics.max_drawdown) if metrics.max_drawdown != 0 else 0.0

        results.append(
            AllocationComparisonResult(
                method=method.value,
                sharpe=metrics.sharpe_ratio,
                mdd=metrics.max_drawdown,
                calmar=calmar,
                cagr=metrics.cagr,
                total_return=metrics.total_return,
            )
        )

    # Sharpe 내림차순 정렬
    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results
