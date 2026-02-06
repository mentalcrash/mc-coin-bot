"""Tiered Validator implementation.

단계별 검증을 수행하는 TieredValidator를 제공합니다.

Rules Applied:
    - #10 Python Standards: Modern typing, match statement
    - #26 VectorBT Standards: Stateless design
"""

from __future__ import annotations

import statistics
import time
from datetime import UTC
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.backtest.validation.levels import ValidationLevel
from src.backtest.validation.models import (
    MILESTONE_MAX_SHARPE_DECAY,
    MILESTONE_MIN_CONSISTENCY,
    MILESTONE_MIN_OOS_SHARPE,
    MULTI_CPCV_MAX_PBO,
    MULTI_DEFLATED_SHARPE_MIN,
    MULTI_QUICK_MAX_DEGRADATION,
    MULTI_QUICK_MIN_OOS_SHARPE,
    MULTI_WFA_MAX_SHARPE_DECAY,
    MULTI_WFA_MIN_CONSISTENCY,
    MULTI_WFA_MIN_OOS_SHARPE,
    QUICK_MAX_SHARPE_DECAY,
    QUICK_MIN_OOS_SHARPE,
    FoldResult,
    MonteCarloResult,
    SplitInfo,
    ValidationResult,
)
from src.backtest.validation.splitters import (
    split_cpcv,
    split_is_oos,
    split_multi_cpcv,
    split_multi_is_oos,
    split_multi_walk_forward,
    split_walk_forward,
)

if TYPE_CHECKING:
    from src.backtest.engine import BacktestEngine
    from src.backtest.request import MultiAssetBacktestRequest
    from src.data.market_data import MarketDataSet, MultiSymbolData
    from src.portfolio.portfolio import Portfolio
    from src.strategy.base import BaseStrategy


class TieredValidator:
    """단계별 검증 수행자.

    개발 단계에 따라 적절한 수준의 검증을 수행합니다.
    Stateless 설계를 따릅니다.

    Attributes:
        _engine: BacktestEngine 인스턴스 (DI)

    Example:
        >>> from src.backtest.validation import TieredValidator, ValidationLevel
        >>> from src.backtest.engine import BacktestEngine
        >>>
        >>> validator = TieredValidator()
        >>> result = validator.validate(
        ...     level=ValidationLevel.QUICK,
        ...     data=market_data,
        ...     strategy=strategy,
        ...     portfolio=portfolio,
        ... )
        >>> print(result.verdict)
    """

    def __init__(self, engine: BacktestEngine | None = None) -> None:
        """TieredValidator 초기화.

        Args:
            engine: BacktestEngine 인스턴스 (None이면 새로 생성)
        """
        # Lazy import to avoid circular dependency
        if engine is None:
            from src.backtest.engine import BacktestEngine

            engine = BacktestEngine()
        self._engine = engine

    def validate(
        self,
        level: ValidationLevel,
        data: MarketDataSet,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        *,
        # Quick 옵션
        split_ratio: float = 0.7,
        # Milestone 옵션
        n_folds: int = 5,
        expanding: bool = True,
        # Final 옵션
        n_splits: int = 5,
        n_test_splits: int = 2,
        n_monte_carlo: int = 1000,
    ) -> ValidationResult:
        """검증 수행.

        Args:
            level: 검증 레벨 (quick/milestone/final)
            data: 전체 MarketDataSet
            strategy: 전략 인스턴스
            portfolio: 포트폴리오 설정
            split_ratio: [QUICK] Train 비율 (기본값 0.7)
            n_folds: [MILESTONE] Fold 수 (기본값 5)
            expanding: [MILESTONE] 누적 Train 여부 (기본값 True)
            n_splits: [FINAL] CPCV 그룹 수 (기본값 5)
            n_test_splits: [FINAL] Test 그룹 수 (기본값 2)
            n_monte_carlo: [FINAL] Monte Carlo 시뮬레이션 횟수 (기본값 1000)

        Returns:
            ValidationResult

        Example:
            >>> result = validator.validate(
            ...     level=ValidationLevel.QUICK,
            ...     data=data,
            ...     strategy=strategy,
            ...     portfolio=portfolio,
            ... )
        """
        logger.info(f"Validation started: level={level.value}")

        match level:
            case ValidationLevel.QUICK:
                return self._run_quick(data, strategy, portfolio, split_ratio=split_ratio)
            case ValidationLevel.MILESTONE:
                return self._run_milestone(
                    data, strategy, portfolio, n_folds=n_folds, expanding=expanding
                )
            case ValidationLevel.FINAL:
                return self._run_final(
                    data,
                    strategy,
                    portfolio,
                    n_splits=n_splits,
                    n_test_splits=n_test_splits,
                    n_monte_carlo=n_monte_carlo,
                )

    def _run_quick(
        self,
        data: MarketDataSet,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        split_ratio: float = 0.7,
    ) -> ValidationResult:
        """Quick Validation: IS/OOS Split (70/30).

        가장 빠른 검증 방법. 매 iteration마다 사용합니다.

        Args:
            data: 전체 데이터
            strategy: 전략
            portfolio: 포트폴리오
            split_ratio: Train 비율 (기본값 0.7)

        Returns:
            ValidationResult
        """
        from src.backtest.request import BacktestRequest

        start_time = time.perf_counter()
        logger.debug(f"Quick validation: split_ratio={split_ratio}")

        # 데이터 분할
        train_data, test_data = split_is_oos(data, ratio=split_ratio)
        logger.debug(f"Split: Train={train_data.periods}, Test={test_data.periods}")

        # Train 백테스트
        train_request = BacktestRequest(
            data=train_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        train_result = self._engine.run(train_request)

        # Test 백테스트
        test_request = BacktestRequest(
            data=test_data,
            strategy=strategy,
            portfolio=portfolio,
        )
        test_result = self._engine.run(test_request)

        # Fold 결과 생성
        from src.backtest.validation.splitters import get_split_info_is_oos

        split_info = get_split_info_is_oos(data, ratio=split_ratio)

        fold_result = FoldResult(
            fold_id=0,
            split=split_info,
            train_sharpe=train_result.metrics.sharpe_ratio,
            test_sharpe=test_result.metrics.sharpe_ratio,
            train_return=train_result.metrics.total_return,
            test_return=test_result.metrics.total_return,
            train_max_drawdown=train_result.metrics.max_drawdown,
            test_max_drawdown=test_result.metrics.max_drawdown,
            train_trades=train_result.metrics.total_trades,
            test_trades=test_result.metrics.total_trades,
        )

        # Pass/Fail 판정
        failure_reasons: list[str] = []

        if test_result.metrics.sharpe_ratio < QUICK_MIN_OOS_SHARPE:
            failure_reasons.append(
                f"OOS Sharpe ({test_result.metrics.sharpe_ratio:.2f}) < {QUICK_MIN_OOS_SHARPE}"
            )

        sharpe_decay = fold_result.sharpe_decay
        if sharpe_decay > QUICK_MAX_SHARPE_DECAY:
            failure_reasons.append(
                f"Sharpe Decay ({sharpe_decay:.1%}) > {QUICK_MAX_SHARPE_DECAY:.0%}"
            )

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Quick validation complete: passed={passed}, IS_sharpe={train_result.metrics.sharpe_ratio:.2f}, OOS_sharpe={test_result.metrics.sharpe_ratio:.2f}, decay={sharpe_decay:.1%}"
        )

        return ValidationResult(
            level=ValidationLevel.QUICK,
            fold_results=(fold_result,),
            monte_carlo=None,
            avg_train_sharpe=train_result.metrics.sharpe_ratio,
            avg_test_sharpe=test_result.metrics.sharpe_ratio,
            sharpe_stability=0.0,  # 단일 Fold이므로 0
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=1,
            computation_time_seconds=computation_time,
        )

    def _run_milestone(
        self,
        data: MarketDataSet,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        n_folds: int = 5,
        expanding: bool = True,
    ) -> ValidationResult:
        """Milestone Validation: Walk-Forward (5 fold).

        중요한 변경사항에 대한 검증. 새 필터 추가 등에 사용합니다.

        Args:
            data: 전체 데이터
            strategy: 전략
            portfolio: 포트폴리오
            n_folds: Fold 수 (기본값 5)
            expanding: True면 누적 Train (기본값 True)

        Returns:
            ValidationResult
        """
        from src.backtest.request import BacktestRequest

        start_time = time.perf_counter()
        logger.debug(f"Milestone validation: n_folds={n_folds}, expanding={expanding}")

        # Walk-Forward 분할
        splits = split_walk_forward(
            data,
            n_folds=n_folds,
            expanding=expanding,
        )

        fold_results: list[FoldResult] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []

        for train_data, test_data, split_info in splits:
            logger.debug(
                f"Fold {split_info.fold_id}: Train={split_info.train_periods}, Test={split_info.test_periods}"
            )

            # Train 백테스트
            train_request = BacktestRequest(
                data=train_data,
                strategy=strategy,
                portfolio=portfolio,
            )
            train_result = self._engine.run(train_request)

            # Test 백테스트
            test_request = BacktestRequest(
                data=test_data,
                strategy=strategy,
                portfolio=portfolio,
            )
            test_result = self._engine.run(test_request)

            fold_result = FoldResult(
                fold_id=split_info.fold_id,
                split=split_info,
                train_sharpe=train_result.metrics.sharpe_ratio,
                test_sharpe=test_result.metrics.sharpe_ratio,
                train_return=train_result.metrics.total_return,
                test_return=test_result.metrics.total_return,
                train_max_drawdown=train_result.metrics.max_drawdown,
                test_max_drawdown=test_result.metrics.max_drawdown,
                train_trades=train_result.metrics.total_trades,
                test_trades=test_result.metrics.total_trades,
            )

            fold_results.append(fold_result)
            train_sharpes.append(train_result.metrics.sharpe_ratio)
            test_sharpes.append(test_result.metrics.sharpe_ratio)

        # 통계 계산
        avg_train_sharpe = statistics.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = statistics.mean(test_sharpes) if test_sharpes else 0.0
        sharpe_stability = statistics.stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0

        # 일관성 계산
        consistent_count = sum(1 for f in fold_results if f.is_consistent)
        consistency_ratio = consistent_count / len(fold_results) if fold_results else 0.0

        # Pass/Fail 판정
        failure_reasons: list[str] = []

        if avg_test_sharpe < MILESTONE_MIN_OOS_SHARPE:
            failure_reasons.append(
                f"Avg OOS Sharpe ({avg_test_sharpe:.2f}) < {MILESTONE_MIN_OOS_SHARPE}"
            )

        avg_sharpe_decay = (
            (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
            if avg_train_sharpe != 0
            else 0.0
        )
        if avg_sharpe_decay > MILESTONE_MAX_SHARPE_DECAY:
            failure_reasons.append(
                f"Avg Sharpe Decay ({avg_sharpe_decay:.1%}) > {MILESTONE_MAX_SHARPE_DECAY:.0%}"
            )

        if consistency_ratio < MILESTONE_MIN_CONSISTENCY:
            failure_reasons.append(
                f"Consistency ({consistency_ratio:.1%}) < {MILESTONE_MIN_CONSISTENCY:.0%}"
            )

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Milestone validation complete: passed={passed}, avg_IS_sharpe={avg_train_sharpe:.2f}, avg_OOS_sharpe={avg_test_sharpe:.2f}, consistency={consistency_ratio:.1%}"
        )

        return ValidationResult(
            level=ValidationLevel.MILESTONE,
            fold_results=tuple(fold_results),
            monte_carlo=None,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_stability=sharpe_stability,
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=len(fold_results),
            computation_time_seconds=computation_time,
        )

    def _run_final(
        self,
        data: MarketDataSet,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        n_splits: int = 5,
        n_test_splits: int = 2,
        n_monte_carlo: int = 1000,
    ) -> ValidationResult:
        """Final Validation: CPCV + Monte Carlo.

        전략 완성 시 최종 검증. 가장 엄격한 검증입니다.

        Args:
            data: 전체 데이터
            strategy: 전략
            portfolio: 포트폴리오
            n_splits: CPCV 그룹 수 (기본값 5)
            n_test_splits: Test 그룹 수 (기본값 2)
            n_monte_carlo: Monte Carlo 시뮬레이션 횟수 (기본값 1000)

        Returns:
            ValidationResult
        """
        from src.backtest.request import BacktestRequest

        start_time = time.perf_counter()
        logger.debug(
            f"Final validation: n_splits={n_splits}, n_test_splits={n_test_splits}, n_monte_carlo={n_monte_carlo}"
        )

        # CPCV 분할
        fold_results: list[FoldResult] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []

        for train_data, test_data, split_info in split_cpcv(
            data,
            n_splits=n_splits,
            n_test_splits=n_test_splits,
        ):
            logger.debug(
                f"CPCV Fold {split_info.fold_id}: Train={split_info.train_periods}, Test={split_info.test_periods}"
            )

            # Train 백테스트
            train_request = BacktestRequest(
                data=train_data,
                strategy=strategy,
                portfolio=portfolio,
            )
            train_result = self._engine.run(train_request)

            # Test 백테스트
            test_request = BacktestRequest(
                data=test_data,
                strategy=strategy,
                portfolio=portfolio,
            )
            test_result = self._engine.run(test_request)

            fold_result = FoldResult(
                fold_id=split_info.fold_id,
                split=split_info,
                train_sharpe=train_result.metrics.sharpe_ratio,
                test_sharpe=test_result.metrics.sharpe_ratio,
                train_return=train_result.metrics.total_return,
                test_return=test_result.metrics.total_return,
                train_max_drawdown=train_result.metrics.max_drawdown,
                test_max_drawdown=test_result.metrics.max_drawdown,
                train_trades=train_result.metrics.total_trades,
                test_trades=test_result.metrics.total_trades,
            )

            fold_results.append(fold_result)
            train_sharpes.append(train_result.metrics.sharpe_ratio)
            test_sharpes.append(test_result.metrics.sharpe_ratio)

        # 통계 계산
        avg_train_sharpe = statistics.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = statistics.mean(test_sharpes) if test_sharpes else 0.0
        sharpe_stability = statistics.stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0

        # Monte Carlo 시뮬레이션 (Returns Bootstrap)
        monte_carlo = self._run_monte_carlo(
            test_sharpes=test_sharpes,
            n_simulations=n_monte_carlo,
        )

        # Pass/Fail 판정
        failure_reasons: list[str] = []

        # 기본 기준
        from src.backtest.validation.models import (
            FINAL_MAX_P_VALUE,
            FINAL_MAX_SHARPE_DECAY,
            FINAL_MIN_CONSISTENCY,
            FINAL_MIN_OOS_SHARPE,
        )

        if avg_test_sharpe < FINAL_MIN_OOS_SHARPE:
            failure_reasons.append(
                f"Avg OOS Sharpe ({avg_test_sharpe:.2f}) < {FINAL_MIN_OOS_SHARPE}"
            )

        avg_sharpe_decay = (
            (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
            if avg_train_sharpe != 0
            else 0.0
        )
        if avg_sharpe_decay > FINAL_MAX_SHARPE_DECAY:
            failure_reasons.append(
                f"Avg Sharpe Decay ({avg_sharpe_decay:.1%}) > {FINAL_MAX_SHARPE_DECAY:.0%}"
            )

        consistent_count = sum(1 for f in fold_results if f.is_consistent)
        consistency_ratio = consistent_count / len(fold_results) if fold_results else 0.0

        if consistency_ratio < FINAL_MIN_CONSISTENCY:
            failure_reasons.append(
                f"Consistency ({consistency_ratio:.1%}) < {FINAL_MIN_CONSISTENCY:.0%}"
            )

        # Monte Carlo 통계적 유의성
        if monte_carlo.p_value > FINAL_MAX_P_VALUE:
            failure_reasons.append(f"P-value ({monte_carlo.p_value:.3f}) > {FINAL_MAX_P_VALUE}")

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Final validation complete: passed={passed}, avg_IS_sharpe={avg_train_sharpe:.2f}, avg_OOS_sharpe={avg_test_sharpe:.2f}, p_value={monte_carlo.p_value:.3f}"
        )

        return ValidationResult(
            level=ValidationLevel.FINAL,
            fold_results=tuple(fold_results),
            monte_carlo=monte_carlo,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_stability=sharpe_stability,
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=len(fold_results),
            computation_time_seconds=computation_time,
        )

    def _run_monte_carlo(
        self,
        test_sharpes: list[float],
        n_simulations: int = 1000,
    ) -> MonteCarloResult:
        """Monte Carlo 시뮬레이션 (Bootstrap).

        OOS Sharpe 값들을 부트스트랩하여 분포를 추정합니다.

        Args:
            test_sharpes: OOS Sharpe 값 목록
            n_simulations: 시뮬레이션 횟수

        Returns:
            MonteCarloResult
        """
        if not test_sharpes:
            return MonteCarloResult(
                n_simulations=n_simulations,
                sharpe_mean=0.0,
                sharpe_std=0.0,
                sharpe_percentiles={5: 0, 25: 0, 50: 0, 75: 0, 95: 0},
                sharpe_ci_lower=0.0,
                sharpe_ci_upper=0.0,
                p_value=1.0,
            )

        # Bootstrap resampling
        rng = np.random.default_rng(seed=42)
        bootstrap_means: list[float] = []

        for _ in range(n_simulations):
            sample = rng.choice(test_sharpes, size=len(test_sharpes), replace=True)
            bootstrap_means.append(float(np.mean(sample)))

        # 통계 계산
        sharpe_mean = float(np.mean(bootstrap_means))
        sharpe_std = float(np.std(bootstrap_means))

        percentiles = np.percentile(bootstrap_means, [5, 25, 50, 75, 95])
        sharpe_percentiles = {
            5: float(percentiles[0]),
            25: float(percentiles[1]),
            50: float(percentiles[2]),
            75: float(percentiles[3]),
            95: float(percentiles[4]),
        }

        # 95% 신뢰구간
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))

        # P-value 계산 (H0: mean <= 0)
        # Bootstrap 분포에서 0 이하인 비율
        p_value = float(np.mean(np.array(bootstrap_means) <= 0))

        return MonteCarloResult(
            n_simulations=n_simulations,
            sharpe_mean=sharpe_mean,
            sharpe_std=sharpe_std,
            sharpe_percentiles=sharpe_percentiles,
            sharpe_ci_lower=ci_lower,
            sharpe_ci_upper=ci_upper,
            p_value=p_value,
        )

    # =========================================================================
    # Multi-Asset Validation
    # =========================================================================

    def validate_multi(
        self,
        level: ValidationLevel,
        data: MultiSymbolData,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        weights: dict[str, float] | None = None,
        *,
        split_ratio: float = 0.7,
        n_folds: int = 5,
        expanding: bool = True,
        n_splits: int = 5,
        n_test_splits: int = 2,
        n_monte_carlo: int = 1000,
    ) -> ValidationResult:
        """멀티에셋 포트폴리오 검증.

        Args:
            level: 검증 레벨 (quick/milestone/final)
            data: 멀티에셋 데이터
            strategy: 전략 인스턴스
            portfolio: 포트폴리오 설정
            weights: 자산 배분 비중 (None이면 EW)
            split_ratio: [QUICK] Train 비율
            n_folds: [MILESTONE] Fold 수
            expanding: [MILESTONE] 누적 Train 여부
            n_splits: [FINAL] CPCV 그룹 수
            n_test_splits: [FINAL] Test 그룹 수
            n_monte_carlo: [FINAL] Monte Carlo 횟수

        Returns:
            ValidationResult
        """
        logger.info(
            f"Multi-asset validation started: level={level.value}, assets={len(data.symbols)}"
        )

        match level:
            case ValidationLevel.QUICK:
                return self._run_quick_multi(
                    data, strategy, portfolio, weights, split_ratio=split_ratio
                )
            case ValidationLevel.MILESTONE:
                return self._run_milestone_multi(
                    data,
                    strategy,
                    portfolio,
                    weights,
                    n_folds=n_folds,
                    expanding=expanding,
                )
            case ValidationLevel.FINAL:
                return self._run_final_multi(
                    data,
                    strategy,
                    portfolio,
                    weights,
                    n_splits=n_splits,
                    n_test_splits=n_test_splits,
                    n_monte_carlo=n_monte_carlo,
                )

    def _run_quick_multi(
        self,
        data: MultiSymbolData,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        weights: dict[str, float] | None,
        split_ratio: float = 0.7,
    ) -> ValidationResult:
        """Multi-Asset Quick Validation: IS/OOS Split."""
        start_time = time.perf_counter()
        logger.debug(f"Multi-asset quick validation: split_ratio={split_ratio}")

        train_data, test_data = split_multi_is_oos(data, ratio=split_ratio)

        train_request = self._make_multi_request(train_data, strategy, portfolio, weights)
        test_request = self._make_multi_request(test_data, strategy, portfolio, weights)

        train_result = self._engine.run_multi(train_request)
        test_result = self._engine.run_multi(test_request)

        train_sharpe = train_result.portfolio_metrics.sharpe_ratio
        test_sharpe = test_result.portfolio_metrics.sharpe_ratio

        ref_df = data.ohlcv[data.symbols[0]]
        split_idx = int(len(ref_df) * split_ratio)
        ref_index = ref_df.index

        fold_result = FoldResult(
            fold_id=0,
            split=SplitInfo(
                fold_id=0,
                train_start=ref_index[0].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
                train_end=ref_index[split_idx - 1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
                test_start=ref_index[split_idx].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
                test_end=ref_index[-1].to_pydatetime().replace(tzinfo=UTC),  # type: ignore[union-attr]
                train_periods=split_idx,
                test_periods=len(ref_df) - split_idx,
            ),
            train_sharpe=train_sharpe,
            test_sharpe=test_sharpe,
            train_return=train_result.portfolio_metrics.total_return,
            test_return=test_result.portfolio_metrics.total_return,
            train_max_drawdown=train_result.portfolio_metrics.max_drawdown,
            test_max_drawdown=test_result.portfolio_metrics.max_drawdown,
        )

        failure_reasons: list[str] = []

        if test_sharpe < MULTI_QUICK_MIN_OOS_SHARPE:
            failure_reasons.append(f"OOS Sharpe ({test_sharpe:.2f}) < {MULTI_QUICK_MIN_OOS_SHARPE}")

        sharpe_decay = fold_result.sharpe_decay
        if sharpe_decay > MULTI_QUICK_MAX_DEGRADATION:
            failure_reasons.append(
                f"Sharpe Decay ({sharpe_decay:.1%}) > {MULTI_QUICK_MAX_DEGRADATION:.0%}"
            )

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Multi-asset quick validation: passed={passed}, IS={train_sharpe:.2f}, OOS={test_sharpe:.2f}"
        )

        return ValidationResult(
            level=ValidationLevel.QUICK,
            fold_results=(fold_result,),
            monte_carlo=None,
            avg_train_sharpe=train_sharpe,
            avg_test_sharpe=test_sharpe,
            sharpe_stability=0.0,
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=1,
            computation_time_seconds=computation_time,
        )

    def _run_milestone_multi(
        self,
        data: MultiSymbolData,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        weights: dict[str, float] | None,
        n_folds: int = 5,
        expanding: bool = True,
    ) -> ValidationResult:
        """Multi-Asset Milestone Validation: Walk-Forward."""
        start_time = time.perf_counter()
        logger.debug(f"Multi-asset milestone: n_folds={n_folds}, expanding={expanding}")

        splits = split_multi_walk_forward(data, n_folds=n_folds, expanding=expanding)

        fold_results: list[FoldResult] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []

        for train_data, test_data, split_info in splits:
            train_request = self._make_multi_request(train_data, strategy, portfolio, weights)
            test_request = self._make_multi_request(test_data, strategy, portfolio, weights)

            train_result = self._engine.run_multi(train_request)
            test_result = self._engine.run_multi(test_request)

            fold_result = FoldResult(
                fold_id=split_info.fold_id,
                split=split_info,
                train_sharpe=train_result.portfolio_metrics.sharpe_ratio,
                test_sharpe=test_result.portfolio_metrics.sharpe_ratio,
                train_return=train_result.portfolio_metrics.total_return,
                test_return=test_result.portfolio_metrics.total_return,
                train_max_drawdown=train_result.portfolio_metrics.max_drawdown,
                test_max_drawdown=test_result.portfolio_metrics.max_drawdown,
            )

            fold_results.append(fold_result)
            train_sharpes.append(train_result.portfolio_metrics.sharpe_ratio)
            test_sharpes.append(test_result.portfolio_metrics.sharpe_ratio)

        avg_train_sharpe = statistics.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = statistics.mean(test_sharpes) if test_sharpes else 0.0
        sharpe_stability = statistics.stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0

        consistent_count = sum(1 for f in fold_results if f.is_consistent)
        consistency_ratio = consistent_count / len(fold_results) if fold_results else 0.0

        failure_reasons: list[str] = []

        if avg_test_sharpe < MULTI_WFA_MIN_OOS_SHARPE:
            failure_reasons.append(
                f"Avg OOS Sharpe ({avg_test_sharpe:.2f}) < {MULTI_WFA_MIN_OOS_SHARPE}"
            )

        avg_decay = (
            (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
            if avg_train_sharpe != 0
            else 0.0
        )
        if avg_decay > MULTI_WFA_MAX_SHARPE_DECAY:
            failure_reasons.append(
                f"Avg Sharpe Decay ({avg_decay:.1%}) > {MULTI_WFA_MAX_SHARPE_DECAY:.0%}"
            )

        if consistency_ratio < MULTI_WFA_MIN_CONSISTENCY:
            failure_reasons.append(
                f"Consistency ({consistency_ratio:.1%}) < {MULTI_WFA_MIN_CONSISTENCY:.0%}"
            )

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Multi-asset milestone: passed={passed}, avg_IS={avg_train_sharpe:.2f}, avg_OOS={avg_test_sharpe:.2f}"
        )

        return ValidationResult(
            level=ValidationLevel.MILESTONE,
            fold_results=tuple(fold_results),
            monte_carlo=None,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_stability=sharpe_stability,
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=len(fold_results),
            computation_time_seconds=computation_time,
        )

    def _run_final_multi(
        self,
        data: MultiSymbolData,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        weights: dict[str, float] | None,
        n_splits: int = 5,
        n_test_splits: int = 2,
        n_monte_carlo: int = 1000,
    ) -> ValidationResult:
        """Multi-Asset Final Validation: CPCV + DSR + PBO."""
        start_time = time.perf_counter()
        logger.debug(f"Multi-asset final: n_splits={n_splits}, n_test_splits={n_test_splits}")

        fold_results: list[FoldResult] = []
        train_sharpes: list[float] = []
        test_sharpes: list[float] = []

        for train_data, test_data, split_info in split_multi_cpcv(
            data,
            n_splits=n_splits,
            n_test_splits=n_test_splits,
        ):
            train_request = self._make_multi_request(train_data, strategy, portfolio, weights)
            test_request = self._make_multi_request(test_data, strategy, portfolio, weights)

            train_result = self._engine.run_multi(train_request)
            test_result = self._engine.run_multi(test_request)

            fold_result = FoldResult(
                fold_id=split_info.fold_id,
                split=split_info,
                train_sharpe=train_result.portfolio_metrics.sharpe_ratio,
                test_sharpe=test_result.portfolio_metrics.sharpe_ratio,
                train_return=train_result.portfolio_metrics.total_return,
                test_return=test_result.portfolio_metrics.total_return,
                train_max_drawdown=train_result.portfolio_metrics.max_drawdown,
                test_max_drawdown=test_result.portfolio_metrics.max_drawdown,
            )

            fold_results.append(fold_result)
            train_sharpes.append(train_result.portfolio_metrics.sharpe_ratio)
            test_sharpes.append(test_result.portfolio_metrics.sharpe_ratio)

        avg_train_sharpe = statistics.mean(train_sharpes) if train_sharpes else 0.0
        avg_test_sharpe = statistics.mean(test_sharpes) if test_sharpes else 0.0
        sharpe_stability = statistics.stdev(test_sharpes) if len(test_sharpes) > 1 else 0.0

        # PBO 계산
        from src.backtest.validation.pbo import calculate_pbo

        pbo = calculate_pbo(train_sharpes, test_sharpes) if len(train_sharpes) >= 2 else 0.5  # noqa: PLR2004

        # DSR 계산
        from src.backtest.validation.deflated_sharpe import deflated_sharpe_ratio

        n_obs = data.ohlcv[data.symbols[0]].shape[0]
        dsr = deflated_sharpe_ratio(
            observed_sharpe=avg_test_sharpe,
            n_trials=len(fold_results),
            n_observations=n_obs,
        )

        # Monte Carlo
        monte_carlo = self._run_monte_carlo(test_sharpes, n_simulations=n_monte_carlo)

        # Pass/Fail 판정
        failure_reasons: list[str] = []

        if avg_test_sharpe < MULTI_WFA_MIN_OOS_SHARPE:
            failure_reasons.append(
                f"Avg OOS Sharpe ({avg_test_sharpe:.2f}) < {MULTI_WFA_MIN_OOS_SHARPE}"
            )

        if pbo > MULTI_CPCV_MAX_PBO:
            failure_reasons.append(f"PBO ({pbo:.2f}) > {MULTI_CPCV_MAX_PBO}")

        if dsr < MULTI_DEFLATED_SHARPE_MIN:
            failure_reasons.append(f"DSR ({dsr:.2f}) < {MULTI_DEFLATED_SHARPE_MIN}")

        passed = len(failure_reasons) == 0
        computation_time = time.perf_counter() - start_time

        logger.info(
            f"Multi-asset final: passed={passed}, avg_OOS={avg_test_sharpe:.2f}, PBO={pbo:.2f}, DSR={dsr:.2f}"
        )

        return ValidationResult(
            level=ValidationLevel.FINAL,
            fold_results=tuple(fold_results),
            monte_carlo=monte_carlo,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_stability=sharpe_stability,
            passed=passed,
            failure_reasons=tuple(failure_reasons),
            total_folds=len(fold_results),
            computation_time_seconds=computation_time,
        )

    def _make_multi_request(
        self,
        data: MultiSymbolData,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        weights: dict[str, float] | None,
    ) -> MultiAssetBacktestRequest:
        """멀티에셋 백테스트 요청 생성 헬퍼."""
        from src.backtest.request import MultiAssetBacktestRequest

        return MultiAssetBacktestRequest(
            data=data,
            strategy=strategy,
            portfolio=portfolio,
            weights=weights,
        )
