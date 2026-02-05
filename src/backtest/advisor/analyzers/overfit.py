"""Overfitting score analyzer.

전략의 과적합 정도를 정량화합니다.

Rules Applied:
    - #12 Data Engineering: Vectorization
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.backtest.advisor.models import OverfitScore

if TYPE_CHECKING:
    from src.backtest.validation.models import ValidationResult

# Overfit 분석 상수
_DEFAULT_OVERFIT_PROBABILITY = 0.5  # 데이터 없을 때 기본 확률
_OOS_SHARPE_GOOD_THRESHOLD = 0.5  # OOS Sharpe가 이 이상이면 양호
_WEIGHT_DECAY = 0.35  # Sharpe decay 가중치
_WEIGHT_STABILITY = 0.20  # 안정성 가중치
_WEIGHT_INCONSISTENCY = 0.25  # 비일관성 가중치
_WEIGHT_OOS = 0.20  # OOS 성과 가중치


class OverfitAnalyzer:
    """과적합 스코어 분석기.

    Validation 결과를 기반으로 과적합 정도를 정량화합니다.

    Example:
        >>> analyzer = OverfitAnalyzer()
        >>> result = analyzer.analyze(validation_result)
        >>> print(result.overfit_probability)
    """

    def analyze(
        self,
        validation_result: ValidationResult,
    ) -> OverfitScore:
        """과적합 스코어 분석 수행.

        Args:
            validation_result: Tiered Validation 결과

        Returns:
            OverfitScore 분석 결과
        """
        # Fold 결과에서 IS/OOS 지표 추출
        fold_results = validation_result.fold_results

        if not fold_results:
            return OverfitScore(
                is_sharpe=0.0,
                oos_sharpe=0.0,
                is_return=0.0,
                oos_return=0.0,
                overfit_probability=_DEFAULT_OVERFIT_PROBABILITY,
                parameter_sensitivity=None,
            )

        # 평균 IS/OOS 지표
        is_sharpe = validation_result.avg_train_sharpe
        oos_sharpe = validation_result.avg_test_sharpe

        is_return = sum(f.train_return for f in fold_results) / len(fold_results)
        oos_return = sum(f.test_return for f in fold_results) / len(fold_results)

        # 과적합 확률 계산
        overfit_probability = self._calculate_overfit_probability(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            sharpe_stability=validation_result.sharpe_stability,
            consistency_ratio=validation_result.consistency_ratio,
        )

        return OverfitScore(
            is_sharpe=is_sharpe,
            oos_sharpe=oos_sharpe,
            is_return=is_return,
            oos_return=oos_return,
            overfit_probability=overfit_probability,
            parameter_sensitivity=None,  # 추후 구현
        )

    def _calculate_overfit_probability(
        self,
        is_sharpe: float,
        oos_sharpe: float,
        sharpe_stability: float,
        consistency_ratio: float,
    ) -> float:
        """과적합 확률 계산.

        여러 지표를 종합하여 과적합 확률을 추정합니다.

        Args:
            is_sharpe: In-Sample Sharpe
            oos_sharpe: Out-of-Sample Sharpe
            sharpe_stability: OOS Sharpe 안정성 (낮을수록 좋음)
            consistency_ratio: Fold 일관성 비율 (높을수록 좋음)

        Returns:
            과적합 확률 (0-1)
        """
        # 1. Sharpe Decay 점수 (0-1, 높을수록 나쁨)
        if is_sharpe == 0:
            decay_score = _DEFAULT_OVERFIT_PROBABILITY
        else:
            decay = (is_sharpe - oos_sharpe) / abs(is_sharpe)
            decay_score = min(1.0, max(0.0, decay))

        # 2. 안정성 점수 (0-1, 높을수록 나쁨)
        # 안정성이 1.0 이상이면 매우 불안정
        stability_score = min(1.0, sharpe_stability)

        # 3. 비일관성 점수 (0-1, 높을수록 나쁨)
        inconsistency_score = 1.0 - consistency_ratio

        # 4. OOS 성과 점수 (0-1, 높을수록 나쁨)
        # OOS Sharpe가 threshold 이하면 나쁨
        if oos_sharpe > _OOS_SHARPE_GOOD_THRESHOLD:
            oos_score = 0.0
        else:
            oos_score = (_OOS_SHARPE_GOOD_THRESHOLD - oos_sharpe) / _OOS_SHARPE_GOOD_THRESHOLD
        oos_score = min(1.0, max(0.0, oos_score))

        # 가중 평균
        probability = (
            _WEIGHT_DECAY * decay_score
            + _WEIGHT_STABILITY * stability_score
            + _WEIGHT_INCONSISTENCY * inconsistency_score
            + _WEIGHT_OOS * oos_score
        )

        return min(1.0, max(0.0, probability))
