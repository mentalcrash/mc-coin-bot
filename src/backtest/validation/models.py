"""Validation result models.

검증 결과를 표현하는 Pydantic 모델을 정의합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, computed_field
    - #10 Python Standards: Modern typing (X | None, list[])
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from src.backtest.validation.levels import ValidationLevel


class SplitInfo(BaseModel):
    """데이터 분할 정보.

    Train/Test 분할의 메타데이터를 저장합니다.

    Attributes:
        fold_id: Fold 번호 (0-indexed)
        train_start: Train 시작 시각
        train_end: Train 종료 시각
        test_start: Test 시작 시각
        test_end: Test 종료 시각
        train_periods: Train 기간 수
        test_periods: Test 기간 수
    """

    model_config = ConfigDict(frozen=True)

    fold_id: int = Field(..., ge=0, description="Fold 번호")
    train_start: datetime = Field(..., description="Train 시작 시각")
    train_end: datetime = Field(..., description="Train 종료 시각")
    test_start: datetime = Field(..., description="Test 시작 시각")
    test_end: datetime = Field(..., description="Test 종료 시각")
    train_periods: int = Field(..., ge=1, description="Train 기간 수")
    test_periods: int = Field(..., ge=1, description="Test 기간 수")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def train_ratio(self) -> float:
        """Train 비율."""
        total = self.train_periods + self.test_periods
        return self.train_periods / total if total > 0 else 0.0


class FoldResult(BaseModel):
    """단일 Fold 검증 결과.

    Walk-Forward 또는 CPCV의 개별 Fold 결과를 저장합니다.

    Attributes:
        fold_id: Fold 번호
        split: 분할 정보
        train_sharpe: Train 기간 Sharpe Ratio
        test_sharpe: Test 기간 Sharpe Ratio
        train_return: Train 기간 총 수익률 (%)
        test_return: Test 기간 총 수익률 (%)
        train_max_drawdown: Train 기간 최대 낙폭 (%)
        test_max_drawdown: Test 기간 최대 낙폭 (%)
    """

    model_config = ConfigDict(frozen=True)

    fold_id: int = Field(..., ge=0, description="Fold 번호")
    split: SplitInfo = Field(..., description="분할 정보")

    # 성과 지표
    train_sharpe: float = Field(..., description="Train Sharpe Ratio")
    test_sharpe: float = Field(..., description="Test Sharpe Ratio")
    train_return: float = Field(..., description="Train 총 수익률 (%)")
    test_return: float = Field(..., description="Test 총 수익률 (%)")
    train_max_drawdown: float = Field(..., description="Train 최대 낙폭 (%)")
    test_max_drawdown: float = Field(..., description="Test 최대 낙폭 (%)")

    # 거래 통계
    train_trades: int = Field(default=0, ge=0, description="Train 거래 횟수")
    test_trades: int = Field(default=0, ge=0, description="Test 거래 횟수")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sharpe_decay(self) -> float:
        """Sharpe 감소율 ((Train - Test) / Train).

        높을수록 과적합 가능성이 높음.
        음수면 OOS가 더 좋은 성과 (드문 경우).
        """
        if self.train_sharpe == 0:
            return 0.0
        return (self.train_sharpe - self.test_sharpe) / abs(self.train_sharpe)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def return_decay(self) -> float:
        """수익률 감소율 ((Train - Test) / Train)."""
        if self.train_return == 0:
            return 0.0
        return (self.train_return - self.test_return) / abs(self.train_return)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_consistent(self) -> bool:
        """Train/Test 성과가 일관적인지 여부.

        Test Sharpe가 양수이고 Decay가 50% 미만이면 일관적.
        """
        max_acceptable_decay = 0.5  # 50%
        return self.test_sharpe > 0 and self.sharpe_decay < max_acceptable_decay


class MonteCarloResult(BaseModel):
    """Monte Carlo 시뮬레이션 결과.

    Returns 부트스트랩 기반 신뢰구간을 계산합니다.

    Attributes:
        n_simulations: 시뮬레이션 횟수
        sharpe_mean: Sharpe 평균
        sharpe_std: Sharpe 표준편차
        sharpe_percentiles: Sharpe 백분위수 {5, 25, 50, 75, 95}
        sharpe_ci_lower: 95% 신뢰구간 하한
        sharpe_ci_upper: 95% 신뢰구간 상한
        p_value: 랜덤 대비 유의성 (낮을수록 유의미)
    """

    model_config = ConfigDict(frozen=True)

    n_simulations: int = Field(..., ge=100, description="시뮬레이션 횟수")

    # 분포 통계
    sharpe_mean: float = Field(..., description="Sharpe 평균")
    sharpe_std: float = Field(..., ge=0, description="Sharpe 표준편차")
    sharpe_percentiles: dict[int, float] = Field(
        ..., description="Sharpe 백분위수 {5, 25, 50, 75, 95}"
    )

    # 신뢰 구간
    sharpe_ci_lower: float = Field(..., description="95% CI 하한")
    sharpe_ci_upper: float = Field(..., description="95% CI 상한")

    # 유의성
    p_value: float = Field(..., ge=0, le=1, description="P-value")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_significant(self) -> bool:
        """통계적으로 유의미한지 (p < 0.05)."""
        significance_level = 0.05
        return self.p_value < significance_level


class ValidationResult(BaseModel):
    """Tiered Validation 결과.

    검증 수행 결과를 종합적으로 저장합니다.

    Attributes:
        level: 검증 레벨
        fold_results: Fold별 결과 (QUICK, MILESTONE)
        monte_carlo: Monte Carlo 결과 (FINAL only)
        avg_train_sharpe: 평균 Train Sharpe
        avg_test_sharpe: 평균 Test Sharpe
        sharpe_stability: Test Sharpe 표준편차 (낮을수록 안정적)
        passed: 검증 통과 여부
        failure_reasons: 실패 사유 목록
        total_folds: 총 Fold 수
        computation_time_seconds: 계산 시간 (초)
        created_at: 생성 시각
    """

    model_config = ConfigDict(frozen=True)

    level: ValidationLevel = Field(..., description="검증 레벨")

    # Fold 결과 (QUICK, MILESTONE)
    fold_results: tuple[FoldResult, ...] = Field(default_factory=tuple, description="Fold별 결과")

    # Monte Carlo (FINAL only)
    monte_carlo: MonteCarloResult | None = Field(default=None, description="Monte Carlo 결과")

    # 요약 통계
    avg_train_sharpe: float = Field(..., description="평균 Train Sharpe")
    avg_test_sharpe: float = Field(..., description="평균 Test Sharpe")
    sharpe_stability: float = Field(..., ge=0, description="Test Sharpe 표준편차 (안정성)")

    # 합격/불합격
    passed: bool = Field(..., description="검증 통과 여부")
    failure_reasons: tuple[str, ...] = Field(default_factory=tuple, description="실패 사유")

    # 메타데이터
    total_folds: int = Field(..., ge=1, description="총 Fold 수")
    computation_time_seconds: float = Field(..., ge=0, description="계산 시간 (초)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="생성 시각",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_sharpe_decay(self) -> float:
        """평균 Sharpe 감소율."""
        if self.avg_train_sharpe == 0:
            return 0.0
        return (self.avg_train_sharpe - self.avg_test_sharpe) / abs(self.avg_train_sharpe)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def consistency_ratio(self) -> float:
        """일관성 비율 (일관적인 Fold / 전체 Fold)."""
        if not self.fold_results:
            return 0.0
        consistent_count = sum(1 for f in self.fold_results if f.is_consistent)
        return consistent_count / len(self.fold_results)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overfit_probability(self) -> float:
        """과적합 확률 추정 (0-1).

        Sharpe Decay와 일관성을 기반으로 계산.
        """
        decay_score = min(1.0, max(0.0, self.avg_sharpe_decay))
        consistency_score = 1.0 - self.consistency_ratio

        # 가중 평균 (Decay 60%, Consistency 40%)
        return 0.6 * decay_score + 0.4 * consistency_score

    @computed_field  # type: ignore[prop-decorator]
    @property
    def verdict(self) -> Literal["PASS", "WARN", "FAIL"]:
        """검증 판정."""
        warn_threshold = 0.3  # 30% 과적합 확률 이상이면 경고
        if self.passed:
            if self.overfit_probability > warn_threshold:
                return "WARN"
            return "PASS"
        return "FAIL"

    def summary(self) -> dict[str, str]:
        """요약 정보 반환."""
        return {
            "level": self.level.value,
            "verdict": self.verdict,
            "avg_train_sharpe": f"{self.avg_train_sharpe:.2f}",
            "avg_test_sharpe": f"{self.avg_test_sharpe:.2f}",
            "sharpe_decay": f"{self.avg_sharpe_decay:.1%}",
            "consistency": f"{self.consistency_ratio:.1%}",
            "overfit_probability": f"{self.overfit_probability:.1%}",
            "folds": str(self.total_folds),
            "time": f"{self.computation_time_seconds:.1f}s",
        }


# =============================================================================
# Pass/Fail 기준 상수
# =============================================================================

# Quick Validation 기준
QUICK_MIN_OOS_SHARPE = 0.3  # OOS Sharpe 최소값
QUICK_MAX_SHARPE_DECAY = 0.5  # Sharpe Decay 최대값 (50%)

# Milestone Validation 기준
MILESTONE_MIN_OOS_SHARPE = 0.5
MILESTONE_MAX_SHARPE_DECAY = 0.4
MILESTONE_MIN_CONSISTENCY = 0.6  # 60% 이상 Fold가 일관적

# Final Validation 기준
FINAL_MIN_OOS_SHARPE = 0.7
FINAL_MAX_SHARPE_DECAY = 0.3
FINAL_MIN_CONSISTENCY = 0.8
FINAL_MAX_P_VALUE = 0.05  # 통계적 유의성
