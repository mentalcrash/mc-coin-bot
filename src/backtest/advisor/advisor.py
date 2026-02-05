"""Strategy Advisor implementation.

전략 빌드업을 위한 코치/가이드 역할을 수행합니다.

Rules Applied:
    - #10 Python Standards: Modern typing
    - #26 VectorBT Standards: Stateless design
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.backtest.advisor.analyzers.loss import LossAnalyzer
from src.backtest.advisor.analyzers.overfit import OverfitAnalyzer
from src.backtest.advisor.analyzers.regime import RegimeAnalyzer
from src.backtest.advisor.analyzers.signal import SignalAnalyzer
from src.backtest.advisor.models import AdvisorReport
from src.backtest.advisor.suggestions import SuggestionEngine

if TYPE_CHECKING:
    import pandas as pd

    from src.backtest.validation.models import ValidationResult
    from src.models.backtest import BacktestResult

# 종합 점수 계산 상수 - Sharpe Ratio 기준
_SHARPE_EXCELLENT = 2.0
_SHARPE_GOOD = 1.5
_SHARPE_FAIR = 1.0
_SHARPE_POOR = 0.5
_SHARPE_EXCELLENT_SCORE = 25
_SHARPE_GOOD_SCORE = 20
_SHARPE_FAIR_SCORE = 15
_SHARPE_POOR_SCORE = 10
_SHARPE_WEAK_SCORE = 5
_SHARPE_NEGATIVE_PENALTY = 10

# MDD 기준 (%)
_MDD_EXCELLENT = -10
_MDD_GOOD = -20
_MDD_FAIR = -30
_MDD_POOR = -40
_MDD_EXCELLENT_SCORE = 15
_MDD_GOOD_SCORE = 10
_MDD_FAIR_SCORE = 5
_MDD_POOR_PENALTY = 10

# Win Rate 기준 (%)
_WINRATE_EXCELLENT = 55
_WINRATE_GOOD = 50
_WINRATE_FAIR = 45
_WINRATE_POOR = 40
_WINRATE_EXCELLENT_SCORE = 10
_WINRATE_GOOD_SCORE = 7
_WINRATE_FAIR_SCORE = 5
_WINRATE_POOR_SCORE = 2

# 과적합 감점
_OVERFIT_PENALTY_MULTIPLIER = 20

# 준비 수준 임계값
_HIGH_PRIORITY_DEVELOPMENT_THRESHOLD = 2
_OVERFIT_DEVELOPMENT_THRESHOLD = 0.5
_PRODUCTION_SCORE_THRESHOLD = 75
_TESTING_SCORE_THRESHOLD = 55
_BASE_SCORE = 50.0


class StrategyAdvisor:
    """전략 코치/가이드.

    백테스트 결과를 분석하고 개선 방향을 제안합니다.
    Stateless 설계를 따릅니다.

    Example:
        >>> advisor = StrategyAdvisor()
        >>> report = advisor.analyze(
        ...     result=backtest_result,
        ...     returns=strategy_returns,
        ...     benchmark_returns=benchmark_returns,
        ...     validation_result=validation_result,
        ... )
        >>> print(report.summary())
    """

    def __init__(
        self,
        loss_analyzer: LossAnalyzer | None = None,
        regime_analyzer: RegimeAnalyzer | None = None,
        signal_analyzer: SignalAnalyzer | None = None,
        overfit_analyzer: OverfitAnalyzer | None = None,
        suggestion_engine: SuggestionEngine | None = None,
    ) -> None:
        """StrategyAdvisor 초기화.

        Args:
            loss_analyzer: 손실 분석기 (None이면 기본값)
            regime_analyzer: 레짐 분석기 (None이면 기본값)
            signal_analyzer: 시그널 분석기 (None이면 기본값)
            overfit_analyzer: 과적합 분석기 (None이면 기본값)
            suggestion_engine: 제안 생성기 (None이면 기본값)
        """
        self._loss = loss_analyzer or LossAnalyzer()
        self._regime = regime_analyzer or RegimeAnalyzer()
        self._signal = signal_analyzer or SignalAnalyzer()
        self._overfit = overfit_analyzer or OverfitAnalyzer()
        self._suggestions = suggestion_engine or SuggestionEngine()

    def analyze(
        self,
        result: BacktestResult,
        returns: pd.Series,  # type: ignore[type-arg]
        benchmark_returns: pd.Series,  # type: ignore[type-arg]
        validation_result: ValidationResult | None = None,
    ) -> AdvisorReport:
        """종합 분석 수행.

        Args:
            result: 백테스트 결과
            returns: 전략 수익률 시리즈
            benchmark_returns: 벤치마크 수익률 시리즈
            validation_result: Tiered Validation 결과 (선택적)

        Returns:
            AdvisorReport 종합 리포트
        """
        # 개별 분석 수행
        loss_concentration = self._loss.analyze(result, returns)
        regime_profile = self._regime.analyze(result, returns, benchmark_returns)
        signal_quality = self._signal.analyze(result, returns)

        # 과적합 분석 (Validation 결과가 있을 때만)
        overfit_score = None
        if validation_result:
            overfit_score = self._overfit.analyze(validation_result)

        # 개선 제안 생성
        suggestions = self._suggestions.generate(
            loss=loss_concentration,
            regime=regime_profile,
            signal=signal_quality,
            overfit=overfit_score,
        )

        # 종합 점수 계산
        overall_score = self._calculate_overall_score(
            result=result,
            loss=loss_concentration,
            regime=regime_profile,
            signal=signal_quality,
            overfit=overfit_score,
        )

        # 준비 수준 결정
        readiness_level = self._determine_readiness(
            overall_score=overall_score,
            suggestions=suggestions,
            overfit=overfit_score,
        )

        return AdvisorReport(
            loss_concentration=loss_concentration,
            regime_profile=regime_profile,
            signal_quality=signal_quality,
            overfit_score=overfit_score,
            suggestions=suggestions,
            overall_score=overall_score,
            readiness_level=readiness_level,
            strategy_name=result.config.strategy_name,
        )

    def _calculate_overall_score(
        self,
        result: BacktestResult,
        loss: object,
        regime: object,
        signal: object,
        overfit: object | None,
    ) -> float:
        """종합 점수 계산 (0-100).

        여러 지표를 종합하여 전략의 전반적인 품질을 점수화합니다.
        """
        score = _BASE_SCORE
        score += self._score_sharpe(result.metrics.sharpe_ratio)
        score += self._score_mdd(result.metrics.max_drawdown)
        score += self._score_winrate(result.metrics.win_rate)
        score += self._score_overfit(overfit)
        return max(0.0, min(100.0, score))

    def _score_sharpe(self, sharpe: float) -> float:
        """Sharpe Ratio 기반 점수 (최대 +25, 최소 -10)."""
        thresholds = [
            (_SHARPE_EXCELLENT, _SHARPE_EXCELLENT_SCORE),
            (_SHARPE_GOOD, _SHARPE_GOOD_SCORE),
            (_SHARPE_FAIR, _SHARPE_FAIR_SCORE),
            (_SHARPE_POOR, _SHARPE_POOR_SCORE),
            (0, _SHARPE_WEAK_SCORE),
        ]
        for threshold, score in thresholds:
            if sharpe > threshold:
                return score
        return -_SHARPE_NEGATIVE_PENALTY

    def _score_mdd(self, mdd: float) -> float:
        """MDD 기반 점수 (최대 +15, 최소 -10)."""
        thresholds = [
            (_MDD_EXCELLENT, _MDD_EXCELLENT_SCORE),
            (_MDD_GOOD, _MDD_GOOD_SCORE),
            (_MDD_FAIR, _MDD_FAIR_SCORE),
            (_MDD_POOR, 0),
        ]
        for threshold, score in thresholds:
            if mdd > threshold:
                return score
        return -_MDD_POOR_PENALTY

    def _score_winrate(self, win_rate: float) -> float:
        """Win Rate 기반 점수 (최대 +10)."""
        thresholds = [
            (_WINRATE_EXCELLENT, _WINRATE_EXCELLENT_SCORE),
            (_WINRATE_GOOD, _WINRATE_GOOD_SCORE),
            (_WINRATE_FAIR, _WINRATE_FAIR_SCORE),
            (_WINRATE_POOR, _WINRATE_POOR_SCORE),
        ]
        for threshold, score in thresholds:
            if win_rate > threshold:
                return score
        return 0

    def _score_overfit(self, overfit: object | None) -> float:
        """과적합 위험 감점 (최대 -20)."""
        if overfit is None:
            return 0
        from src.backtest.advisor.models import OverfitScore

        if isinstance(overfit, OverfitScore):
            return -overfit.overfit_probability * _OVERFIT_PENALTY_MULTIPLIER
        return 0

    def _determine_readiness(
        self,
        overall_score: float,
        suggestions: tuple,  # type: ignore[type-arg]
        overfit: object | None,
    ) -> Literal["development", "testing", "production"]:
        """준비 수준 결정.

        Args:
            overall_score: 종합 점수
            suggestions: 개선 제안 목록
            overfit: 과적합 스코어

        Returns:
            준비 수준 ("development", "testing", "production")
        """
        # High priority 제안이 있으면 development
        from src.backtest.advisor.models import ImprovementSuggestion

        high_priority_count = sum(
            1 for s in suggestions if isinstance(s, ImprovementSuggestion) and s.priority == "high"
        )

        if high_priority_count >= _HIGH_PRIORITY_DEVELOPMENT_THRESHOLD:
            return "development"

        # 과적합 위험이 높으면 development
        if overfit is not None:
            from src.backtest.advisor.models import OverfitScore

            if (
                isinstance(overfit, OverfitScore)
                and overfit.overfit_probability > _OVERFIT_DEVELOPMENT_THRESHOLD
            ):
                return "development"

        # 점수 기반 판단
        if overall_score >= _PRODUCTION_SCORE_THRESHOLD:
            return "production"
        if overall_score >= _TESTING_SCORE_THRESHOLD:
            return "testing"
        return "development"
