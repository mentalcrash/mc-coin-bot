"""Suggestion engine for strategy improvements.

분석 결과를 기반으로 구체적인 개선 제안을 생성합니다.

Rules Applied:
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from src.backtest.advisor.models import (
    ImprovementSuggestion,
    LossConcentration,
    OverfitScore,
    RegimeProfile,
    SignalQuality,
)


class SuggestionEngine:
    """개선 제안 생성기.

    분석 결과를 기반으로 구체적인 개선 제안을 생성합니다.

    Example:
        >>> engine = SuggestionEngine()
        >>> suggestions = engine.generate(
        ...     loss=loss_concentration,
        ...     regime=regime_profile,
        ...     signal=signal_quality,
        ...     overfit=overfit_score,
        ... )
    """

    def generate(
        self,
        loss: LossConcentration,
        regime: RegimeProfile,
        signal: SignalQuality,
        overfit: OverfitScore | None = None,
    ) -> tuple[ImprovementSuggestion, ...]:
        """개선 제안 생성.

        Args:
            loss: 손실 집중 분석 결과
            regime: 레짐 프로파일
            signal: 시그널 품질
            overfit: 과적합 스코어 (선택적)

        Returns:
            개선 제안 목록
        """
        suggestions: list[ImprovementSuggestion] = []

        # 손실 집중 관련 제안
        suggestions.extend(self._analyze_loss_patterns(loss))

        # 레짐 관련 제안
        suggestions.extend(self._analyze_regime_patterns(regime))

        # 시그널 품질 관련 제안
        suggestions.extend(self._analyze_signal_quality(signal))

        # 과적합 관련 제안
        if overfit:
            suggestions.extend(self._analyze_overfit_risk(overfit))

        # 우선순위로 정렬
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_order[s.priority])

        return tuple(suggestions)

    def _analyze_loss_patterns(
        self,
        loss: LossConcentration,
    ) -> list[ImprovementSuggestion]:
        """손실 패턴 분석 기반 제안."""
        suggestions: list[ImprovementSuggestion] = []

        # 아시아 세션 손실 집중
        total_pnl = sum(loss.hourly_pnl.values())
        if total_pnl != 0:
            asia_ratio = abs(loss.asia_session_pnl / total_pnl) if loss.asia_session_pnl < 0 else 0

            threshold_high = 0.5  # 50%
            threshold_medium = 0.3  # 30%

            if asia_ratio > threshold_high:
                suggestions.append(
                    ImprovementSuggestion(
                        priority="high",
                        category="execution",
                        title="아시아 세션 필터 추가",
                        description=f"손실의 {asia_ratio:.0%}가 아시아 세션(00:00-08:00 UTC)에 집중됨",
                        expected_impact="MDD 감소, 안정성 향상",
                        implementation_hint="session_filter 파라미터를 추가하여 특정 시간대 거래 제외",
                    )
                )
            elif asia_ratio > threshold_medium:
                suggestions.append(
                    ImprovementSuggestion(
                        priority="medium",
                        category="execution",
                        title="아시아 세션 포지션 축소 고려",
                        description=f"손실의 {asia_ratio:.0%}가 아시아 세션에 집중됨",
                        expected_impact="위험 조정 수익 개선",
                        implementation_hint="아시아 세션에서 포지션 사이즈 50% 감소",
                    )
                )

        # 연속 손실 패턴
        max_loss_threshold = 5
        if loss.max_consecutive_losses > max_loss_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="risk",
                    title="연속 손실 리미터 추가",
                    description=f"최대 {loss.max_consecutive_losses}회 연속 손실 발생",
                    expected_impact="극단적 손실 방지",
                    implementation_hint="N회 연속 손실 시 일시 중단 로직 추가",
                )
            )

        # 대규모 손실
        large_loss_threshold = 5
        if loss.large_loss_count > large_loss_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="high",
                    category="risk",
                    title="손절 로직 강화",
                    description=f"{loss.large_loss_threshold:.1f}% 이상 손실이 {loss.large_loss_count}회 발생",
                    expected_impact="MDD 대폭 감소",
                    implementation_hint="system_stop_loss 파라미터 조정 또는 trailing stop 추가",
                )
            )

        return suggestions

    def _analyze_regime_patterns(
        self,
        regime: RegimeProfile,
    ) -> list[ImprovementSuggestion]:
        """레짐 패턴 분석 기반 제안."""
        suggestions: list[ImprovementSuggestion] = []

        # 횡보장 성과 부진
        sideways_threshold = 0.0
        if regime.sideways_sharpe < sideways_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="high",
                    category="signal",
                    title="횡보장 레짐 필터 추가",
                    description=f"횡보장 Sharpe가 {regime.sideways_sharpe:.2f}로 부진함",
                    expected_impact="Whipsaw 감소, 전체 Sharpe 향상",
                    implementation_hint="ADX 또는 Bollinger Band Width 기반 레짐 필터",
                )
            )

        # 레짐 의존성
        if regime.is_regime_dependent:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="signal",
                    title="레짐 적응형 파라미터 고려",
                    description=f"레짐 간 Sharpe 편차가 {regime.regime_spread:.2f}로 큼",
                    expected_impact="레짐별 성과 균등화",
                    implementation_hint="레짐별로 다른 lookback 또는 vol_target 사용",
                )
            )

        # 하락장 성과 부진
        bear_threshold = 0.0
        if regime.bear_sharpe < bear_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="signal",
                    title="숏 전략 강화 또는 현금 보유",
                    description=f"하락장 Sharpe가 {regime.bear_sharpe:.2f}로 부진함",
                    expected_impact="하락장 방어력 향상",
                    implementation_hint="short_mode를 'full'로 변경하거나 하락장 감지 시 현금화",
                )
            )

        return suggestions

    def _analyze_signal_quality(
        self,
        signal: SignalQuality,
    ) -> list[ImprovementSuggestion]:
        """시그널 품질 분석 기반 제안."""
        suggestions: list[ImprovementSuggestion] = []

        # 낮은 적중률
        hit_rate_threshold = 45.0
        if signal.hit_rate < hit_rate_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="high",
                    category="signal",
                    title="시그널 생성 로직 재검토",
                    description=f"적중률이 {signal.hit_rate:.1f}%로 낮음",
                    expected_impact="거래당 기대값 향상",
                    implementation_hint="lookback 기간 조정 또는 확인 시그널 추가",
                )
            )

        # 낮은 손익비
        rr_threshold = 1.0
        if signal.risk_reward_ratio is not None and signal.risk_reward_ratio < rr_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="risk",
                    title="손익비 개선 필요",
                    description=f"손익비가 {signal.risk_reward_ratio:.2f}로 1.0 미만",
                    expected_impact="수익성 개선",
                    implementation_hint="익절 타이밍 조정 또는 손절 강화",
                )
            )

        # 과도한 거래
        efficiency_threshold = 0.1
        if signal.signal_efficiency < efficiency_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="low",
                    category="execution",
                    title="거래 빈도 최적화",
                    description=f"시그널 효율이 {signal.signal_efficiency:.1%}로 낮음",
                    expected_impact="거래 비용 절감",
                    implementation_hint="rebalance_threshold 조정으로 불필요한 거래 감소",
                )
            )

        return suggestions

    def _analyze_overfit_risk(
        self,
        overfit: OverfitScore,
    ) -> list[ImprovementSuggestion]:
        """과적합 위험 분석 기반 제안."""
        suggestions: list[ImprovementSuggestion] = []

        # 높은 과적합 확률
        high_risk_threshold = 0.5
        moderate_risk_threshold = 0.3

        if overfit.overfit_probability > high_risk_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="high",
                    category="data",
                    title="과적합 위험 높음 - 전략 단순화 필요",
                    description=f"과적합 확률이 {overfit.overfit_probability:.0%}로 높음",
                    expected_impact="OOS 성과 안정화",
                    implementation_hint="lookback 기간 증가, 파라미터 수 감소, 더 많은 데이터로 검증",
                )
            )
        elif overfit.overfit_probability > moderate_risk_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="data",
                    title="과적합 위험 중간 - 모니터링 필요",
                    description=f"과적합 확률이 {overfit.overfit_probability:.0%}",
                    expected_impact="안정적인 실전 성과",
                    implementation_hint="Milestone 또는 Final 검증으로 추가 확인",
                )
            )

        # Sharpe Decay가 큰 경우
        decay_threshold = 0.4
        if overfit.sharpe_decay > decay_threshold:
            suggestions.append(
                ImprovementSuggestion(
                    priority="medium",
                    category="data",
                    title="IS/OOS 성과 차이 큼",
                    description=f"Sharpe가 IS→OOS에서 {overfit.sharpe_decay:.0%} 감소",
                    expected_impact="실전 성과 예측 신뢰도 향상",
                    implementation_hint="Walk-Forward 최적화 또는 앙상블 전략 고려",
                )
            )

        return suggestions
