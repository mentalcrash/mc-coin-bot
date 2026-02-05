"""Advisor result models.

Strategy Advisor의 분석 결과를 표현하는 Pydantic 모델을 정의합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, computed_field
    - #10 Python Standards: Modern typing (X | None, list[])
"""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# 세션 시간대 정의 (UTC)
_ASIA_SESSION_START = 0
_ASIA_SESSION_END = 9  # exclusive
_EUROPE_SESSION_START = 8
_EUROPE_SESSION_END = 17  # exclusive
_US_SESSION_START = 16
_US_SESSION_END = 24  # exclusive

# 과적합 위험 수준 임계값
_RISK_LOW_THRESHOLD = 0.2
_RISK_MODERATE_THRESHOLD = 0.4
_RISK_HIGH_THRESHOLD = 0.6


class LossConcentration(BaseModel):
    """손실 집중 분석 결과.

    손실이 특정 시간대, 요일, 패턴에 집중되는지 분석합니다.

    Attributes:
        hourly_pnl: 시간대별 PnL {hour: pnl}
        worst_hours: 손실 집중 시간대 top3
        weekday_pnl: 요일별 PnL {0=Mon, ...}
        worst_weekdays: 손실 집중 요일 top2
        max_consecutive_losses: 최대 연속 손실 횟수
        avg_consecutive_losses: 평균 연속 손실 횟수
        large_loss_threshold: 대규모 손실 임계값 (%)
        large_loss_count: 대규모 손실 횟수
        large_loss_total: 대규모 손실 합계 (%)
    """

    model_config = ConfigDict(frozen=True)

    # 시간대별 손실
    hourly_pnl: dict[int, float] = Field(..., description="시간대별 PnL (%)")
    worst_hours: tuple[int, ...] = Field(..., description="손실 집중 시간대 top3")

    # 요일별 손실
    weekday_pnl: dict[int, float] = Field(..., description="요일별 PnL (%)")
    worst_weekdays: tuple[int, ...] = Field(..., description="손실 집중 요일 top2")

    # 연속 손실 패턴
    max_consecutive_losses: int = Field(..., ge=0, description="최대 연속 손실")
    avg_consecutive_losses: float = Field(..., ge=0, description="평균 연속 손실")

    # 대규모 손실
    large_loss_threshold: float = Field(..., description="대규모 손실 임계값 (%)")
    large_loss_count: int = Field(..., ge=0, description="대규모 손실 횟수")
    large_loss_total: float = Field(..., description="대규모 손실 합계 (%)")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def asia_session_pnl(self) -> float:
        """아시아 세션 (00:00-08:00 UTC) PnL."""
        return sum(self.hourly_pnl.get(h, 0) for h in range(_ASIA_SESSION_START, _ASIA_SESSION_END))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def europe_session_pnl(self) -> float:
        """유럽 세션 (08:00-16:00 UTC) PnL."""
        return sum(
            self.hourly_pnl.get(h, 0) for h in range(_EUROPE_SESSION_START, _EUROPE_SESSION_END)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def us_session_pnl(self) -> float:
        """미국 세션 (16:00-24:00 UTC) PnL."""
        return sum(self.hourly_pnl.get(h, 0) for h in range(_US_SESSION_START, _US_SESSION_END))


class RegimeProfile(BaseModel):
    """레짐별 성과 프로파일.

    다양한 시장 레짐에서의 전략 성과를 분석합니다.

    Attributes:
        bull_sharpe: Bull 시장 Sharpe
        bear_sharpe: Bear 시장 Sharpe
        sideways_sharpe: Sideways 시장 Sharpe
        high_vol_sharpe: 고변동성 Sharpe
        low_vol_sharpe: 저변동성 Sharpe
        regime_distribution: 레짐 분포 {regime: pct}
        weakest_regime: 가장 약한 레짐
    """

    model_config = ConfigDict(frozen=True)

    # 추세 레짐별 Sharpe
    bull_sharpe: float = Field(..., description="Bull 시장 Sharpe")
    bear_sharpe: float = Field(..., description="Bear 시장 Sharpe")
    sideways_sharpe: float = Field(..., description="Sideways 시장 Sharpe")

    # 변동성 레짐별 Sharpe
    high_vol_sharpe: float = Field(..., description="고변동성 Sharpe")
    low_vol_sharpe: float = Field(..., description="저변동성 Sharpe")

    # 레짐 분포
    regime_distribution: dict[str, float] = Field(..., description="레짐 분포 (%)")

    # 가장 약한 레짐
    weakest_regime: str = Field(..., description="가장 약한 레짐")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def regime_spread(self) -> float:
        """레짐 간 Sharpe 편차 (최고 - 최저)."""
        sharpes = [self.bull_sharpe, self.bear_sharpe, self.sideways_sharpe]
        return max(sharpes) - min(sharpes)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_regime_dependent(self) -> bool:
        """레짐 의존적인지 여부 (편차 > 1.0)."""
        regime_spread_threshold = 1.0
        return self.regime_spread > regime_spread_threshold


class SignalQuality(BaseModel):
    """시그널 품질 분석.

    시그널의 예측력과 효율성을 분석합니다.

    Attributes:
        hit_rate: 방향 적중률 (%)
        profit_factor: 수익 팩터
        avg_win: 평균 수익 (%)
        avg_loss: 평균 손실 (%)
        risk_reward_ratio: 손익비
        avg_holding_periods: 평균 보유 기간
        signal_count: 총 시그널 수
        trade_count: 실제 거래 수
        signal_efficiency: 시그널 효율 (trade/signal)
    """

    model_config = ConfigDict(frozen=True)

    # 기본 통계
    hit_rate: float = Field(..., ge=0, le=100, description="방향 적중률 (%)")
    profit_factor: float | None = Field(None, description="수익 팩터")

    # 손익
    avg_win: float = Field(..., description="평균 수익 (%)")
    avg_loss: float = Field(..., description="평균 손실 (%)")
    risk_reward_ratio: float | None = Field(None, description="손익비")

    # 보유 기간
    avg_holding_periods: float = Field(..., ge=0, description="평균 보유 기간")
    holding_distribution: dict[str, int] = Field(..., description="보유 기간 분포")

    # 시그널 효율
    signal_count: int = Field(..., ge=0, description="총 시그널 수")
    trade_count: int = Field(..., ge=0, description="실제 거래 수")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def signal_efficiency(self) -> float:
        """시그널 효율 (trade / signal)."""
        if self.signal_count == 0:
            return 0.0
        return self.trade_count / self.signal_count

    @computed_field  # type: ignore[prop-decorator]
    @property
    def expectancy(self) -> float:
        """기대값 (평균 거래당 수익)."""
        win_rate = self.hit_rate / 100
        return (win_rate * self.avg_win) + ((1 - win_rate) * self.avg_loss)


class OverfitScore(BaseModel):
    """Overfitting 스코어.

    전략의 과적합 정도를 정량화합니다.

    Attributes:
        is_sharpe: In-Sample Sharpe
        oos_sharpe: Out-of-Sample Sharpe
        sharpe_decay: Sharpe 감소율
        is_return: IS 총 수익률 (%)
        oos_return: OOS 총 수익률 (%)
        return_decay: 수익률 감소율
        overfit_probability: 과적합 확률 (0-1)
        parameter_sensitivity: 파라미터 민감도 (0-1)
    """

    model_config = ConfigDict(frozen=True)

    # IS/OOS 비교
    is_sharpe: float = Field(..., description="In-Sample Sharpe")
    oos_sharpe: float = Field(..., description="Out-of-Sample Sharpe")
    is_return: float = Field(..., description="IS 총 수익률 (%)")
    oos_return: float = Field(..., description="OOS 총 수익률 (%)")

    # 과적합 지표
    overfit_probability: float = Field(..., ge=0, le=1, description="과적합 확률")
    parameter_sensitivity: float | None = Field(None, ge=0, le=1, description="파라미터 민감도")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sharpe_decay(self) -> float:
        """Sharpe 감소율 ((IS - OOS) / |IS|)."""
        if self.is_sharpe == 0:
            return 0.0
        return (self.is_sharpe - self.oos_sharpe) / abs(self.is_sharpe)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def return_decay(self) -> float:
        """수익률 감소율 ((IS - OOS) / |IS|)."""
        if self.is_return == 0:
            return 0.0
        return (self.is_return - self.oos_return) / abs(self.is_return)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def risk_level(self) -> Literal["low", "moderate", "high", "critical"]:
        """과적합 위험 수준."""
        prob = self.overfit_probability
        if prob < _RISK_LOW_THRESHOLD:
            return "low"
        if prob < _RISK_MODERATE_THRESHOLD:
            return "moderate"
        if prob < _RISK_HIGH_THRESHOLD:
            return "high"
        return "critical"


class ImprovementSuggestion(BaseModel):
    """개선 제안.

    전략 개선을 위한 구체적인 제안을 저장합니다.

    Attributes:
        priority: 우선순위 (high/medium/low)
        category: 카테고리 (signal/risk/execution/data)
        title: 제안 제목
        description: 상세 설명
        expected_impact: 예상 영향
        implementation_hint: 구현 힌트 (선택적)
    """

    model_config = ConfigDict(frozen=True)

    priority: Literal["high", "medium", "low"] = Field(..., description="우선순위")
    category: Literal["signal", "risk", "execution", "data"] = Field(..., description="카테고리")
    title: str = Field(..., min_length=1, description="제안 제목")
    description: str = Field(..., min_length=1, description="상세 설명")
    expected_impact: str = Field(..., description="예상 영향")
    implementation_hint: str | None = Field(None, description="구현 힌트")


class AdvisorReport(BaseModel):
    """Strategy Advisor 종합 리포트.

    모든 분석 결과와 개선 제안을 통합합니다.

    Attributes:
        loss_concentration: 손실 집중 분석
        regime_profile: 레짐 프로파일
        signal_quality: 시그널 품질
        overfit_score: 과적합 스코어 (Validation 있을 때만)
        suggestions: 개선 제안 목록
        overall_score: 종합 점수 (0-100)
        readiness_level: 준비 수준
        created_at: 생성 시각
        strategy_name: 전략 이름
    """

    model_config = ConfigDict(frozen=True)

    # 분석 결과
    loss_concentration: LossConcentration = Field(..., description="손실 집중 분석")
    regime_profile: RegimeProfile = Field(..., description="레짐 프로파일")
    signal_quality: SignalQuality = Field(..., description="시그널 품질")
    overfit_score: OverfitScore | None = Field(
        None, description="과적합 스코어 (Validation 있을 때만)"
    )

    # 개선 제안
    suggestions: tuple[ImprovementSuggestion, ...] = Field(
        default_factory=tuple, description="개선 제안"
    )

    # 종합 평가
    overall_score: float = Field(..., ge=0, le=100, description="종합 점수")
    readiness_level: Literal["development", "testing", "production"] = Field(
        ..., description="준비 수준"
    )

    # 메타데이터
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="생성 시각",
    )
    strategy_name: str = Field(..., description="전략 이름")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_priority_suggestions(self) -> int:
        """High 우선순위 제안 수."""
        return sum(1 for s in self.suggestions if s.priority == "high")

    def summary(self) -> dict[str, str]:
        """요약 정보 반환."""
        return {
            "strategy": self.strategy_name,
            "overall_score": f"{self.overall_score:.0f}/100",
            "readiness": self.readiness_level,
            "suggestions": str(len(self.suggestions)),
            "high_priority": str(self.high_priority_suggestions),
            "weakest_regime": self.regime_profile.weakest_regime,
            "hit_rate": f"{self.signal_quality.hit_rate:.1f}%",
        }
