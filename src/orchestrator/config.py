"""Orchestrator Configuration Models.

Strategy Orchestrator의 Pod, 졸업/퇴출 기준, 포트폴리오 설정을 정의합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, Field validators
    - #10 Python Standards: Modern typing, StrEnum
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.orchestrator.models import AllocationMethod, RebalanceTrigger


class PodConfig(BaseModel):
    """개별 Pod(전략 슬롯) 설정.

    하나의 전략 인스턴스가 운용되는 슬롯의 식별, 자본 배분, 리스크 한도를 정의합니다.

    Attributes:
        pod_id: Pod 고유 식별자
        strategy_name: Registry에 등록된 전략 이름
        strategy_params: 전략 생성 파라미터
        symbols: 거래 대상 심볼 목록
        timeframe: 타임프레임
        initial_fraction: 초기 자본 비중
        max_fraction: 최대 자본 비중
        min_fraction: 최소 자본 비중
        max_drawdown: 최대 허용 낙폭
        drawdown_warning: 낙폭 경고 임계값
        max_leverage: 최대 레버리지
        system_stop_loss: 시스템 손절 비율 (None=비활성)
        use_trailing_stop: 트레일링 스탑 활성화
        trailing_stop_atr_multiplier: 트레일링 스탑 ATR 배수
        rebalance_threshold: 리밸런스 이탈 임계값
    """

    model_config = ConfigDict(frozen=True)

    # Identification
    pod_id: str = Field(description="Pod 고유 식별자")
    strategy_name: str = Field(description="Registry에 등록된 전략 이름")
    strategy_params: dict[str, Any] = Field(
        default_factory=dict,
        description="전략 생성 파라미터",
    )
    symbols: tuple[str, ...] = Field(
        min_length=1,
        description="거래 대상 심볼 목록",
    )
    timeframe: str = Field(default="1D", description="타임프레임")

    # Capital allocation
    initial_fraction: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="초기 자본 비중",
    )
    max_fraction: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="최대 자본 비중",
    )
    min_fraction: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="최소 자본 비중",
    )

    # Risk limits
    max_drawdown: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="최대 허용 낙폭",
    )
    drawdown_warning: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="낙폭 경고 임계값",
    )
    max_leverage: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="최대 레버리지",
    )

    # PM parameters
    system_stop_loss: float | None = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="시스템 손절 비율 (None=비활성)",
    )
    use_trailing_stop: bool = Field(
        default=False,
        description="트레일링 스탑 활성화",
    )
    trailing_stop_atr_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="트레일링 스탑 ATR 배수",
    )
    rebalance_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="리밸런스 이탈 임계값",
    )

    @model_validator(mode="after")
    def validate_fractions_and_drawdown(self) -> Self:
        """자본 비중 및 낙폭 일관성 검증.

        Raises:
            ValueError: min > initial, initial > max, 또는 warning >= max_drawdown
        """
        if self.min_fraction > self.initial_fraction:
            msg = (
                f"min_fraction ({self.min_fraction}) cannot exceed "
                f"initial_fraction ({self.initial_fraction})"
            )
            raise ValueError(msg)
        if self.initial_fraction > self.max_fraction:
            msg = (
                f"initial_fraction ({self.initial_fraction}) cannot exceed "
                f"max_fraction ({self.max_fraction})"
            )
            raise ValueError(msg)
        if self.drawdown_warning >= self.max_drawdown:
            msg = (
                f"drawdown_warning ({self.drawdown_warning}) must be less than "
                f"max_drawdown ({self.max_drawdown})"
            )
            raise ValueError(msg)
        return self


class GraduationCriteria(BaseModel):
    """Incubation → Production 졸업 기준.

    모든 조건을 충족해야 Pod가 Production으로 승격됩니다.

    Attributes:
        min_live_days: 최소 실운용 일수
        min_sharpe: 최소 Sharpe 비율
        max_drawdown: 최대 낙폭 한도
        min_trade_count: 최소 거래 수
        min_calmar: 최소 Calmar 비율
        max_backtest_live_gap: 백테스트-실거래 성과 괴리 한도
        max_portfolio_correlation: 기존 포트폴리오와의 최대 상관계수
    """

    model_config = ConfigDict(frozen=True)

    min_live_days: int = Field(
        default=30,
        ge=1,
        description="최소 실운용 일수",
    )
    min_sharpe: float = Field(
        default=0.5,
        ge=0.0,
        description="최소 Sharpe 비율",
    )
    max_drawdown: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="최대 낙폭 한도",
    )
    min_trade_count: int = Field(
        default=5,
        ge=1,
        description="최소 거래 수",
    )
    min_calmar: float = Field(
        default=0.3,
        ge=0.0,
        description="최소 Calmar 비율",
    )
    max_backtest_live_gap: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="백테스트-실거래 성과 괴리 한도",
    )
    max_portfolio_correlation: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="기존 포트폴리오와의 최대 상관계수",
    )


class RetirementCriteria(BaseModel):
    """Pod 퇴출(Retired) 기준.

    Hard 기준은 즉시 퇴출, Soft 기준은 Probation 후 퇴출.

    Attributes:
        max_drawdown_breach: 즉시 퇴출 낙폭 임계값
        consecutive_loss_months: 연속 손실 개월 수
        rolling_sharpe_floor: Probation 진입 Sharpe 하한
        probation_days: Probation 유예 기간 (일)
    """

    model_config = ConfigDict(frozen=True)

    # Hard criteria — 즉시 퇴출
    max_drawdown_breach: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="즉시 퇴출 낙폭 임계값",
    )
    consecutive_loss_months: int = Field(
        default=6,
        ge=1,
        description="연속 손실 개월 수",
    )

    # Soft criteria — Probation 후 퇴출
    rolling_sharpe_floor: float = Field(
        default=0.3,
        ge=0.0,
        description="Probation 진입 Sharpe 하한",
    )
    probation_days: int = Field(
        default=30,
        ge=1,
        description="Probation 유예 기간 (일)",
    )


class OrchestratorConfig(BaseModel):
    """Strategy Orchestrator 전체 설정.

    여러 Pod을 동시 운용하며 자본을 동적으로 배분하는 프레임워크의 메인 설정입니다.

    Attributes:
        pods: Pod 설정 목록 (최소 1개)
        allocation_method: 자본 배분 방법
        kelly_fraction: Kelly 비율 스케일링 계수
        kelly_confidence_ramp: Kelly 신뢰도 램프업 기간 (일)
        rebalance_trigger: 리밸런싱 트리거 유형
        rebalance_calendar_days: 정기 리밸런싱 주기 (일)
        rebalance_drift_threshold: 가중치 이탈 리밸런싱 임계값
        max_portfolio_volatility: 포트폴리오 최대 변동성
        max_portfolio_drawdown: 포트폴리오 최대 낙폭
        max_gross_leverage: 최대 총 레버리지
        max_single_pod_risk_pct: 단일 Pod 최대 리스크 비중
        daily_loss_limit: 일일 손실 한도
        graduation: 졸업 기준
        retirement: 퇴출 기준
        correlation_lookback: 상관계수 계산 lookback (일)
        correlation_stress_threshold: 스트레스 상관계수 임계값
        cost_bps: 거래 비용 (bps)
    """

    model_config = ConfigDict(frozen=True)

    # Pods
    pods: tuple[PodConfig, ...] = Field(
        min_length=1,
        description="Pod 설정 목록 (최소 1개)",
    )

    # Allocation
    allocation_method: AllocationMethod = Field(
        default=AllocationMethod.RISK_PARITY,
        description="자본 배분 방법",
    )
    kelly_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Kelly 비율 스케일링 계수",
    )
    kelly_confidence_ramp: int = Field(
        default=180,
        ge=1,
        description="Kelly 신뢰도 램프업 기간 (일)",
    )

    # Rebalance
    rebalance_trigger: RebalanceTrigger = Field(
        default=RebalanceTrigger.HYBRID,
        description="리밸런싱 트리거 유형",
    )
    rebalance_calendar_days: int = Field(
        default=7,
        ge=1,
        description="정기 리밸런싱 주기 (일)",
    )
    rebalance_drift_threshold: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="가중치 이탈 리밸런싱 임계값",
    )

    # Portfolio risk
    max_portfolio_volatility: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description="포트폴리오 최대 변동성",
    )
    max_portfolio_drawdown: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="포트폴리오 최대 낙폭",
    )
    max_gross_leverage: float = Field(
        default=3.0,
        ge=1.0,
        le=20.0,
        description="최대 총 레버리지",
    )
    max_single_pod_risk_pct: float = Field(
        default=0.40,
        gt=0.0,
        le=1.0,
        description="단일 Pod 최대 리스크 비중",
    )
    daily_loss_limit: float = Field(
        default=0.03,
        gt=0.0,
        le=1.0,
        description="일일 손실 한도",
    )

    # Lifecycle
    graduation: GraduationCriteria = Field(
        default_factory=GraduationCriteria,
        description="졸업 기준",
    )
    retirement: RetirementCriteria = Field(
        default_factory=RetirementCriteria,
        description="퇴출 기준",
    )

    # Correlation
    correlation_lookback: int = Field(
        default=90,
        ge=5,
        description="상관계수 계산 lookback (일)",
    )
    correlation_stress_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="스트레스 상관계수 임계값",
    )

    # Cost
    cost_bps: float = Field(
        default=4.0,
        ge=0.0,
        description="거래 비용 (bps)",
    )

    @model_validator(mode="after")
    def validate_pods(self) -> Self:
        """Pod 설정 일관성 검증.

        Raises:
            ValueError: pod_id 중복 또는 initial_fraction 합계 > 1.0
        """
        pod_ids = [p.pod_id for p in self.pods]
        if len(pod_ids) != len(set(pod_ids)):
            duplicates = [pid for pid in pod_ids if pod_ids.count(pid) > 1]
            msg = f"Duplicate pod_id found: {sorted(set(duplicates))}"
            raise ValueError(msg)

        total_initial = sum(p.initial_fraction for p in self.pods)
        if total_initial > 1.0:
            msg = f"Sum of initial_fraction ({total_initial:.2f}) exceeds 1.0"
            raise ValueError(msg)

        _max_fraction_cap = 1.5
        total_max = sum(p.max_fraction for p in self.pods)
        if total_max > _max_fraction_cap:
            msg = f"Sum of max_fraction ({total_max:.2f}) exceeds {_max_fraction_cap}"
            raise ValueError(msg)

        return self

    @property
    def all_symbols(self) -> tuple[str, ...]:
        """모든 Pod의 심볼을 중복 제거하여 반환."""
        seen: set[str] = set()
        result: list[str] = []
        for pod in self.pods:
            for symbol in pod.symbols:
                if symbol not in seen:
                    seen.add(symbol)
                    result.append(symbol)
        return tuple(result)

    @property
    def all_timeframes(self) -> tuple[str, ...]:
        """모든 Pod의 고유 TF 목록 (순서 보존)."""
        seen: set[str] = set()
        result: list[str] = []
        for pod in self.pods:
            if pod.timeframe not in seen:
                seen.add(pod.timeframe)
                result.append(pod.timeframe)
        return tuple(result)

    @property
    def n_pods(self) -> int:
        """Pod 수."""
        return len(self.pods)
