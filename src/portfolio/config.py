"""Portfolio Manager Configuration.

이 모듈은 전략-독립적인 포트폴리오 집행 설정을 정의합니다.
전략이 출력한 target_weights를 어떻게 집행할지 결정하며,
모든 전략에 동일하게 적용할 수 있는 범용 설정입니다.

핵심 원칙:
    - 전략 책임: 시그널 생성, 변동성 타겟팅, target_weights 계산
    - PM 책임: 집행 규칙, 리스크 가드레일, 비용 적용
    - Engine 책임: freq (연환산), 리포팅

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing (Literal, Self)
    - #23 Exception Handling: 검증 실패 시 명확한 에러
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.portfolio.cost_model import CostModel

# 검증 임계값 상수
MIN_REBALANCE_THRESHOLD = 0.01  # 최소 리밸런싱 임계값 (1%)
MIN_REASONABLE_STOP_LOSS = 0.02  # 최소 합리적 손절 비율 (2%)


class PortfolioManagerConfig(BaseModel):
    """전략-독립적인 포트폴리오 집행 설정.

    전략이 출력한 target_weights를 어떻게 집행할지 결정합니다.
    모든 전략에 동일하게 적용할 수 있는 범용 설정입니다.

    Attributes:
        execution_mode: 실행 모드 (orders: 연속 리밸런싱, signals: 이벤트 기반)
        price_type: 체결 가격 타입 (close: 종가, next_open: 다음 시가)
        size_type: 포지션 사이징 방식
        upon_opposite_entry: 반대 포지션 진입 시 처리 방식
        accumulate: 포지션 누적 허용 여부
        rebalance_threshold: 리밸런싱 임계값 (비용 최적화)
        max_leverage_cap: 최대 레버리지 상한 (전략 요청과 무관)
        system_stop_loss: 시스템 레벨 손절 (최후의 방어선)
        use_trailing_stop: Trailing Stop 활성화 여부
        trailing_stop_atr_multiplier: Trailing Stop ATR 배수
        cash_sharing: 자산별 마진 격리 여부
        cost_model: 거래 비용 모델

    Example:
        >>> # VW-TSMOM: 연속 리밸런싱 모드 (기본값)
        >>> config = PortfolioManagerConfig(
        ...     execution_mode="orders",
        ...     max_leverage_cap=2.0,
        ... )
        >>> # 단순 전략: 이벤트 기반 모드
        >>> config = PortfolioManagerConfig(execution_mode="signals")
    """

    model_config = ConfigDict(frozen=True)

    # ==========================================================================
    # Execution Rules (집행 규칙)
    # ==========================================================================
    execution_mode: Literal["orders", "signals"] = Field(
        default="orders",
        description=(
            "백테스트 실행 모드: "
            "orders=연속 리밸런싱 (VW-TSMOM 등 동적 레버리지 전략), "
            "signals=이벤트 기반 (단순 entry/exit)"
        ),
    )
    price_type: Literal["close", "next_open"] = Field(
        default="next_open",
        description="체결 가격 타입 (next_open: Look-Ahead Bias 방지)",
    )
    size_type: Literal["targetpercent", "amount", "value"] = Field(
        default="targetpercent",
        description="포지션 사이징 방식 (targetpercent 권장)",
    )
    upon_opposite_entry: Literal["close", "reverse"] = Field(
        default="reverse",
        description="반대 포지션 진입 시 처리 (reverse: 직접 전환)",
    )
    accumulate: bool = Field(
        default=False,
        description="포지션 누적 허용 여부 (False: 단일 포지션만)",
    )

    # ==========================================================================
    # Rebalancing (리밸런싱)
    # ==========================================================================
    rebalance_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="리밸런싱 임계값 (0.05 = 5% 이상 차이 시 리밸런싱)",
    )

    # ==========================================================================
    # Risk Guardrails (전략 외부 안전장치)
    # ==========================================================================
    max_leverage_cap: float = Field(
        default=3.0,  # 3.0x가 적정 (4.0x와 동일한 결과)
        ge=0.5,
        le=10.0,
        description="최대 레버리지 상한 (전략 요청과 무관한 시스템 제한)",
    )
    system_stop_loss: float | None = Field(
        default=0.10,  # 10% Stop Loss (안전장치)
        ge=0.01,
        le=0.50,
        description="시스템 레벨 손절 (0.10 = 10%, None = 비활성화)",
    )

    # ==========================================================================
    # Trailing Stop (추적 손절)
    # ==========================================================================
    use_trailing_stop: bool = Field(
        default=False,
        description="Trailing Stop 활성화 여부",
    )
    trailing_stop_atr_multiplier: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Trailing Stop ATR 배수 (예: 2.0 = 2 ATR)",
    )

    # ==========================================================================
    # Capital Management (자금 관리)
    # ==========================================================================
    cash_sharing: bool = Field(
        default=False,
        description="자산 간 현금 공유 여부 (False: 격리 마진)",
    )

    # ==========================================================================
    # Cost Model (비용 모델)
    # ==========================================================================
    cost_model: CostModel = Field(
        default_factory=CostModel.binance_futures,
        description="거래 비용 모델 (수수료, 슬리피지 등)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정 값이 비합리적일 경우
        """
        # system_stop_loss가 설정된 경우 합리적인 범위인지 확인
        # 손절 비율이 너무 타이트하면 경고
        if self.system_stop_loss is not None and self.system_stop_loss < MIN_REASONABLE_STOP_LOSS:
            msg = (
                f"system_stop_loss ({self.system_stop_loss:.1%}) is very tight. "
                f"Consider >= {MIN_REASONABLE_STOP_LOSS:.1%} to avoid premature exits."
            )
            raise ValueError(msg)

        # rebalance_threshold가 너무 낮으면 거래 비용 증가 경고
        if self.rebalance_threshold < MIN_REBALANCE_THRESHOLD:
            msg = (
                f"rebalance_threshold ({self.rebalance_threshold:.1%}) is very low. "
                "This may cause excessive trading costs."
            )
            raise ValueError(msg)

        return self

    # ==========================================================================
    # Preset Methods (프리셋 메서드)
    # ==========================================================================
    @classmethod
    def default(cls) -> Self:
        """기본 설정.

        바이낸스 선물 기준 일반적인 설정입니다.

        Returns:
            기본 PortfolioManagerConfig
        """
        return cls()

    @classmethod
    def conservative(cls) -> Self:
        """보수적 설정.

        낮은 레버리지, 높은 리밸런싱 임계값, 타이트한 손절로
        리스크를 최소화하는 설정입니다.

        Returns:
            보수적 PortfolioManagerConfig
        """
        return cls(
            max_leverage_cap=1.5,
            rebalance_threshold=0.10,  # 10%
            system_stop_loss=0.05,  # 5%
            cost_model=CostModel.conservative(),
        )

    @classmethod
    def aggressive(cls) -> Self:
        """공격적 설정.

        높은 레버리지, 빠른 리밸런싱, 관대한 손절로
        수익을 극대화하는 설정입니다.

        Returns:
            공격적 PortfolioManagerConfig
        """
        return cls(
            max_leverage_cap=5.0,
            rebalance_threshold=0.02,  # 2%
            system_stop_loss=0.15,  # 15%
            cost_model=CostModel.optimistic(),
        )

    @classmethod
    def paper_trading(cls) -> Self:
        """페이퍼 트레이딩 설정.

        비용 없음, 관대한 레버리지로 전략 로직만 테스트합니다.

        Warning:
            실제 성과와 크게 다를 수 있습니다. 연구 목적으로만 사용하세요.

        Returns:
            페이퍼 트레이딩용 PortfolioManagerConfig
        """
        return cls(
            max_leverage_cap=10.0,
            rebalance_threshold=0.01,  # 1%
            system_stop_loss=None,
            cost_model=CostModel.zero(),
        )

    @classmethod
    def binance_vip(cls, vip_level: int = 1) -> Self:
        """바이낸스 VIP 등급별 설정.

        Args:
            vip_level: VIP 등급 (0-9)

        Returns:
            해당 VIP 등급에 맞는 PortfolioManagerConfig

        Example:
            >>> config = PortfolioManagerConfig.binance_vip(1)
        """
        if vip_level == 0:
            cost = CostModel.binance_futures()
        elif vip_level == 1:
            cost = CostModel.binance_vip1()
        else:
            # VIP 2+ 는 VIP 1과 동일하게 처리 (추후 확장 가능)
            cost = CostModel.binance_vip1()

        return cls(
            max_leverage_cap=3.0,
            rebalance_threshold=0.05,
            system_stop_loss=0.10,
            cost_model=cost,
        )

    @classmethod
    def signals_mode(cls) -> Self:
        """이벤트 기반 시그널 모드 설정.

        단순한 entry/exit 전략에 적합합니다.
        from_signals를 사용하여 진입/청산 시그널에만 반응합니다.

        Note:
            VW-TSMOM과 같이 연속 리밸런싱이 필요한 전략에는
            기본값인 execution_mode="orders"를 사용하세요.

        Returns:
            이벤트 기반 모드 PortfolioManagerConfig
        """
        return cls(
            execution_mode="signals",
            max_leverage_cap=3.0,
            rebalance_threshold=0.05,
            system_stop_loss=0.10,
        )

    # ==========================================================================
    # Utility Methods (유틸리티 메서드)
    # ==========================================================================
    def to_vbt_params(self) -> dict[str, object]:
        """VectorBT Portfolio.from_signals() 파라미터로 변환.

        Returns:
            VectorBT 호환 파라미터 딕셔너리

        Example:
            >>> config = PortfolioManagerConfig()
            >>> pf = vbt.Portfolio.from_signals(..., **config.to_vbt_params())
        """
        vbt_params: dict[str, object] = {
            "accumulate": self.accumulate,
            "upon_opposite_entry": self.upon_opposite_entry,
            **self.cost_model.to_vbt_params(),
        }

        # stop_loss는 sl_stop 파라미터로 전달 (None이면 제외)
        if self.system_stop_loss is not None:
            vbt_params["sl_stop"] = self.system_stop_loss

        return vbt_params

    def clamp_leverage(self, requested_leverage: float) -> float:
        """요청된 레버리지를 최대 상한으로 클램핑.

        Args:
            requested_leverage: 전략이 요청한 레버리지

        Returns:
            max_leverage_cap으로 제한된 레버리지
        """
        return min(abs(requested_leverage), self.max_leverage_cap)

    def should_rebalance(
        self,
        current_weight: float,
        target_weight: float,
    ) -> bool:
        """리밸런싱 필요 여부 판단.

        Args:
            current_weight: 현재 비중 (0.0~1.0)
            target_weight: 목표 비중 (0.0~1.0)

        Returns:
            리밸런싱 필요 시 True
        """
        diff = abs(current_weight - target_weight)
        return diff >= self.rebalance_threshold

    def summary(self) -> dict[str, object]:
        """설정 요약 정보 반환.

        Returns:
            핵심 설정만 포함된 딕셔너리
        """
        return {
            "execution_mode": self.execution_mode,
            "price_type": self.price_type,
            "size_type": self.size_type,
            "max_leverage_cap": self.max_leverage_cap,
            "system_stop_loss": (
                f"{self.system_stop_loss:.1%}" if self.system_stop_loss else "Disabled"
            ),
            "rebalance_threshold": f"{self.rebalance_threshold:.1%}",
            "cost_model": type(self.cost_model).__name__,
            "round_trip_cost": f"{self.cost_model.round_trip_cost:.2%}",
        }
