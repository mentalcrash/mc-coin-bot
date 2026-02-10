"""VPIN Flow Toxicity Strategy Configuration.

Volume-Synchronized Probability of Informed Trading으로 flow toxicity를 측정합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 -> 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class VPINFlowConfig(BaseModel):
    """VPIN Flow Toxicity 전략 설정.

    BVC(Bulk Volume Classification)로 buy/sell volume을 근사하고,
    VPIN(Volume-Synchronized PIN)으로 정보거래 확률을 측정합니다.

    Signal Formula:
        1. buy_pct = norm.cdf((close - open) / (high - low + eps))
        2. v_buy = volume * buy_pct, v_sell = volume * (1 - buy_pct)
        3. order_imbalance = |v_buy - v_sell|
        4. vpin = rolling_sum(order_imbalance) / rolling_sum(volume)
        5. flow_direction = sign(rolling_sum(v_buy) - rolling_sum(v_sell))
        6. High toxicity: vpin > threshold_high → follow flow direction
        7. Low/Mid: neutral
        8. strength = direction * vol_scalar

    Attributes:
        n_buckets: Rolling VPIN 윈도우
        threshold_high: 고독성 임계값
        threshold_low: 저독성 임계값
        flow_direction_period: 플로우 방향 EMA 기간
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # VPIN 파라미터
    # =========================================================================
    n_buckets: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Rolling VPIN 윈도우 (버킷 수)",
    )
    threshold_high: float = Field(
        default=0.7,
        ge=0.3,
        le=0.95,
        description="고독성 임계값 (이 이상이면 informed trading 추종)",
    )
    threshold_low: float = Field(
        default=0.3,
        ge=0.05,
        le=0.5,
        description="저독성 임계값 (이 이하면 안정, 중립)",
    )
    flow_direction_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="플로우 방향 rolling 기간",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.30,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프 (0으로 나누기 방지)",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
    )

    # =========================================================================
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (Trailing Stop용)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.HEDGE_ONLY,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="헤지 숏 활성화 드로다운 임계값 (예: -0.07 = -7%)",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="헤지 숏 강도 비율 (롱 대비, 예: 0.8 = 80%)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증."""
        if self.threshold_high <= self.threshold_low:
            msg = (
                f"threshold_high ({self.threshold_high}) must be > "
                f"threshold_low ({self.threshold_low})"
            )
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.n_buckets, self.flow_direction_period, self.atr_period) + 1
