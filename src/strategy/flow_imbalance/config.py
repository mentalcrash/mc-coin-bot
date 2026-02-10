"""Flow Imbalance Strategy Configuration.

BVC bar position + OFI direction + VPIN activity로
주문 흐름 불균형을 감지하는 1H 전략입니다.

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


class FlowImbalanceConfig(BaseModel):
    """Flow Imbalance 전략 설정.

    BVC(Bulk Volume Classification)로 매수/매도 볼륨을 분류하고,
    OFI(Order Flow Imbalance)와 VPIN으로 주문 흐름 방향을 결정합니다.

    Signal Formula:
        1. buy_ratio = (close - low) / (high - low)
        2. buy_vol = volume * buy_ratio, sell_vol = volume * (1 - buy_ratio)
        3. ofi = rolling_sum(buy_vol - sell_vol) / rolling_sum(volume)
        4. vpin_proxy = rolling_std(buy_ratio)
        5. Long: ofi > entry_threshold & vpin > vpin_threshold
        6. Short: ofi < -entry_threshold & vpin > vpin_threshold
        7. Exit: |ofi| < exit_threshold OR timeout

    Attributes:
        ofi_window: OFI rolling 윈도우 (1H bars)
        ofi_entry_threshold: OFI 진입 임계값
        ofi_exit_threshold: OFI 청산 임계값
        vpin_window: VPIN proxy 윈도우 (1H bars)
        vpin_threshold: VPIN 활동성 임계값
        timeout_bars: 최대 포지션 유지 기간 (1H bars)
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Flow 파라미터
    # =========================================================================
    ofi_window: int = Field(
        default=6,
        ge=2,
        le=48,
        description="OFI rolling 윈도우 (1H bars)",
    )
    ofi_entry_threshold: float = Field(
        default=0.6,
        ge=0.1,
        le=0.95,
        description="OFI 진입 임계값 (절대값)",
    )
    ofi_exit_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="OFI 청산 임계값 (절대값)",
    )

    # =========================================================================
    # VPIN 파라미터
    # =========================================================================
    vpin_window: int = Field(
        default=24,
        ge=6,
        le=168,
        description="VPIN proxy 윈도우 (1H bars)",
    )
    vpin_threshold: float = Field(
        default=0.15,
        ge=0.01,
        le=0.50,
        description="VPIN 활동성 임계값",
    )

    # =========================================================================
    # Timeout 파라미터
    # =========================================================================
    timeout_bars: int = Field(
        default=24,
        ge=1,
        le=168,
        description="최대 포지션 유지 기간 (1H bars)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
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
        default=8760.0,
        gt=0,
        description="연환산 계수 (1H: 8760 = 24*365)",
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
        default=ShortMode.FULL,
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
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.ofi_exit_threshold >= self.ofi_entry_threshold:
            msg = (
                f"ofi_exit_threshold ({self.ofi_exit_threshold}) must be < "
                f"ofi_entry_threshold ({self.ofi_entry_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return max(self.vpin_window, self.ofi_window, self.atr_period) + 1
