"""Hour Seasonality Strategy Configuration.

Per-hour rolling t-stat으로 시간대별 수익률 패턴을 감지하고,
volume confirm으로 필터링하는 1H 전략입니다.

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


class HourSeasonConfig(BaseModel):
    """Hour Seasonality 전략 설정.

    각 시간대(0~23h)의 과거 N일간 수익률에 대해 rolling t-stat을 계산하고,
    통계적으로 유의미한 시간대에서만 진입합니다.

    Signal Formula:
        1. same-hour returns: 24 bars 간격으로 lagged returns 수집
        2. hour_t_stat = mean(lagged_returns) / (std(lagged_returns) / sqrt(n))
        3. rel_volume = volume / volume.rolling(vol_confirm_window).median()
        4. Long: t_stat > threshold & rel_volume > vol_confirm
        5. Short: t_stat < -threshold & rel_volume > vol_confirm

    Attributes:
        season_window_days: Same-hour 수익률 윈도우 (일 단위)
        t_stat_threshold: t-stat 진입 임계값
        vol_confirm_window: Volume confirm 윈도우 (1H bars)
        vol_confirm_threshold: Volume confirm 임계값 (relative volume)
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Seasonality 파라미터
    # =========================================================================
    season_window_days: int = Field(
        default=45,
        ge=7,
        le=90,
        description="Same-hour 수익률 윈도우 (일 단위, 45 = 45일, n=45 관측치)",
    )
    t_stat_threshold: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="t-stat 진입 임계값 (절대값)",
    )
    t_stat_exit_threshold: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="t-stat 청산 임계값 (hysteresis: entry > exit 방지 과빈번 전환)",
    )

    # =========================================================================
    # Volume 파라미터
    # =========================================================================
    vol_confirm_window: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Volume confirm 윈도우 (1H bars, 168 = 7일)",
    )
    vol_confirm_threshold: float = Field(
        default=1.0,
        ge=0.3,
        le=3.0,
        description="Volume confirm 임계값 (relative volume >= threshold)",
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
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        if self.t_stat_exit_threshold >= self.t_stat_threshold:
            msg = (
                f"t_stat_exit_threshold ({self.t_stat_exit_threshold}) must be < "
                f"t_stat_threshold ({self.t_stat_threshold})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (1H bars)."""
        return self.season_window_days * 24 + 1
