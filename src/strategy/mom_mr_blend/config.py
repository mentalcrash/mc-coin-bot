"""Momentum + Mean Reversion Blend Strategy Configuration.

Momentum Z-Score(28d)와 Mean Reversion Z-Score(14d)를 블렌딩하는
전략의 설정을 정의하는 Pydantic 모델입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 가중치 합계 허용 상한 (부동소수점 오차 허용)
_MAX_TOTAL_WEIGHT = 1.01


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


class MomMrBlendConfig(BaseModel):
    """Momentum + Mean Reversion Blend 전략 설정.

    Momentum Z-Score와 Mean Reversion Z-Score를 가중 블렌딩하여
    추세장에서는 모멘텀, 횡보장에서는 평균회귀 알파를 포착합니다.

    Signal Formula:
        1. mom_returns = close / close.shift(mom_lookback) - 1
        2. mom_zscore = (mom - rolling_mean) / rolling_std  [window=mom_z_window]
        3. mr_deviation = (close - SMA(mr_lookback)) / SMA
        4. mr_zscore = (dev - rolling_mean) / rolling_std  [window=mr_z_window]
        5. mom_signal = sign(mom_zscore.shift(1))
        6. mr_signal = -sign(mr_zscore.shift(1))  (contrarian)
        7. combined = mom_weight * mom_signal + mr_weight * mr_signal
        8. strength = combined * vol_scalar

    Attributes:
        mom_lookback: 모멘텀 수익률 계산 기간
        mom_z_window: 모멘텀 z-score rolling 윈도우
        mr_lookback: 평균회귀 SMA 기간
        mr_z_window: 평균회귀 z-score rolling 윈도우
        mom_weight: 모멘텀 가중치
        mr_weight: 평균회귀 가중치
        vol_window: 실현 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = MomMrBlendConfig(
        ...     mom_lookback=28,
        ...     mr_lookback=14,
        ...     vol_target=0.40,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Momentum Parameters
    # ======================================================================
    mom_lookback: int = Field(
        default=28,
        ge=5,
        le=120,
        description="모멘텀 수익률 계산 기간 (캔들 수)",
    )
    mom_z_window: int = Field(
        default=90,
        ge=20,
        le=365,
        description="모멘텀 z-score rolling 윈도우",
    )

    # ======================================================================
    # Mean Reversion Parameters
    # ======================================================================
    mr_lookback: int = Field(
        default=14,
        ge=3,
        le=60,
        description="평균회귀 SMA 기간",
    )
    mr_z_window: int = Field(
        default=90,
        ge=20,
        le=365,
        description="평균회귀 z-score rolling 윈도우",
    )

    # ======================================================================
    # Blend Weights
    # ======================================================================
    mom_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="모멘텀 시그널 가중치",
    )
    mr_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="평균회귀 시그널 가중치",
    )

    # ======================================================================
    # Volatility / Position Sizing
    # ======================================================================
    vol_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="실현 변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.40 = 40%)",
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
        description="연환산 계수 (일봉: 365, 4시간봉: 2190)",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (DISABLED=0, HEDGE_ONLY=1, FULL=2)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # vol_target >= min_volatility
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        # mom_weight + mr_weight > 0
        total_weight = self.mom_weight + self.mr_weight
        if total_weight <= 0:
            msg = f"mom_weight ({self.mom_weight}) + mr_weight ({self.mr_weight}) must be > 0"
            raise ValueError(msg)

        # mom_weight + mr_weight <= 1.0 (approximately)
        if total_weight > _MAX_TOTAL_WEIGHT:
            msg = (
                f"mom_weight ({self.mom_weight}) + mr_weight ({self.mr_weight}) = "
                f"{total_weight:.2f} should be <= 1.0"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            필요한 캔들 수
        """
        return (
            max(
                self.mom_lookback + self.mom_z_window,
                self.mr_lookback + self.mr_z_window,
                self.vol_window,
            )
            + 1
        )

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> MomMrBlendConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MomMrBlendConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "2h": 4380.0,
            "3h": 2920.0,
            "4h": 2190.0,
            "6h": 1460.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 365.0)

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> MomMrBlendConfig:
        """보수적 설정 (긴 lookback, 낮은 vol_target).

        Returns:
            보수적 파라미터의 MomMrBlendConfig
        """
        return cls(
            mom_lookback=42,
            mr_lookback=21,
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> MomMrBlendConfig:
        """공격적 설정 (짧은 lookback, 높은 vol_target).

        Returns:
            공격적 파라미터의 MomMrBlendConfig
        """
        return cls(
            mom_lookback=14,
            mr_lookback=7,
            vol_target=0.50,
        )
