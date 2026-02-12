"""TTM Squeeze Strategy Configuration.

Bollinger Bands가 Keltner Channels 안으로 수축(squeeze) 후
해제될 때 momentum 방향으로 진입하는 전략 설정입니다.

References:
    Carter, J. (2012). "Mastering the Trade."
    TTM Squeeze = BB inside KC (low volatility) -> expansion (breakout).
"""

from __future__ import annotations

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


class TtmSqueezeConfig(BaseModel):
    """TTM Squeeze 전략 설정.

    Bollinger Bands와 Keltner Channels를 이용한 squeeze 감지 +
    momentum 기반 방향 결정 전략의 파라미터를 정의합니다.

    Signal Logic:
        1. Squeeze ON: BB upper < KC upper AND BB lower > KC lower
        2. Squeeze OFF: BB가 KC 밖으로 확장
        3. Entry: Squeeze ON -> OFF 전환 시 momentum 방향 진입
        4. Momentum: close - midline (donchian midline)
        5. Exit: close가 SMA를 역방향 크로스

    Attributes:
        bb_period: Bollinger Band 계산 기간
        bb_std: BB 표준편차 배수
        kc_period: Keltner Channel 계산 기간
        kc_mult: KC ATR 배수
        mom_period: 모멘텀 계산 lookback (donchian midline)
        exit_sma_period: 청산 SMA 기간
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = TtmSqueezeConfig(
        ...     bb_period=20,
        ...     kc_mult=1.5,
        ...     vol_target=0.40,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Bollinger Bands Parameters
    # ======================================================================
    bb_period: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Bollinger Band 계산 기간",
    )
    bb_std: float = Field(
        default=2.0,
        ge=1.0,
        le=3.0,
        description="BB 표준편차 배수",
    )

    # ======================================================================
    # Keltner Channel Parameters
    # ======================================================================
    kc_period: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Keltner Channel 계산 기간",
    )
    kc_mult: float = Field(
        default=1.5,
        ge=0.5,
        le=3.0,
        description="KC ATR 배수",
    )

    # ======================================================================
    # Momentum Parameters
    # ======================================================================
    mom_period: int = Field(
        default=20,
        ge=5,
        le=50,
        description="모멘텀 lookback (donchian midline 계산)",
    )

    # ======================================================================
    # Exit Parameters
    # ======================================================================
    exit_sma_period: int = Field(
        default=21,
        ge=5,
        le=50,
        description="청산 SMA 기간",
    )

    # ======================================================================
    # Volatility / Position Sizing
    # ======================================================================
    vol_window: int = Field(
        default=20,
        ge=5,
        le=100,
        description="변동성 계산 윈도우 (캔들 수)",
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
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: vol_target < min_volatility일 경우
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        모든 rolling 윈도우 중 가장 긴 것 + 1 (shift 여유).

        Returns:
            필요한 캔들 수
        """
        return (
            max(
                self.bb_period,
                self.kc_period,
                self.mom_period,
                self.exit_sma_period,
                self.vol_window,
            )
            + 1
        )

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> TtmSqueezeConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 TtmSqueezeConfig
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
    def conservative(cls) -> TtmSqueezeConfig:
        """보수적 설정 (넓은 BB, 넓은 KC, 낮은 vol target).

        Returns:
            보수적 파라미터의 TtmSqueezeConfig
        """
        return cls(
            bb_std=2.0,
            kc_mult=2.0,
            vol_target=0.30,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> TtmSqueezeConfig:
        """공격적 설정 (좁은 BB, 좁은 KC, 높은 vol target).

        Returns:
            공격적 파라미터의 TtmSqueezeConfig
        """
        return cls(
            bb_std=1.5,
            kc_mult=1.0,
            vol_target=0.50,
            min_volatility=0.05,
        )
