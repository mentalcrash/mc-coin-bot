"""Larry Williams Volatility Breakout Strategy Configuration.

전일 변동폭(High - Low)의 k배를 돌파하면 진입하는 단기 변동성 돌파 전략 설정입니다.
1-bar hold: 돌파 발생 시 1일 보유 후 청산합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)


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


class LarryVBConfig(BaseModel):
    """Larry Williams Volatility Breakout 전략 설정.

    전일 변동폭(High - Low)에 k_factor를 곱한 만큼을 당일 시가에 더한 레벨을
    돌파하면 진입합니다. 1-bar hold 패턴으로 다음 바에서 청산합니다.

    Signal Formula:
        1. prev_range = (High - Low).shift(1)
        2. breakout_upper = Open + k * prev_range
        3. breakout_lower = Open - k * prev_range
        4. long_breakout = Close > breakout_upper
        5. short_breakout = Close < breakout_lower
        6. direction = shift(1) 적용 (1-bar hold)

    Attributes:
        k_factor: 돌파 배수 (0.5 = 전일 변동폭의 50%)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드 (기본: FULL)

    Example:
        >>> config = LarryVBConfig(k_factor=0.5, vol_target=0.40)
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Breakout 파라미터
    # =========================================================================
    k_factor: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="돌파 배수 (전일 변동폭 * k_factor)",
    )

    # =========================================================================
    # 변동성 스케일링 파라미터
    # =========================================================================
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

    # =========================================================================
    # 시간 프레임 관련
    # =========================================================================
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # =========================================================================
    # Short Mode
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (기본: FULL Long/Short)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
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

        prev_range를 위한 shift(1) + vol_window + 시그널 shift(1)을 포함합니다.

        Returns:
            필요한 캔들 수
        """
        return self.vol_window + 2

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> LarryVBConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 맞는 LarryVBConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 365.0)

        if timeframe != "1d":
            logger.warning(
                "Larry VB strategy is designed for 1D timeframe, got '%s'.",
                timeframe,
            )

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> LarryVBConfig:
        """보수적 설정 (높은 k_factor, 낮은 vol_target).

        높은 k_factor로 더 큰 돌파만 포착, 낮은 vol_target으로 포지션 축소.

        Returns:
            보수적 파라미터의 LarryVBConfig
        """
        return cls(
            k_factor=0.7,
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> LarryVBConfig:
        """공격적 설정 (낮은 k_factor, 높은 vol_target).

        낮은 k_factor로 작은 돌파도 포착, 높은 vol_target으로 포지션 확대.

        Returns:
            공격적 파라미터의 LarryVBConfig
        """
        return cls(
            k_factor=0.3,
            vol_target=0.50,
        )
