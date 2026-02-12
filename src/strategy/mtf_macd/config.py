"""MTF MACD Strategy Configuration.

MACD(12,26,9) 기반 추세 필터 + crossover 진입 전략의 설정을 정의합니다.
VBT 간소화 버전: Daily MACD를 trend/entry 양쪽에 사용합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ShortMode(IntEnum):
    """숏 포지션 처리 모드.

    Attributes:
        DISABLED: Long-Only 모드 (숏 시그널 -> 중립)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    FULL = 2


class MtfMacdConfig(BaseModel):
    """MTF MACD 전략 설정.

    MACD(12,26,9) crossover + trend filter 전략의 핵심 파라미터입니다.
    Daily MACD를 trend 판단과 entry 신호 양쪽에 사용합니다.

    Signal Logic:
        - Long Entry: MACD > Signal Line crossover AND MACD > 0
        - Short Entry: MACD < Signal Line crossover AND MACD < 0
        - Long Exit: bearish candle (close < open)
        - Short Exit: bullish candle (close > open)

    Attributes:
        fast_period: Fast EMA 기간
        slow_period: Slow EMA 기간
        signal_period: Signal Line EMA 기간
        vol_window: 실현 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = MtfMacdConfig(fast_period=12, slow_period=26)
        >>> config.warmup_periods()
        36
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # MACD Parameters
    # ======================================================================
    fast_period: int = Field(
        default=12,
        ge=5,
        le=50,
        description="Fast EMA 기간",
    )
    slow_period: int = Field(
        default=26,
        ge=10,
        le=100,
        description="Slow EMA 기간",
    )
    signal_period: int = Field(
        default=9,
        ge=3,
        le=30,
        description="Signal Line EMA 기간",
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
        description="연환산 계수 (일봉: 365)",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        검증 규칙:
            - fast_period < slow_period
            - vol_target >= min_volatility

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        if self.fast_period >= self.slow_period:
            msg = f"fast_period ({self.fast_period}) must be < slow_period ({self.slow_period})"
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """워밍업 기간 (캔들 수).

        slow_period + signal_period + 1 (shift 여유분).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self.slow_period + self.signal_period + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> MtfMacdConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MtfMacdConfig
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
    def conservative(cls) -> MtfMacdConfig:
        """보수적 설정 (표준 MACD, 낮은 vol target).

        Returns:
            보수적 파라미터의 MtfMacdConfig
        """
        return cls(
            fast_period=12,
            slow_period=26,
            signal_period=9,
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> MtfMacdConfig:
        """공격적 설정 (빠른 MACD, 높은 vol target).

        Returns:
            공격적 파라미터의 MtfMacdConfig
        """
        return cls(
            fast_period=8,
            slow_period=17,
            signal_period=9,
            vol_target=0.50,
        )
