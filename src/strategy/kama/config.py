"""KAMA Trend Following Strategy Configuration.

Kaufman Adaptive Moving Average 기반 추세 추종 전략의 설정을 정의하는 Pydantic 모델입니다.
KAMA는 Efficiency Ratio를 사용하여 추세장과 횡보장을 자동으로 구분합니다.

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
        DISABLED: Long-Only 모드 (숏 시그널 → 중립)
        HEDGE_ONLY: 헤지 목적 숏만 (드로다운 임계값 초과 시)
        FULL: 완전한 Long/Short 모드
    """

    DISABLED = 0
    HEDGE_ONLY = 1
    FULL = 2


class KAMAConfig(BaseModel):
    """KAMA 추세 추종 전략 설정.

    Kaufman Adaptive Moving Average를 사용하여 추세 방향을 판단하고,
    ATR 필터로 잡음을 제거합니다. Efficiency Ratio가 높을수록
    빠른 이동평균에 가까워져 추세 추종이 적극적으로 작동합니다.

    Signal Formula:
        1. ER = |close - close.shift(er_lookback)| / sum(|close.diff()|, er_lookback)
        2. SC = (ER * (fast_sc - slow_sc) + slow_sc) ** 2
        3. KAMA[i] = KAMA[i-1] + SC[i] * (close[i] - KAMA[i-1])
        4. long = (close > KAMA + atr_mult * ATR) & (KAMA rising)
        5. short = (close < KAMA - atr_mult * ATR) & (KAMA falling)

    Attributes:
        er_lookback: Efficiency Ratio 룩백 기간
        fast_period: 빠른 Smoothing Constant 기간
        slow_period: 느린 Smoothing Constant 기간
        atr_period: ATR 계산 기간
        atr_multiplier: ATR 기반 진입 필터 배수

    Example:
        >>> config = KAMAConfig(
        ...     er_lookback=10,
        ...     fast_period=2,
        ...     slow_period=30,
        ...     vol_target=0.30,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # KAMA Parameters
    # ======================================================================
    er_lookback: int = Field(
        default=10,
        ge=5,
        le=100,
        description="Efficiency Ratio 룩백 기간",
    )
    fast_period: int = Field(
        default=2,
        ge=2,
        le=10,
        description="빠른 Smoothing Constant 기간 (추세장)",
    )
    slow_period: int = Field(
        default=30,
        ge=10,
        le=100,
        description="느린 Smoothing Constant 기간 (횡보장)",
    )

    # ======================================================================
    # ATR Filter
    # ======================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간",
    )
    atr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="ATR 기반 진입 필터 배수 (close가 KAMA +/- atr_mult*ATR 초과 시 진입)",
    )

    # ======================================================================
    # Volatility / Position Sizing
    # ======================================================================
    vol_window: int = Field(
        default=30,
        ge=6,
        le=365,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.30,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.30 = 30%)",
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
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
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
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # slow_period > fast_period
        if self.slow_period <= self.fast_period:
            msg = f"slow_period ({self.slow_period}) must be > fast_period ({self.fast_period})"
            raise ValueError(msg)

        # vol_target >= min_volatility
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            필요한 캔들 수
        """
        periods = [
            self.er_lookback,
            self.slow_period,
            self.vol_window,
            self.atr_period,
        ]
        return max(periods) + 1
