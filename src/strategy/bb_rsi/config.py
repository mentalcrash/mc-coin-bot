"""BB+RSI Mean Reversion Strategy Configuration.

볼린저밴드 + RSI 기반 평균회귀 전략의 설정을 정의하는 Pydantic 모델입니다.
횡보장(ADX < 25)에서 과매수/과매도 구간의 평균회귀를 포착합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from enum import IntEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 부동소수점 비교 허용 오차
WEIGHT_SUM_TOLERANCE = 1e-6


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


class BBRSIConfig(BaseModel):
    """BB+RSI 평균회귀 전략 설정.

    볼린저밴드와 RSI를 조합하여 과매수/과매도 구간에서
    평균회귀 시그널을 생성합니다. ADX 필터로 횡보장에서만 활성화됩니다.

    Signal Formula:
        1. bb_signal = -(close - bb_middle) / (bb_upper - bb_lower) * 2
        2. rsi_signal = (50 - RSI) / 50
        3. combined = bb_weight * bb_signal + rsi_weight * rsi_signal
        4. strength = combined.shift(1) * vol_scalar.shift(1)

    Attributes:
        bb_period: 볼린저밴드 SMA 기간
        bb_std: 볼린저밴드 표준편차 배수
        rsi_period: RSI 계산 기간
        rsi_oversold: 과매도 임계값 (진입 조건)
        rsi_overbought: 과매수 임계값 (진입 조건)
        bb_weight: BB 시그널 가중치
        rsi_weight: RSI 시그널 가중치

    Example:
        >>> config = BBRSIConfig(
        ...     bb_period=20,
        ...     rsi_period=14,
        ...     vol_target=0.20,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Bollinger Bands
    # ======================================================================
    bb_period: int = Field(
        default=20,
        ge=5,
        le=100,
        description="볼린저밴드 SMA 기간",
    )
    bb_std: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="볼린저밴드 표준편차 배수",
    )

    # ======================================================================
    # RSI
    # ======================================================================
    rsi_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="RSI 계산 기간 (Wilder's RSI)",
    )
    rsi_oversold: float = Field(
        default=30.0,
        ge=10.0,
        le=45.0,
        description="과매도 임계값 (이 이하면 롱 진입 조건)",
    )
    rsi_overbought: float = Field(
        default=70.0,
        ge=55.0,
        le=90.0,
        description="과매수 임계값 (이 이상이면 숏 진입 조건)",
    )

    # ======================================================================
    # Signal Combination
    # ======================================================================
    bb_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="BB 시그널 가중치 (bb_weight + rsi_weight = 1.0)",
    )
    rsi_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="RSI 시그널 가중치 (bb_weight + rsi_weight = 1.0)",
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
        default=0.20,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.20 = 20%)",
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
    # Regime Filter (ADX)
    # ======================================================================
    use_adx_filter: bool = Field(
        default=True,
        description="ADX 레짐 필터 활성화 (추세장에서 포지션 축소)",
    )
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ADX 계산 기간",
    )
    adx_threshold: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="ADX 임계값 (이 이상이면 추세장 → 포지션 축소)",
    )
    trending_position_scale: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="추세장(ADX >= threshold)에서 포지션 스케일",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
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

    # ======================================================================
    # ATR Stop (Portfolio 레이어에 위임)
    # ======================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (trailing stop 참조용)",
    )
    atr_stop_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="ATR trailing stop 배수",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # bb_weight + rsi_weight = 1.0 검증
        weight_sum = self.bb_weight + self.rsi_weight
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            msg = (
                f"bb_weight ({self.bb_weight}) + rsi_weight ({self.rsi_weight}) "
                f"= {weight_sum}, must equal 1.0"
            )
            raise ValueError(msg)

        # vol_target >= min_volatility
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        # rsi_oversold < rsi_overbought
        if self.rsi_oversold >= self.rsi_overbought:
            msg = (
                f"rsi_oversold ({self.rsi_oversold}) must be < "
                f"rsi_overbought ({self.rsi_overbought})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            필요한 캔들 수
        """
        periods = [self.bb_period, self.rsi_period, self.vol_window]
        if self.use_adx_filter:
            periods.append(self.adx_period)
        periods.append(self.atr_period)
        return max(periods) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> "BBRSIConfig":
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 BBRSIConfig
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
    def conservative(cls) -> "BBRSIConfig":
        """보수적 설정 (넓은 밴드, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 BBRSIConfig
        """
        return cls(
            bb_period=30,
            bb_std=2.5,
            vol_target=0.15,
            min_volatility=0.08,
            atr_stop_multiplier=2.0,
        )

    @classmethod
    def aggressive(cls) -> "BBRSIConfig":
        """공격적 설정 (좁은 밴드, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 BBRSIConfig
        """
        return cls(
            bb_period=14,
            bb_std=1.5,
            vol_target=0.30,
            min_volatility=0.05,
            atr_stop_multiplier=1.0,
        )
