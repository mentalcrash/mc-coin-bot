"""Enhanced VW-TSMOM Strategy Configuration.

이 모듈은 Enhanced VW-TSMOM 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
기존 TSMOM의 log1p(volume) 가중 대신 볼륨 비율 정규화(volume_ratio)를 적용합니다.

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


class EnhancedTSMOMConfig(BaseModel):
    """Enhanced VW-TSMOM 전략 설정 (Volume Ratio Normalization).

    기존 TSMOM의 log1p(volume) 가중 방식 대신 상대적 볼륨 비율을 사용하여
    모멘텀을 계산합니다. 평균 거래량 대비 현재 거래량 비율을 가중치로 적용합니다.

    Signal Formula:
        1. vol_ratio = volume / volume.rolling(volume_lookback).mean()
        2. vol_ratio = vol_ratio.clip(upper=volume_clip_max)
        3. weighted_return = log_return * vol_ratio
        4. evw_momentum = weighted_return.rolling(lookback).sum()
        5. direction = sign(evw_momentum)
        6. strength = direction * vol_scalar

    Attributes:
        lookback: 모멘텀 계산 기간 (캔들 수)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 연간 목표 변동성 (0.0~1.0)
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        volume_lookback: 거래량 이동평균 윈도우
        volume_clip_max: 거래량 비율 클리핑 상한
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
        atr_period: ATR 계산 기간

    Example:
        >>> config = EnhancedTSMOMConfig(
        ...     lookback=30,
        ...     vol_target=0.35,
        ...     volume_lookback=20,
        ...     volume_clip_max=5.0,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Momentum
    # ======================================================================
    lookback: int = Field(
        default=30,
        ge=6,
        le=365,
        description="모멘텀 계산 기간 (캔들 수)",
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
        description="연간 목표 변동성 (0.0~1.0)",
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
    # Volume Ratio Normalization (NEW)
    # ======================================================================
    volume_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="거래량 이동평균 윈도우 (상대적 볼륨 비율 계산용)",
    )
    volume_clip_max: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="거래량 비율 클리핑 상한 (이상치 영향 제한)",
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

    # ======================================================================
    # ATR Stop (Portfolio 레이어에 위임)
    # ======================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (trailing stop 참조용)",
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

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        evw_momentum은 returns(1) -> volume rolling(volume_lookback) -> momentum rolling(lookback)
        순서로 계산되므로, 실질적 워밍업은 volume_lookback + lookback 입니다.

        Returns:
            필요한 캔들 수
        """
        # evw_momentum: 1 (returns) + volume_lookback + lookback
        evw_warmup = 1 + self.volume_lookback + self.lookback
        # realized_vol: 1 (returns) + vol_window
        vol_warmup = 1 + self.vol_window
        # atr
        atr_warmup = self.atr_period

        return max(evw_warmup, vol_warmup, atr_warmup) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> "EnhancedTSMOMConfig":
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 EnhancedTSMOMConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        lookback_map: dict[str, int] = {
            "1m": 60,
            "5m": 48,
            "15m": 48,
            "1h": 24,
            "4h": 24,
            "1d": 30,
        }

        annualization = annualization_map.get(timeframe, 365.0)
        lookback = lookback_map.get(timeframe, 30)

        return cls(
            lookback=lookback,
            vol_window=lookback,
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> "EnhancedTSMOMConfig":
        """보수적 설정 (긴 lookback, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 EnhancedTSMOMConfig
        """
        return cls(
            lookback=48,
            vol_window=48,
            vol_target=0.10,
            min_volatility=0.08,
            volume_lookback=30,
            volume_clip_max=3.0,
        )

    @classmethod
    def aggressive(cls) -> "EnhancedTSMOMConfig":
        """공격적 설정 (짧은 lookback, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 EnhancedTSMOMConfig
        """
        return cls(
            lookback=12,
            vol_window=12,
            vol_target=0.20,
            min_volatility=0.05,
            volume_lookback=10,
            volume_clip_max=8.0,
        )
