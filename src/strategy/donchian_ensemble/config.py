"""Donchian Ensemble Strategy Configuration.

9개 lookback 기간의 Donchian Channel 신호를 평균하는 앙상블 전략 설정.
각 lookback마다 +1/0/-1 breakout 신호를 계산하고, 전체 평균을 사용합니다.

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


class DonchianEnsembleConfig(BaseModel):
    """Donchian Ensemble 전략 설정.

    9개 Donchian Channel의 breakout 시그널을 앙상블하여
    추세 방향을 결정하고, 변동성 스케일링으로 포지션을 조절합니다.

    Signal Formula:
        1. 각 lookback에 대해 Donchian Channel (upper=high.rolling.max, lower=low.rolling.min)
        2. signal_i = +1 if close > prev_upper, -1 if close < prev_lower, 0 otherwise
        3. ensemble = mean(all signal_i)
        4. direction = sign(ensemble)
        5. strength = ensemble * vol_scalar

    Attributes:
        lookbacks: Donchian Channel lookback 기간들 (캔들 수)
        atr_period: ATR 계산 기간
        vol_target: 연간 목표 변동성 (0.0~1.0)
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 모드 (DISABLED/FULL)

    Example:
        >>> config = DonchianEnsembleConfig(
        ...     lookbacks=(20, 60, 150),
        ...     vol_target=0.30,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # Donchian Channel lookback 기간들
    lookbacks: tuple[int, ...] = Field(
        default=(5, 10, 20, 30, 60, 90, 150, 250, 360),
        description="Donchian Channel lookback 기간들 (캔들 수)",
    )

    # ATR for volatility
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간",
    )

    # Volatility Scaling
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
        description="최소 변동성 클램프",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365, 시간봉: 8760)",
    )

    # Short Mode
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (DISABLED/FULL)",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: vol_target < min_volatility이거나 lookback < 2일 경우
        """
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        min_lookback = 2
        for lb in self.lookbacks:
            if lb < min_lookback:
                msg = f"All lookbacks must be >= {min_lookback}, got {lb}"
                raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        가장 긴 lookback + 1 (shift(1) 포함).

        Returns:
            필요한 캔들 수
        """
        return max(self.lookbacks) + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> DonchianEnsembleConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 DonchianEnsembleConfig
        """
        annualization_map: dict[str, float] = {
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 365.0)

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> DonchianEnsembleConfig:
        """보수적 설정 (긴 lookback 위주, 낮은 vol_target).

        Returns:
            보수적 파라미터의 DonchianEnsembleConfig
        """
        return cls(
            lookbacks=(20, 30, 60, 90, 150, 250, 360),
            vol_target=0.30,
        )

    @classmethod
    def aggressive(cls) -> DonchianEnsembleConfig:
        """공격적 설정 (짧은 lookback 위주, 높은 vol_target).

        Returns:
            공격적 파라미터의 DonchianEnsembleConfig
        """
        return cls(
            lookbacks=(5, 10, 20, 30, 60),
            vol_target=0.50,
        )
