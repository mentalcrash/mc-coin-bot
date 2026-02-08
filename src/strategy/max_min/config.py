"""MAX/MIN Combined Strategy Configuration.

이 모듈은 MAX/MIN 복합 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
신고가(MAX) trend-following과 신저가(MIN) mean-reversion을 결합합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode

_WEIGHT_TOLERANCE = 1e-6


class MaxMinConfig(BaseModel):
    """MAX/MIN 복합 전략 설정.

    신고가 돌파(trend-following)와 신저가 돌파(mean-reversion)를
    가중 결합하여 시그널을 생성합니다.

    Signal Formula:
        1. max_signal = (close > rolling_max).astype(float)  # 신고가 → 매수 (trend)
        2. min_signal = (close < rolling_min).astype(float)  # 신저가 → 매수 (MR)
        3. combined = max_weight * max_signal + min_weight * min_signal
        4. strength = combined * vol_scalar

    Attributes:
        lookback: Rolling high/low 계산 기간 (캔들 수)
        max_weight: MAX(trend) 시그널 가중치
        min_weight: MIN(mean reversion) 시그널 가중치
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = MaxMinConfig(lookback=10, max_weight=0.7, min_weight=0.3)
    """

    model_config = ConfigDict(frozen=True)

    # Rolling window
    lookback: int = Field(
        default=10,
        ge=5,
        le=60,
        description="Rolling high/low 계산 기간 (캔들 수)",
    )

    # 시그널 가중치
    max_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MAX(trend-following) 시그널 가중치",
    )
    min_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="MIN(mean-reversion) 시그널 가중치",
    )

    # 변동성 파라미터
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

    # 시간 프레임 관련
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # 숏 모드 설정
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
            ValueError: 가중치 합이 1.0이 아니거나, vol_target < min_volatility인 경우
        """
        # max_weight + min_weight == 1.0 (허용 오차 1e-6)
        weight_sum = self.max_weight + self.min_weight
        if abs(weight_sum - 1.0) > _WEIGHT_TOLERANCE:
            msg = (
                f"max_weight ({self.max_weight}) + min_weight ({self.min_weight}) "
                f"must sum to 1.0, got {weight_sum}"
            )
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

        lookback (rolling max/min) + vol_window (변동성) + 1 (shift).

        Returns:
            필요한 캔들 수
        """
        return self.lookback + self.vol_window + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> MaxMinConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "15m", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 MaxMinConfig
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
            "15m": 24,
            "1h": 24,
            "4h": 12,
            "1d": 10,
        }

        annualization = annualization_map.get(timeframe, 8760.0)
        lookback = lookback_map.get(timeframe, 10)

        return cls(
            lookback=lookback,
            vol_window=max(lookback, 6),
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> MaxMinConfig:
        """보수적 설정 (긴 lookback, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 MaxMinConfig
        """
        return cls(
            lookback=20,
            max_weight=0.6,
            min_weight=0.4,
            vol_window=48,
            vol_target=0.15,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> MaxMinConfig:
        """공격적 설정 (짧은 lookback, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 MaxMinConfig
        """
        return cls(
            lookback=5,
            max_weight=0.4,
            min_weight=0.6,
            vol_window=14,
            vol_target=0.40,
            min_volatility=0.05,
        )
