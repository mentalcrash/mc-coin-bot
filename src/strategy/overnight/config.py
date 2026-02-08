"""Overnight Seasonality Strategy Configuration.

시간대 기반 진입/청산으로 crypto overnight effect를 포착하는 전략 설정입니다.
1H 타임프레임에서 특정 시간대에만 포지션을 유지합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

import logging
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode

logger = logging.getLogger(__name__)


class OvernightConfig(BaseModel):
    """Overnight Seasonality 전략 설정.

    특정 UTC 시간대에 진입하고 청산하는 시간 기반 전략입니다.
    Crypto 시장의 야간 계절성 효과를 포착합니다.

    Signal Formula:
        1. in_position = (hour >= entry_hour) | (hour < exit_hour)  [wrap-around]
        2. direction = in_position_shifted (shift(1) 적용)
        3. strength = direction * vol_scalar

    Attributes:
        entry_hour: UTC 진입 시간 (예: 22 = 22:00 UTC)
        exit_hour: UTC 청산 시간 (예: 0 = 00:00 UTC)
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수 (1H: 8760)
        use_vol_filter: 변동성 필터 활성화 여부
        vol_filter_threshold: 변동성 비율 임계값
        vol_filter_lookback: 변동성 필터 롤링 윈도우
        short_mode: 숏 모드 (기본: Long-Only)

    Example:
        >>> config = OvernightConfig(
        ...     entry_hour=22,
        ...     exit_hour=0,
        ...     vol_target=0.30,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Time-Based Parameters
    # ======================================================================
    entry_hour: int = Field(
        default=22,
        ge=0,
        le=23,
        description="UTC 진입 시간 (0-23)",
    )
    exit_hour: int = Field(
        default=0,
        ge=0,
        le=23,
        description="UTC 청산 시간 (0-23)",
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
        default=8760.0,
        gt=0,
        description="연환산 계수 (1H: 8760, 4H: 2190, 1D: 365)",
    )

    # ======================================================================
    # Volatility Filter
    # ======================================================================
    use_vol_filter: bool = Field(
        default=False,
        description="변동성 필터 활성화 (고변동성 시 strength 스케일업)",
    )
    vol_filter_threshold: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="변동성 비율 임계값 (realized_vol / rolling_mean)",
    )
    vol_filter_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="변동성 필터 롤링 윈도우",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.DISABLED,
        description="숏 포지션 처리 모드 (기본: Long-Only)",
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

        # entry_hour != exit_hour
        if self.entry_hour == self.exit_hour:
            msg = f"entry_hour ({self.entry_hour}) must differ from exit_hour ({self.exit_hour})"
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            필요한 캔들 수
        """
        return self.vol_window + 1

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> OvernightConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Overnight 전략은 1H 타임프레임에서 가장 의미 있습니다.
        다른 타임프레임 사용 시 경고를 출력합니다.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 맞는 OvernightConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 8760.0)

        if timeframe != "1h":
            logger.warning(
                "Overnight strategy is designed for 1H timeframe, got '%s'. Hour-based entry/exit may not work correctly.",
                timeframe,
            )

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> OvernightConfig:
        """보수적 설정 (낮은 변동성 타겟, vol 필터 비활성).

        Returns:
            보수적 파라미터의 OvernightConfig
        """
        return cls(
            vol_target=0.15,
            min_volatility=0.08,
            vol_window=48,
            use_vol_filter=False,
        )

    @classmethod
    def aggressive(cls) -> OvernightConfig:
        """공격적 설정 (높은 변동성 타겟, vol 필터 활성).

        Returns:
            공격적 파라미터의 OvernightConfig
        """
        return cls(
            vol_target=0.40,
            min_volatility=0.05,
            vol_window=20,
            use_vol_filter=True,
            vol_filter_threshold=1.3,
        )
