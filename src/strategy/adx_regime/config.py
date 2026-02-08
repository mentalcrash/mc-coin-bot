"""ADX Regime Filter Strategy Configuration.

ADX(Average Directional Index) 기반 momentum/mean-reversion 자동 전환 전략의
설정을 정의하는 Pydantic 모델입니다.

ADX가 높으면(추세장) momentum 전략, 낮으면(횡보장) mean-reversion 전략을
자동으로 블렌딩하여 시장 레짐에 적응합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class ADXRegimeConfig(BaseModel):
    """ADX Regime Filter 전략 설정.

    ADX 수준에 따라 momentum과 mean-reversion 시그널을 블렌딩합니다.
    ADX < adx_low → MR only, ADX > adx_high → Trend only, 중간 → 선형 블렌딩.

    Signal Formula:
        1. trend_weight = clip((adx - adx_low) / (adx_high - adx_low), 0, 1)
        2. mr_weight = 1 - trend_weight
        3. blended = trend_weight * mom_direction + mr_weight * mr_direction
        4. strength = blended * vol_scalar (Shift(1) 적용)

    Attributes:
        adx_period: ADX 계산 기간
        adx_low: 이 이하면 MR only (횡보장)
        adx_high: 이 이상이면 Trend only (추세장)
        mom_lookback: 모멘텀 lookback
        mr_lookback: Z-Score lookback
        mr_entry_z: MR 진입 z-score 임계값
        mr_exit_z: MR 청산 z-score 임계값

    Example:
        >>> config = ADXRegimeConfig(
        ...     adx_period=14,
        ...     adx_low=15.0,
        ...     adx_high=25.0,
        ...     vol_target=0.30,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # ADX Regime Thresholds
    # ======================================================================
    adx_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ADX 계산 기간",
    )
    adx_low: float = Field(
        default=15.0,
        ge=5.0,
        le=40.0,
        description="ADX 하한 (이 이하면 MR only)",
    )
    adx_high: float = Field(
        default=25.0,
        ge=10.0,
        le=50.0,
        description="ADX 상한 (이 이상이면 Trend only)",
    )

    # ======================================================================
    # Momentum Leg
    # ======================================================================
    mom_lookback: int = Field(
        default=30,
        ge=6,
        le=365,
        description="모멘텀 계산 기간 (캔들 수)",
    )

    # ======================================================================
    # Mean-Reversion Leg (Z-Score)
    # ======================================================================
    mr_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Z-Score 계산 lookback",
    )
    mr_entry_z: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="MR 진입 z-score 임계값",
    )
    mr_exit_z: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="MR 청산 z-score 임계값",
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
        # adx_low < adx_high
        if self.adx_low >= self.adx_high:
            msg = f"adx_low ({self.adx_low}) must be < adx_high ({self.adx_high})"
            raise ValueError(msg)

        # mr_exit_z < mr_entry_z
        if self.mr_exit_z >= self.mr_entry_z:
            msg = f"mr_exit_z ({self.mr_exit_z}) must be < mr_entry_z ({self.mr_entry_z})"
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

        ADX는 내부적으로 3*period 정도의 warmup이 필요합니다.

        Returns:
            필요한 캔들 수
        """
        return (
            max(
                self.adx_period * 3,
                self.mom_lookback,
                self.mr_lookback,
                self.vol_window,
            )
            + 1
        )

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> ADXRegimeConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 ADXRegimeConfig
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

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> ADXRegimeConfig:
        """보수적 설정 (넓은 ADX 밴드, 낮은 변동성 타겟).

        Returns:
            보수적 파라미터의 ADXRegimeConfig
        """
        return cls(
            adx_period=20,
            adx_low=20.0,
            adx_high=35.0,
            mom_lookback=48,
            mr_lookback=30,
            mr_entry_z=2.5,
            mr_exit_z=0.8,
            vol_target=0.15,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> ADXRegimeConfig:
        """공격적 설정 (좁은 ADX 밴드, 높은 변동성 타겟).

        Returns:
            공격적 파라미터의 ADXRegimeConfig
        """
        return cls(
            adx_period=10,
            adx_low=12.0,
            adx_high=22.0,
            mom_lookback=14,
            mr_lookback=14,
            mr_entry_z=1.5,
            mr_exit_z=0.3,
            vol_target=0.40,
            min_volatility=0.05,
            short_mode=ShortMode.FULL,
        )
