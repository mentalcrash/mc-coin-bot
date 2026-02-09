"""Multi-Factor Ensemble Strategy Configuration.

이 모듈은 Multi-Factor Ensemble 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
3개의 직교 팩터(모멘텀, 거래량 충격, 역변동성)를 균등 가중 결합합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class MultiFactorConfig(BaseModel):
    """Multi-Factor Ensemble 전략 설정.

    3개의 직교 팩터(모멘텀, 거래량 충격, 역변동성)를 z-score 정규화 후
    균등 가중 결합하여 복합 시그널을 생성합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. momentum_factor = z-score(rolling_return(lookback))
        2. volume_shock_factor = z-score(short_vol / long_vol)
        3. volatility_factor = -z-score(rolling_vol) (inverse: low vol premium)
        4. combined_score = (momentum + volume_shock + volatility) / 3
        5. direction = sign(combined_score)
        6. strength = direction * vol_scalar

    Attributes:
        momentum_lookback: 모멘텀 팩터 lookback 기간
        volume_shock_window: 거래량 충격 감지 단기 윈도우
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        zscore_window: z-score 정규화 윈도우
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = MultiFactorConfig(
        ...     momentum_lookback=21,
        ...     vol_target=0.35,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # 모멘텀 팩터 파라미터
    # =========================================================================
    momentum_lookback: int = Field(
        default=21,
        ge=5,
        le=120,
        description="모멘텀 팩터 lookback 기간 (캔들 수)",
    )

    # =========================================================================
    # 거래량 충격 팩터 파라미터
    # =========================================================================
    volume_shock_window: int = Field(
        default=5,
        ge=2,
        le=30,
        description="거래량 충격 감지 단기 윈도우 (캔들 수)",
    )

    # =========================================================================
    # 변동성 파라미터
    # =========================================================================
    vol_window: int = Field(
        default=30,
        ge=5,
        le=120,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (예: 0.35 = 35%)",
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
        description="연환산 계수 (일봉: 365, 4시간봉: 2190, 시간봉: 8760)",
    )

    # =========================================================================
    # Z-Score 정규화 파라미터
    # =========================================================================
    zscore_window: int = Field(
        default=60,
        ge=20,
        le=252,
        description="Z-score 정규화 롤링 윈도우 (캔들 수)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (DISABLED/HEDGE_ONLY/FULL)",
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
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        전략 계산을 시작하기 전 필요한 최소 데이터 양입니다.
        Rolling 계산의 초기 NaN을 피하기 위해 사용됩니다.

        Returns:
            필요한 캔들 수
        """
        return max(self.momentum_lookback, self.vol_window, self.zscore_window) + 1
