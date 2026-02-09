"""Copula Pairs Trading Strategy Configuration.

Engle-Granger cointegration -> spread -> z-score -> mean-reversion signals.
Full copula fitting is deferred to a later phase.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class CopulaPairsConfig(BaseModel):
    """Copula Pairs Trading 전략 설정.

    Engle-Granger cointegration 기반 페어 트레이딩 전략의 파라미터입니다.
    Spread z-score가 entry/exit/stop 임계값을 넘으면 시그널을 생성합니다.

    Attributes:
        formation_window: Cointegration formation 윈도우 (rolling OLS)
        zscore_entry: Z-score 진입 임계값 (양수)
        zscore_exit: Z-score 청산 임계값 (양수)
        zscore_stop: Z-score 스탑 임계값 (양수)
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = CopulaPairsConfig(formation_window=63, zscore_entry=2.0)
    """

    model_config = ConfigDict(frozen=True)

    # Cointegration 파라미터
    formation_window: int = Field(
        default=63,
        ge=20,
        le=252,
        description="Cointegration formation window",
    )

    # Z-score 임계값
    zscore_entry: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Z-score entry threshold",
    )
    zscore_exit: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Z-score exit threshold",
    )
    zscore_stop: float = Field(
        default=3.0,
        ge=2.0,
        le=5.0,
        description="Z-score stop-loss threshold",
    )

    # 변동성 파라미터
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
        description="연간 목표 변동성 (0.0~1.0)",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프",
    )

    # 시간 프레임
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
    )

    # 숏 모드
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

        if self.zscore_exit >= self.zscore_entry:
            msg = f"zscore_exit ({self.zscore_exit}) must be < zscore_entry ({self.zscore_entry})"
            raise ValueError(msg)

        if self.zscore_stop <= self.zscore_entry:
            msg = f"zscore_stop ({self.zscore_stop}) must be > zscore_entry ({self.zscore_entry})"
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        Returns:
            전략 계산에 필요한 최소 캔들 수
        """
        return self.formation_window + 1
