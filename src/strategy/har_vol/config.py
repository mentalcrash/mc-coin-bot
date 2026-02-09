"""HAR Volatility Overlay Strategy Configuration.

이 모듈은 HAR-RV (Heterogeneous Autoregressive Realized Volatility) 전략의
설정을 정의하는 Pydantic 모델을 제공합니다.
HAR 모델로 변동성을 예측하고 vol surprise(realized - forecast)로 매매합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class HARVolConfig(BaseModel):
    """HAR Volatility Overlay 전략 설정.

    HAR-RV 모델을 사용하여 변동성을 예측하고, 실현 변동성과 예측 변동성의
    차이(vol surprise)를 기반으로 매매 시그널을 생성합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. Parkinson volatility 계산 (high/low range-based)
        2. HAR features: rv_daily, rv_weekly, rv_monthly (rolling means)
        3. Rolling OLS로 HAR forecast 생성
        4. vol_surprise = realized - forecast
        5. vol_surprise > threshold → momentum (follow recent returns)
        6. vol_surprise < -threshold → mean-reversion (-recent returns) * 0.5
        7. strength = direction * vol_scalar

    Attributes:
        daily_window: Daily RV 윈도우 (캔들 수)
        weekly_window: Weekly RV 윈도우 (캔들 수)
        monthly_window: Monthly RV 윈도우 (캔들 수)
        training_window: OLS 학습 윈도우 (캔들 수)
        vol_surprise_threshold: Vol surprise 임계값
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = HARVolConfig(
        ...     daily_window=1,
        ...     weekly_window=5,
        ...     monthly_window=22,
        ...     training_window=252,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # HAR 모델 파라미터
    # =========================================================================
    daily_window: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Daily RV 윈도우 (캔들 수)",
    )
    weekly_window: int = Field(
        default=5,
        ge=3,
        le=10,
        description="Weekly RV 윈도우 (캔들 수)",
    )
    monthly_window: int = Field(
        default=22,
        ge=15,
        le=30,
        description="Monthly RV 윈도우 (캔들 수)",
    )
    training_window: int = Field(
        default=252,
        ge=60,
        le=504,
        description="OLS 학습 윈도우 (캔들 수)",
    )

    # =========================================================================
    # Vol Surprise 임계값
    # =========================================================================
    vol_surprise_threshold: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Vol surprise 임계값 (0이면 모든 surprise에 반응)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.0~1.0, 예: 0.35 = 35%)",
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

        if self.training_window <= self.monthly_window:
            msg = (
                f"training_window ({self.training_window}) must be > "
                f"monthly_window ({self.monthly_window})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수).

        training_window + monthly_window + 1 (rolling OLS + features 계산에 필요).

        Returns:
            필요한 캔들 수
        """
        return self.training_window + self.monthly_window + 1
