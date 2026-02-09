"""Funding Rate Carry Strategy Configuration.

이 모듈은 Funding Rate Carry 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
Positive funding rate -> short (receive carry), Negative -> long.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class FundingCarryConfig(BaseModel):
    """Funding Rate Carry 전략 설정.

    Funding rate의 부호를 기반으로 캐리 시그널을 생성합니다.
    Positive FR -> short (funding 수취), Negative FR -> long.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. avg_funding_rate = rolling_mean(funding_rate, lookback)
        2. funding_zscore = (avg_fr - rolling_mean) / rolling_std
        3. direction = -sign(avg_funding_rate) (positive FR -> short)
        4. strength = direction * vol_scalar
        5. entry_threshold: |avg_fr| > threshold일 때만 진입

    Attributes:
        lookback: 평균 funding rate 계산 기간 (8h 단위)
        zscore_window: Z-score 정규화 윈도우
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        entry_threshold: 진입을 위한 최소 |FR| 값
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드

    Example:
        >>> config = FundingCarryConfig(
        ...     lookback=3,
        ...     zscore_window=90,
        ...     vol_target=0.35,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # Funding Rate 파라미터
    # =========================================================================
    lookback: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Average funding rate period (8h units)",
    )
    zscore_window: int = Field(
        default=90,
        ge=10,
        le=365,
        description="Z-score normalization window",
    )
    entry_threshold: float = Field(
        default=0.0001,
        ge=0.0,
        le=0.01,
        description="Minimum |FR| for entry (0=always enter)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_window: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Volatility calculation window",
    )
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="Annualized volatility target",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum volatility clamp (prevent division by zero)",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="Annualization factor (daily: 365, 4h: 2190, hourly: 8760)",
    )

    # =========================================================================
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="Use log returns (recommended: True)",
    )
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="Short mode (FULL for carry strategy)",
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
        return max(self.zscore_window, self.vol_window) + 1
