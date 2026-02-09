"""Vol-Adaptive Trend Strategy Configuration.

이 모듈은 Vol-Adaptive Trend 전략의 설정을 정의하는 Pydantic 모델을 제공합니다.
EMA crossover + RSI confirm + ADX filter + ATR vol-target sizing을 위한
모든 설정이 포함됩니다.

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


class VolAdaptiveConfig(BaseModel):
    """Vol-Adaptive Trend 전략 설정.

    EMA crossover로 추세 방향을 판별하고, RSI와 ADX로 확인/필터링한 후,
    ATR 기반 변동성 타겟팅으로 포지션 사이징하는 전략의 모든 파라미터를 정의합니다.

    Note:
        레버리지 제한(max_leverage_cap)과 시그널 필터링(rebalance_threshold)은
        PortfolioManagerConfig에서 관리합니다. 전략은 순수한 시그널만 생성합니다.

    Signal Formula:
        1. trend = sign(EMA_fast - EMA_slow)
        2. rsi_confirm = RSI confirms trend direction
        3. adx_filter = ADX > threshold (strong trend)
        4. direction = trend * rsi_confirm * adx_filter
        5. strength = direction * vol_scalar

    Attributes:
        ema_fast: 빠른 EMA 기간
        ema_slow: 느린 EMA 기간
        rsi_period: RSI 계산 기간
        rsi_upper: RSI 상단 임계값 (롱 확인)
        rsi_lower: RSI 하단 임계값 (숏 확인)
        adx_period: ADX 계산 기간
        adx_threshold: ADX 최소 임계값 (추세 강도 필터)
        vol_window: 변동성 추정 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        atr_period: ATR 계산 기간
        use_log_returns: 로그 수익률 사용 여부
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율

    Example:
        >>> config = VolAdaptiveConfig(
        ...     ema_fast=10,
        ...     ema_slow=50,
        ...     adx_threshold=25.0,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # EMA Crossover 파라미터
    # =========================================================================
    ema_fast: int = Field(
        default=10,
        ge=3,
        le=30,
        description="빠른 EMA 기간 (캔들 수)",
    )
    ema_slow: int = Field(
        default=50,
        ge=20,
        le=200,
        description="느린 EMA 기간 (캔들 수)",
    )

    # =========================================================================
    # RSI 파라미터
    # =========================================================================
    rsi_period: int = Field(
        default=14,
        ge=5,
        le=30,
        description="RSI 계산 기간",
    )
    rsi_upper: float = Field(
        default=50.0,
        ge=30.0,
        le=80.0,
        description="RSI 상단 임계값 (롱 확인용, RSI > rsi_upper → 롱 확인)",
    )
    rsi_lower: float = Field(
        default=50.0,
        ge=20.0,
        le=70.0,
        description="RSI 하단 임계값 (숏 확인용, RSI < rsi_lower → 숏 확인)",
    )

    # =========================================================================
    # ADX 파라미터
    # =========================================================================
    adx_period: int = Field(
        default=14,
        ge=5,
        le=30,
        description="ADX 계산 기간",
    )
    adx_threshold: float = Field(
        default=20.0,
        ge=10.0,
        le=40.0,
        description="ADX 최소 임계값 (추세 강도 필터, ADX > threshold → 강한 추세)",
    )

    # =========================================================================
    # 변동성 타겟팅 파라미터
    # =========================================================================
    vol_window: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 추정 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (예: 0.40 = 40%)",
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
    # ATR 및 옵션
    # =========================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (Trailing Stop용)",
    )
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
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

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증.

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # ema_slow > ema_fast
        if self.ema_slow <= self.ema_fast:
            msg = f"ema_slow ({self.ema_slow}) must be > ema_fast ({self.ema_fast})"
            raise ValueError(msg)

        # vol_target >= min_volatility
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
        return (
            max(
                self.ema_slow,
                self.vol_window,
                self.adx_period,
                self.rsi_period,
                self.atr_period,
            )
            + 1
        )
