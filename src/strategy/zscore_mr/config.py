"""Z-Score Mean Reversion Strategy Configuration.

동적 lookback z-score 기반 평균회귀 전략의 설정을 정의하는 Pydantic 모델입니다.
변동성 레짐에 따라 short/long lookback을 전환하여 적응적 z-score를 계산합니다.

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


class ZScoreMRConfig(BaseModel):
    """Z-Score 평균회귀 전략 설정.

    변동성 레짐에 따라 short/long lookback을 전환하여
    적응적 z-score 기반 평균회귀 시그널을 생성합니다.

    Signal Formula:
        1. vol_pct = returns.rolling(vol_regime_lookback).std()
                     .rolling(vol_rank_lookback).rank(pct=True)
        2. z_short = (close - close.rolling(short_lb).mean()) / rolling_std
        3. z_long  = (close - close.rolling(long_lb).mean()) / rolling_std
        4. z_score = where(vol_pct > high_vol_pct, z_short, z_long)
        5. long_entry  = z_score.shift(1) < -entry_z
        6. short_entry = z_score.shift(1) > entry_z

    Attributes:
        short_lookback: 고변동성 레짐에서 사용하는 단기 z-score lookback
        long_lookback: 저변동성 레짐에서 사용하는 장기 z-score lookback
        entry_z: z-score 진입 임계값
        exit_z: z-score 청산 임계값 (평균 복귀)

    Example:
        >>> config = ZScoreMRConfig(
        ...     short_lookback=20,
        ...     long_lookback=60,
        ...     entry_z=2.0,
        ...     exit_z=0.5,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # Z-Score Lookback
    # ======================================================================
    short_lookback: int = Field(
        default=20,
        ge=5,
        le=100,
        description="고변동성 레짐에서 사용하는 단기 z-score lookback",
    )
    long_lookback: int = Field(
        default=60,
        ge=20,
        le=365,
        description="저변동성 레짐에서 사용하는 장기 z-score lookback",
    )

    # ======================================================================
    # Entry / Exit Thresholds
    # ======================================================================
    entry_z: float = Field(
        default=2.0,
        ge=0.5,
        le=4.0,
        description="z-score 진입 임계값 (절대값 기준)",
    )
    exit_z: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="z-score 청산 임계값 (평균 복귀 시 포지션 해제)",
    )

    # ======================================================================
    # Volatility Regime Detection
    # ======================================================================
    vol_regime_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 추정 윈도우 (수익률 rolling std)",
    )
    vol_rank_lookback: int = Field(
        default=252,
        ge=60,
        le=500,
        description="변동성 순위 percentile 윈도우",
    )
    high_vol_percentile: float = Field(
        default=0.7,
        ge=0.3,
        le=0.9,
        description="고변동성 레짐 판단 임계값 (이 이상이면 short_lookback 사용)",
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
        default=0.20,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.20 = 20%, 평균회귀는 추세추종보다 낮게)",
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
    # ATR Stop (Portfolio 레이어에 위임)
    # ======================================================================
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (trailing stop 참조용)",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
        description="숏 포지션 처리 모드 (평균회귀는 기본 FULL - 양방향 시그널)",
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
        # entry_z > exit_z
        if self.entry_z <= self.exit_z:
            msg = (
                f"entry_z ({self.entry_z}) must be > "
                f"exit_z ({self.exit_z})"
            )
            raise ValueError(msg)

        # long_lookback > short_lookback
        if self.long_lookback <= self.short_lookback:
            msg = (
                f"long_lookback ({self.long_lookback}) must be > "
                f"short_lookback ({self.short_lookback})"
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

        Returns:
            필요한 캔들 수
        """
        return (
            max(
                self.long_lookback,
                self.vol_rank_lookback,
                self.vol_window,
                self.atr_period,
            )
            + 1
        )

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> "ZScoreMRConfig":
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 ZScoreMRConfig
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
