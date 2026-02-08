"""RSI Crossover Strategy Configuration.

RSI 크로스오버 기반 전략의 설정을 정의하는 Pydantic 모델입니다.
RSI가 과매도(30) 영역을 상향 돌파 시 롱, 과매수(70) 영역을 하향 돌파 시 숏,
40/60 도달 시 청산하는 평균회귀 전략입니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.tsmom.config import ShortMode


class RSICrossoverConfig(BaseModel):
    """RSI Crossover 전략 설정.

    RSI crossover 기반 진입/청산 전략의 핵심 파라미터를 정의합니다.
    과매도/과매수 레벨 크로스오버로 진입하고,
    중립 레벨에서 청산합니다.

    Signal Logic:
        - Long Entry: RSI가 entry_oversold(30) 상향 돌파
        - Short Entry: RSI가 entry_overbought(70) 하향 돌파
        - Long Exit: RSI가 exit_long(60) 도달
        - Short Exit: RSI가 exit_short(40) 도달

    Attributes:
        rsi_period: RSI 계산 기간
        entry_oversold: 롱 진입 임계값 (이 값을 상향 크로스)
        entry_overbought: 숏 진입 임계값 (이 값을 하향 크로스)
        exit_long: 롱 청산 임계값
        exit_short: 숏 청산 임계값
        vol_window: 변동성 계산 윈도우
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수

    Example:
        >>> config = RSICrossoverConfig(
        ...     rsi_period=14,
        ...     entry_oversold=30.0,
        ...     entry_overbought=70.0,
        ...     vol_target=0.25,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ======================================================================
    # RSI Parameters
    # ======================================================================
    rsi_period: int = Field(
        default=14,
        ge=5,
        le=30,
        description="RSI 계산 기간 (Wilder's RSI)",
    )

    # ======================================================================
    # Entry/Exit Thresholds
    # ======================================================================
    entry_oversold: float = Field(
        default=30.0,
        ge=10.0,
        le=45.0,
        description="롱 진입 임계값 (RSI가 이 값을 상향 크로스)",
    )
    entry_overbought: float = Field(
        default=70.0,
        ge=55.0,
        le=90.0,
        description="숏 진입 임계값 (RSI가 이 값을 하향 크로스)",
    )
    exit_long: float = Field(
        default=60.0,
        ge=45.0,
        le=80.0,
        description="롱 청산 임계값 (RSI가 이 값 도달 시 청산)",
    )
    exit_short: float = Field(
        default=40.0,
        ge=20.0,
        le=55.0,
        description="숏 청산 임계값 (RSI가 이 값 도달 시 청산)",
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
        default=0.25,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성 (0.25 = 25%)",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프 (0으로 나누기 방지)",
    )
    annualization_factor: float = Field(
        default=2190.0,
        gt=0,
        description="연환산 계수 (4시간봉: 2190, 일봉: 365)",
    )

    # ======================================================================
    # Short Mode
    # ======================================================================
    short_mode: ShortMode = Field(
        default=ShortMode.FULL,
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

        검증 규칙:
            - entry_oversold < exit_short < exit_long < entry_overbought
            - vol_target >= min_volatility

        Returns:
            검증된 self

        Raises:
            ValueError: 설정이 비합리적일 경우
        """
        # Threshold 순서 검증: oversold < exit_short < exit_long < overbought
        if not (self.entry_oversold < self.exit_short < self.exit_long < self.entry_overbought):
            msg = (
                f"Threshold order must be: "
                f"entry_oversold ({self.entry_oversold}) < "
                f"exit_short ({self.exit_short}) < "
                f"exit_long ({self.exit_long}) < "
                f"entry_overbought ({self.entry_overbought})"
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

        RSI + vol_window + shift(2) 여유분.

        Returns:
            필요한 캔들 수
        """
        return self.rsi_period + self.vol_window + 2

    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs: object) -> RSICrossoverConfig:
        """타임프레임에 맞는 기본 설정 생성.

        Args:
            timeframe: 타임프레임 문자열 (예: "1h", "4h", "1d")
            **kwargs: 추가 설정 오버라이드

        Returns:
            해당 타임프레임에 최적화된 RSICrossoverConfig
        """
        annualization_map: dict[str, float] = {
            "1m": 525600.0,
            "5m": 105120.0,
            "15m": 35040.0,
            "1h": 8760.0,
            "4h": 2190.0,
            "1d": 365.0,
        }

        annualization = annualization_map.get(timeframe, 2190.0)

        return cls(
            annualization_factor=annualization,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def conservative(cls) -> RSICrossoverConfig:
        """보수적 설정 (넓은 entry 범위, 낮은 vol target).

        Returns:
            보수적 파라미터의 RSICrossoverConfig
        """
        return cls(
            rsi_period=14,
            entry_oversold=25.0,
            entry_overbought=75.0,
            exit_long=55.0,
            exit_short=45.0,
            vol_target=0.15,
            min_volatility=0.08,
        )

    @classmethod
    def aggressive(cls) -> RSICrossoverConfig:
        """공격적 설정 (좁은 entry 범위, 높은 vol target).

        Returns:
            공격적 파라미터의 RSICrossoverConfig
        """
        return cls(
            rsi_period=10,
            entry_oversold=35.0,
            entry_overbought=65.0,
            exit_long=55.0,
            exit_short=45.0,
            vol_target=0.35,
            min_volatility=0.05,
        )
