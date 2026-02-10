"""OU Mean Reversion Strategy Configuration.

Ornstein-Uhlenbeck process 파라미터 추정 기반 mean reversion 전략 설정.

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


class OUMeanRevConfig(BaseModel):
    """OU Mean Reversion 전략 설정.

    Rolling window에서 OU process 파라미터를 OLS로 추정합니다.
    Half-life가 짧을 때만 mean reversion 거래를 수행합니다.

    Signal Formula:
        1. Rolling OLS: delta_price = a + b * price_lag
        2. theta = -log(1 + b), half_life = ln(2) / theta
        3. mu = -a / b (long-run mean)
        4. z-score = (price - mu) / rolling_std(price, ou_window)
        5. Long: z < -entry_zscore AND half_life < max_half_life
        6. Short: z > +entry_zscore AND half_life < max_half_life
        7. Exit: |z| < exit_zscore OR half_life > max_half_life OR timeout

    Attributes:
        ou_window: OU 추정 윈도우 (캔들 수)
        entry_zscore: 진입 Z-score 임계값
        exit_zscore: 청산 Z-score 임계값
        max_half_life: MR 거래 최대 half-life (캔들 수)
        exit_timeout_bars: 타임아웃 청산 바 수
        mom_lookback: 변동성 계산 lookback
        vol_target: 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        atr_period: ATR 계산 기간
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 드로다운 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
    """

    model_config = ConfigDict(frozen=True)

    # =========================================================================
    # OU Process 파라미터
    # =========================================================================
    ou_window: int = Field(
        default=120,
        ge=40,
        le=500,
        description="OU 추정 윈도우 (캔들 수, 120 = 4H * 120 = 20일)",
    )
    entry_zscore: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="진입 Z-score 임계값",
    )
    exit_zscore: float = Field(
        default=0.5,
        ge=0.1,
        le=1.5,
        description="청산 Z-score 임계값",
    )
    max_half_life: int = Field(
        default=30,
        ge=10,
        le=100,
        description="MR 거래 최대 half-life (캔들 수, 30 = 5일 at 4H)",
    )
    exit_timeout_bars: int = Field(
        default=30,
        ge=10,
        le=60,
        description="타임아웃 청산 바 수",
    )
    mom_lookback: int = Field(
        default=20,
        ge=5,
        le=60,
        description="변동성 계산 lookback (캔들 수)",
    )

    # =========================================================================
    # 변동성 공통 파라미터
    # =========================================================================
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="연간 목표 변동성",
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
        description="연환산 계수 (4H: 6*365=2190)",
    )

    # =========================================================================
    # 옵션
    # =========================================================================
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부 (권장: True)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=50,
        description="ATR 계산 기간 (Trailing Stop용)",
    )

    # =========================================================================
    # 숏 모드 설정
    # =========================================================================
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
        """설정 일관성 검증."""
        if self.exit_zscore >= self.entry_zscore:
            msg = f"exit_zscore ({self.exit_zscore}) must be < entry_zscore ({self.entry_zscore})"
            raise ValueError(msg)

        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) must be >= min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)

        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (캔들 수)."""
        return max(self.ou_window, self.mom_lookback, self.atr_period) + 1
