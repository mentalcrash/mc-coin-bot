"""Regime-Adaptive TSMOM Configuration.

TSMOM 파라미터 + 레짐별 적응적 파라미터를 정의합니다.

Rules Applied:
    - #11 Pydantic Modeling: frozen=True, field validators
    - #10 Python Standards: Modern typing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.regime.config import RegimeDetectorConfig
from src.strategy.tsmom.config import ShortMode

if TYPE_CHECKING:
    from src.strategy.tsmom.config import TSMOMConfig


class RegimeTSMOMConfig(BaseModel):
    """Regime-Adaptive TSMOM 전략 설정.

    기본 TSMOM 파라미터에 레짐별 적응적 파라미터를 추가합니다.
    trending → 공격적, ranging → 보수적, volatile → 초보수.

    Attributes:
        lookback: 모멘텀 계산 기간 (캔들 수)
        vol_window: 변동성 계산 윈도우 (캔들 수)
        vol_target: 기본 연간 목표 변동성
        min_volatility: 최소 변동성 클램프
        annualization_factor: 연환산 계수
        use_log_returns: 로그 수익률 사용 여부
        momentum_smoothing: 모멘텀 스무딩 윈도우
        short_mode: 숏 포지션 처리 모드
        hedge_threshold: 헤지 숏 활성화 임계값
        hedge_strength_ratio: 헤지 숏 강도 비율
        regime: 레짐 감지 설정
        trending_vol_target: trending에서의 vol_target
        ranging_vol_target: ranging에서의 vol_target
        volatile_vol_target: volatile에서의 vol_target
        trending_leverage_scale: trending에서의 레버리지 스케일
        ranging_leverage_scale: ranging에서의 레버리지 스케일
        volatile_leverage_scale: volatile에서의 레버리지 스케일

    Example:
        >>> config = RegimeTSMOMConfig(
        ...     lookback=30,
        ...     trending_vol_target=0.40,
        ...     ranging_vol_target=0.15,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # ── 기본 TSMOM 파라미터 ──
    lookback: int = Field(
        default=30,
        ge=6,
        le=365,
        description="모멘텀 계산 기간 (캔들 수)",
    )
    vol_window: int = Field(
        default=30,
        ge=6,
        le=365,
        description="변동성 계산 윈도우 (캔들 수)",
    )
    vol_target: float = Field(
        default=0.35,
        ge=0.05,
        le=1.0,
        description="기본 연간 목표 변동성",
    )
    min_volatility: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="최소 변동성 클램프",
    )
    annualization_factor: float = Field(
        default=365.0,
        gt=0,
        description="연환산 계수 (일봉: 365)",
    )
    use_log_returns: bool = Field(
        default=True,
        description="로그 수익률 사용 여부",
    )
    momentum_smoothing: int | None = Field(
        default=None,
        ge=2,
        le=24,
        description="모멘텀 스무딩 윈도우 (선택적)",
    )
    short_mode: ShortMode = Field(
        default=ShortMode.HEDGE_ONLY,
        description="숏 포지션 처리 모드",
    )
    hedge_threshold: float = Field(
        default=-0.07,
        ge=-0.30,
        le=-0.05,
        description="헤지 숏 활성화 드로다운 임계값",
    )
    hedge_strength_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="헤지 숏 강도 비율",
    )

    # ── 레짐 감지 설정 ──
    regime: RegimeDetectorConfig = Field(
        default_factory=RegimeDetectorConfig,
        description="레짐 감지 설정",
    )

    # ── 레짐별 적응적 파라미터 ──
    trending_vol_target: float = Field(
        default=0.40,
        ge=0.05,
        le=1.0,
        description="trending 레짐에서의 vol_target (공격적)",
    )
    ranging_vol_target: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="ranging 레짐에서의 vol_target (보수적)",
    )
    volatile_vol_target: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="volatile 레짐에서의 vol_target (초보수)",
    )
    trending_leverage_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="trending 레짐에서의 레버리지 스케일",
    )
    ranging_leverage_scale: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="ranging 레짐에서의 레버리지 스케일",
    )
    volatile_leverage_scale: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="volatile 레짐에서의 레버리지 스케일",
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """설정 일관성 검증."""
        if self.momentum_smoothing is not None and self.momentum_smoothing > self.lookback:
            msg = (
                f"momentum_smoothing ({self.momentum_smoothing}) must be "
                f"<= lookback ({self.lookback})"
            )
            raise ValueError(msg)
        if self.vol_target < self.min_volatility:
            msg = (
                f"vol_target ({self.vol_target}) should be >= "
                f"min_volatility ({self.min_volatility})"
            )
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """필요한 워밍업 기간 (TSMOM + regime warmup).

        Returns:
            필요한 캔들 수
        """
        tsmom_warmup = max(self.lookback, self.vol_window) + 1
        regime_warmup = self.regime.rv_long_window + 5
        return max(tsmom_warmup, regime_warmup)

    def to_tsmom_config(self) -> TSMOMConfig:
        """TSMOM 전처리용 설정 변환.

        Returns:
            기본 TSMOM 파라미터만 포함하는 TSMOMConfig
        """
        from src.strategy.tsmom.config import TSMOMConfig

        return TSMOMConfig(
            lookback=self.lookback,
            vol_window=self.vol_window,
            vol_target=self.vol_target,
            min_volatility=self.min_volatility,
            annualization_factor=self.annualization_factor,
            use_log_returns=self.use_log_returns,
            momentum_smoothing=self.momentum_smoothing,
            short_mode=self.short_mode,
            hedge_threshold=self.hedge_threshold,
            hedge_strength_ratio=self.hedge_strength_ratio,
        )
