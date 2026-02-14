"""Regime-Adaptive Dual-Alpha Ensemble 전략 설정.

CTREND(기술 ML 모멘텀) + regime-mf-mr(레짐 게이트 평균회귀)을
inverse_volatility 또는 strategy_momentum으로 결합하는 메타 앙상블.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.strategy.ensemble.config import AggregationMethod, ShortMode

_NUM_SUB_STRATEGIES = 2


class EnsRegimeDualConfig(BaseModel):
    """Regime-Adaptive Dual-Alpha Ensemble 설정.

    서로 다른 alpha 소스(trend ML vs regime-gated MR)를 결합하여
    전 시장 환경에 대응. 극도의 분산 효과 기대 (상관 ~0.03).

    Attributes:
        aggregation: 시그널 집계 방법
        vol_lookback: inverse_volatility lookback
        momentum_lookback: strategy_momentum lookback
        top_n: strategy_momentum top N
        min_agreement: majority_vote 최소 합의 비율
        vol_target: 연간 목표 변동성
        vol_window: 변동성 계산 윈도우
        annualization_factor: 연환산 계수
        short_mode: 숏 포지션 처리 모드
        ctrend_weight: CTREND 서브전략 정적 가중치
        regime_mf_mr_weight: regime-mf-mr 서브전략 정적 가중치
    """

    model_config = ConfigDict(frozen=True)

    # --- Aggregation ---
    aggregation: AggregationMethod = Field(
        default=AggregationMethod.INVERSE_VOLATILITY,
    )

    # --- Aggregation Parameters ---
    vol_lookback: int = Field(default=63, ge=5, le=504)
    momentum_lookback: int = Field(default=126, ge=10, le=504)
    top_n: int = Field(default=2, ge=1)
    min_agreement: float = Field(default=0.5, gt=0.0, le=1.0)

    # --- Volatility Scaling ---
    vol_target: float = Field(default=0.35, ge=0.05, le=1.0)
    vol_window: int = Field(default=30, ge=5, le=252)
    annualization_factor: float = Field(default=365.0, gt=0)

    # --- Short Mode ---
    short_mode: ShortMode = Field(default=ShortMode.FULL)

    # --- Sub-Strategy Weights ---
    ctrend_weight: float = Field(default=1.0, gt=0.0)
    regime_mf_mr_weight: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_cross_fields(self) -> Self:
        if (
            self.aggregation == AggregationMethod.STRATEGY_MOMENTUM
            and self.top_n > _NUM_SUB_STRATEGIES
        ):
            msg = f"top_n ({self.top_n}) cannot exceed number of sub-strategies ({_NUM_SUB_STRATEGIES})"
            raise ValueError(msg)
        return self

    def warmup_periods(self) -> int:
        """시그널 생성에 필요한 최소 warmup bar 수."""
        agg_lookback = 0
        if self.aggregation == AggregationMethod.INVERSE_VOLATILITY:
            agg_lookback = self.vol_lookback
        elif self.aggregation == AggregationMethod.STRATEGY_MOMENTUM:
            agg_lookback = self.momentum_lookback
        return agg_lookback + self.vol_window + 1
