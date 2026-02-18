"""Stablecoin Momentum Trend 전략.

Stablecoin 공급 변화율 z-score와 EMA cross를 결합하여 자금 유입/유출 모멘텀을 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.stab_mom_trend.config import StabMomTrendConfig
from src.strategy.stab_mom_trend.preprocessor import preprocess
from src.strategy.stab_mom_trend.signal import generate_signals


@register("stab-mom-trend")
class StabMomTrendStrategy(BaseStrategy):
    """Stablecoin Momentum Trend 전략 구현.

    Stablecoin 유통량 증가 (z-score) + EMA 상승 추세 → 매수
    Stablecoin 유통량 감소 (z-score) + EMA 하락 추세 → 매도 (HEDGE_ONLY)
    """

    def __init__(self, config: StabMomTrendConfig | None = None) -> None:
        self._config = config or StabMomTrendConfig()

    @property
    def name(self) -> str:
        return "stab-mom-trend"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "oc_stablecoin_total_circulating_usd"]

    @property
    def config(self) -> BaseModel:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    @classmethod
    def recommended_config(cls) -> dict[str, Any]:
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "max_leverage_cap": 2.0,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = StabMomTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "stab_change_period": str(self._config.stab_change_period),
            "zscore_window": str(self._config.zscore_window),
            "ema_fast_period": str(self._config.ema_fast_period),
            "ema_slow_period": str(self._config.ema_slow_period),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
