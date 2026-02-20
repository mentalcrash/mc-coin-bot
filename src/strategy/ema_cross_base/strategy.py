"""EMA Cross Base 전략 — 순수 20/100 EMA 크로스오버."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.ema_cross_base.config import EmaCrossBaseConfig
from src.strategy.ema_cross_base.preprocessor import preprocess
from src.strategy.ema_cross_base.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("ema-cross-base")
class EmaCrossBaseStrategy(BaseStrategy):
    """순수 20/100 EMA 크로스오버 베이스라인 전략.

    EMA(20) > EMA(100) → Long, EMA(20) < EMA(100) → Short.
    모멘텀 지속성 활용 — TSMOM 대비 베이스라인 비교 목적.
    """

    def __init__(self, config: EmaCrossBaseConfig | None = None) -> None:
        self._config = config or EmaCrossBaseConfig()

    @property
    def name(self) -> str:
        return "ema-cross-base"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

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
    def from_params(cls, **params: Any) -> EmaCrossBaseStrategy:
        config = EmaCrossBaseConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fast_period": str(self._config.fast_period),
            "slow_period": str(self._config.slow_period),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
