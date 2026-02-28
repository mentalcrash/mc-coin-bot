"""Trend Factor Multi-Horizon 전략.

JFQA 2024 Trend Factor: 5-horizon risk-adjusted return 합산 multi-scale momentum.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.trend_factor_12h.config import TrendFactorConfig
from src.strategy.trend_factor_12h.preprocessor import preprocess
from src.strategy.trend_factor_12h.signal import generate_signals


@register("trend-factor-12h")
class TrendFactorStrategy(BaseStrategy):
    """Trend Factor Multi-Horizon 전략 구현.

    5개 horizon(5/10/20/40/80)에서 risk-adjusted return(ret/vol)을 합산하여
    multi-scale momentum consensus를 포착. t-stat 대비 직접 합산으로 차별화.
    """

    def __init__(self, config: TrendFactorConfig | None = None) -> None:
        self._config = config or TrendFactorConfig()

    @property
    def name(self) -> str:
        return "trend-factor-12h"

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
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = TrendFactorConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "horizons": (
                f"{self._config.horizon_1}/{self._config.horizon_2}/"
                f"{self._config.horizon_3}/{self._config.horizon_4}/"
                f"{self._config.horizon_5}"
            ),
            "entry_threshold": str(self._config.entry_threshold),
            "tanh_scale": str(self._config.tanh_scale),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
