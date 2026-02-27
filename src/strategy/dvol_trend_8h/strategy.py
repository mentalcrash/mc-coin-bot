"""DVOL-Trend 8H 전략.

Deribit DVOL percentile 기반 position sizing + 8H multi-scale Donchian channel.
BTC/ETH only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.dvol_trend_8h.config import DvolTrend8hConfig
from src.strategy.dvol_trend_8h.preprocessor import preprocess
from src.strategy.dvol_trend_8h.signal import generate_signals


@register("dvol-trend-8h")
class DvolTrend8hStrategy(BaseStrategy):
    """DVOL-Trend 8H 전략 구현.

    Deribit DVOL(내재변동성) percentile로 IV regime을 판단하여
    position sizing을 조절하고, 3-scale Donchian Channel(22/66/132)
    breakout consensus로 추세 방향을 결정한다. BTC/ETH only.
    """

    def __init__(self, config: DvolTrend8hConfig | None = None) -> None:
        self._config = config or DvolTrend8hConfig()

    @property
    def name(self) -> str:
        return "dvol-trend-8h"

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
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = DvolTrend8hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "scales": (
                f"{self._config.dc_scale_short}/"
                f"{self._config.dc_scale_mid}/"
                f"{self._config.dc_scale_long}"
            ),
            "entry_threshold": str(self._config.entry_threshold),
            "vol_target": str(self._config.vol_target),
            "dvol_thresholds": (
                f"{self._config.dvol_low_threshold}/{self._config.dvol_high_threshold}"
            ),
            "dvol_multipliers": (
                f"{self._config.dvol_low_multiplier}/{self._config.dvol_high_multiplier}"
            ),
            "short_mode": self._config.short_mode.name,
        }
