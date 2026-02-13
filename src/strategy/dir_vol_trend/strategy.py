"""Directional Volume Trend 전략.

Up-bar/down-bar 거래량 비율 기반 추세추종.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.dir_vol_trend.config import DirVolTrendConfig
from src.strategy.dir_vol_trend.preprocessor import preprocess
from src.strategy.dir_vol_trend.signal import generate_signals


@register("dir-vol-trend")
class DirVolTrendStrategy(BaseStrategy):
    """Directional Volume Trend 전략 구현.

    Up/down bar 거래량 비율로 방향적 conviction 측정.
    """

    def __init__(self, config: DirVolTrendConfig | None = None) -> None:
        self._config = config or DirVolTrendConfig()

    @property
    def name(self) -> str:
        return "dir-vol-trend"

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
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = DirVolTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "dvt_window": str(self._config.dvt_window),
            "dvt_long_threshold": str(self._config.dvt_long_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
