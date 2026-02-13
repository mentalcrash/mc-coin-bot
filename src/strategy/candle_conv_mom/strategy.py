"""Candle Conviction Momentum 전략.

Full-bodied candle 기반 rolling conviction으로 trend continuation 포착.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.candle_conv_mom.config import CandleConvMomConfig
from src.strategy.candle_conv_mom.preprocessor import preprocess
from src.strategy.candle_conv_mom.signal import generate_signals


@register("candle-conv-mom")
class CandleConvMomStrategy(BaseStrategy):
    """Candle Conviction Momentum 전략 구현.

    Rolling conviction(direction x body_ratio)으로 시장 합의 방향 포착.
    """

    def __init__(self, config: CandleConvMomConfig | None = None) -> None:
        self._config = config or CandleConvMomConfig()

    @property
    def name(self) -> str:
        return "candle-conv-mom"

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
        config = CandleConvMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "conv_window": str(self._config.conv_window),
            "conv_threshold": str(self._config.conv_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
