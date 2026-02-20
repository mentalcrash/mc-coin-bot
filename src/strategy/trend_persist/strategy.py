"""Trend Persistence Score 전략.

수익률 부호 일관성으로 추세 품질 측정 후 방향 진입.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.trend_persist.config import TrendPersistConfig
from src.strategy.trend_persist.preprocessor import preprocess
from src.strategy.trend_persist.signal import generate_signals


@register("trend-persist")
class TrendPersistStrategy(BaseStrategy):
    """Trend Persistence Score 전략 구현.

    수익률 부호 일관성(% positive days)으로 추세 지속성 측정.
    """

    def __init__(self, config: TrendPersistConfig | None = None) -> None:
        self._config = config or TrendPersistConfig()

    @property
    def name(self) -> str:
        return "trend-persist"

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
        config = TrendPersistConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "persist_window": str(self._config.persist_window),
            "long_threshold": str(self._config.long_threshold),
            "short_threshold": str(self._config.short_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
