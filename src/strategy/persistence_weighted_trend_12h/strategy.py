"""Persistence-Weighted-Trend 12H 전략.

ER x FD x TrendStrength 복합 추세 품질 점수로 모멘텀 가중.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.persistence_weighted_trend_12h.config import PersistenceWeightedTrendConfig
from src.strategy.persistence_weighted_trend_12h.preprocessor import preprocess
from src.strategy.persistence_weighted_trend_12h.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("persistence-weighted-trend-12h")
class PersistenceWeightedTrend12hStrategy(BaseStrategy):
    """ER x FD x TrendStrength 복합 추세 품질 가중 전략."""

    def __init__(self, config: PersistenceWeightedTrendConfig | None = None) -> None:
        self._config = config or PersistenceWeightedTrendConfig()

    @property
    def name(self) -> str:
        return "persistence-weighted-trend-12h"

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
        """PortfolioManagerConfig kwargs."""
        return {
            "stop_loss_pct": 0.10,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 3.0,
            "rebalance_threshold": 0.10,
            "use_intrabar_stop": True,
            "use_intrabar_trailing_stop": False,
        }

    @classmethod
    def from_params(cls, **params: Any) -> PersistenceWeightedTrend12hStrategy:
        """Parameter sweep용 팩토리."""
        config = PersistenceWeightedTrendConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        """CLI startup panel 표시 정보."""
        return {
            "scale_short": str(self._config.scale_short),
            "scale_mid": str(self._config.scale_mid),
            "scale_long": str(self._config.scale_long),
            "persistence_threshold": str(self._config.persistence_threshold),
            "mom_lookback": str(self._config.mom_lookback),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
