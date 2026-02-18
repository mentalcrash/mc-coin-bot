"""Fear-Greed Divergence 전략.

F&G 극단 + 가격 다이버전스 contrarian. 행동 편향 기반.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fear_divergence.config import FearDivergenceConfig
from src.strategy.fear_divergence.preprocessor import preprocess
from src.strategy.fear_divergence.signal import generate_signals


@register("fear-divergence")
class FearDivergenceStrategy(BaseStrategy):
    """Fear-Greed Divergence 전략 구현.

    F&G 극단 + 가격 다이버전스 → contrarian entry.
    """

    def __init__(self, config: FearDivergenceConfig | None = None) -> None:
        self._config = config or FearDivergenceConfig()

    @property
    def name(self) -> str:
        return "fear-divergence"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "oc_fear_greed"]

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
        config = FearDivergenceConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fg_fear_threshold": str(self._config.fg_fear_threshold),
            "fg_greed_threshold": str(self._config.fg_greed_threshold),
            "fg_deviation": str(self._config.fg_deviation),
            "er_min": str(self._config.er_min),
            "short_mode": self._config.short_mode.name,
        }
