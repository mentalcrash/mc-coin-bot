"""Conviction-Filtered Trend 전략.

UP->UP 레짐 전환을 모멘텀 진입 게이트로 사용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.cft_2h.config import Cft2hConfig
from src.strategy.cft_2h.preprocessor import preprocess
from src.strategy.cft_2h.signal import generate_signals


@register("cft-2h")
class Cft2hStrategy(BaseStrategy):
    """Conviction-Filtered Trend 전략 구현.

    레짐을 필터로 사용하여 모멘텀 conviction 확인 후 진입.
    """

    def __init__(self, config: Cft2hConfig | None = None) -> None:
        self._config = config or Cft2hConfig()

    @property
    def name(self) -> str:
        return "cft-2h"

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
        config = Cft2hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "regime_window": str(self._config.regime_window),
            "mom_lookback": str(self._config.mom_lookback),
            "regime_up_threshold": str(self._config.regime_up_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
