"""Anchored Trend-Following 3H 전략.

Anchor-Mom(12H G5 PASS)의 3H TF 적응.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.atf_3h.config import Atf3hConfig
from src.strategy.atf_3h.preprocessor import preprocess
from src.strategy.atf_3h.signal import generate_signals


@register("atf-3h")
class Atf3hStrategy(BaseStrategy):
    """Anchored Trend-Following 3H 전략 구현.

    심리적 앵커링 효과(rolling high nearness)의 3H TF 불변성 검증.
    """

    def __init__(self, config: Atf3hConfig | None = None) -> None:
        self._config = config or Atf3hConfig()

    @property
    def name(self) -> str:
        return "atf-3h"

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
        config = Atf3hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "nearness_lookback": str(self._config.nearness_lookback),
            "mom_lookback": str(self._config.mom_lookback),
            "strong_nearness": str(self._config.strong_nearness),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
