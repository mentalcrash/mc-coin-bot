"""Fear & Greed Delta 전략.

F&G Index 변화율 기반 센티먼트 모멘텀 활용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fg_delta.config import FgDeltaConfig
from src.strategy.fg_delta.preprocessor import preprocess
from src.strategy.fg_delta.signal import generate_signals


@register("fg-delta")
class FgDeltaStrategy(BaseStrategy):
    """Fear & Greed Delta 전략 구현.

    F&G 변화율(delta)로 센티먼트 모멘텀 포착.
    """

    def __init__(self, config: FgDeltaConfig | None = None) -> None:
        self._config = config or FgDeltaConfig()

    @property
    def name(self) -> str:
        return "fg-delta"

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
        }

    @classmethod
    def from_params(cls, **params: Any) -> BaseStrategy:
        config = FgDeltaConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fg_delta_window": str(self._config.fg_delta_window),
            "fg_smooth_window": str(self._config.fg_smooth_window),
            "delta_threshold": str(self._config.delta_threshold),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
