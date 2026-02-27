"""Macro-Gated Patient Trend (4H) — strategy entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.macro_patience_4h.config import MacroPatience4hConfig
from src.strategy.macro_patience_4h.preprocessor import preprocess
from src.strategy.macro_patience_4h.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals


@register("macro-patience-4h")
class MacroPatience4hStrategy(BaseStrategy):
    """Macro-Gated Patient Trend strategy (4H).

    Uses macro data (DXY/VIX/M2) z-score composite as a direction gate,
    combined with 4H multi-scale Donchian channel breakout for entry timing.
    The macro gate reduces trade frequency to ~25-40/year, overcoming
    the 4H cost barrier while providing 4-8 hour earlier entry vs 12H.
    """

    def __init__(self, config: MacroPatience4hConfig | None = None) -> None:
        self._config = config or MacroPatience4hConfig()

    @property
    def name(self) -> str:
        return "macro-patience-4h"

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
        config = MacroPatience4hConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {k: str(v) for k, v in self._config.model_dump().items()}
