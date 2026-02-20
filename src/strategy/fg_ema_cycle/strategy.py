"""F&G EMA Long-Cycle 전략.

F&G EMA-24w 장기 센티먼트 사이클 크로스오버 활용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fg_ema_cycle.config import FgEmaCycleConfig
from src.strategy.fg_ema_cycle.preprocessor import preprocess
from src.strategy.fg_ema_cycle.signal import generate_signals


@register("fg-ema-cycle")
class FgEmaCycleStrategy(BaseStrategy):
    """F&G EMA Long-Cycle 전략 구현.

    장기 F&G EMA 크로스오버로 매크로 사이클 방향 포착.
    """

    def __init__(self, config: FgEmaCycleConfig | None = None) -> None:
        self._config = config or FgEmaCycleConfig()

    @property
    def name(self) -> str:
        return "fg-ema-cycle"

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
        config = FgEmaCycleConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "ema_slow_span": str(self._config.ema_slow_span),
            "ema_fast_span": str(self._config.ema_fast_span),
            "fear_cycle": str(self._config.fear_cycle),
            "greed_cycle": str(self._config.greed_cycle),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
