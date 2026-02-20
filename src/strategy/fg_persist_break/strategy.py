"""F&G Persistence Break 전략.

F&G 극단 구간의 persistence break(탈출 시점)을 방향 전환 시그널로 활용.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd
    from pydantic import BaseModel

    from src.strategy.types import StrategySignals

from src.strategy.fg_persist_break.config import FgPersistBreakConfig
from src.strategy.fg_persist_break.preprocessor import preprocess
from src.strategy.fg_persist_break.signal import generate_signals


@register("fg-persist-break")
class FgPersistBreakStrategy(BaseStrategy):
    """F&G Persistence Break 전략 구현.

    극단 구간(fear/greed zone)에 N일 이상 체류 후 탈출 시 방향 전환 시그널.
    """

    def __init__(self, config: FgPersistBreakConfig | None = None) -> None:
        self._config = config or FgPersistBreakConfig()

    @property
    def name(self) -> str:
        return "fg-persist-break"

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
        config = FgPersistBreakConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        return {
            "fear_threshold": str(self._config.fear_threshold),
            "greed_threshold": str(self._config.greed_threshold),
            "min_persist": str(self._config.min_persist),
            "price_mom_window": str(self._config.price_mom_window),
            "vol_target": str(self._config.vol_target),
            "short_mode": self._config.short_mode.name,
        }
