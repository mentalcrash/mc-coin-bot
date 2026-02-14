"""Return Persistence Score strategy.

Positive return bar ratio for minimal-parameter trend persistence detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.ret_persist.config import RetPersistConfig, ShortMode
from src.strategy.ret_persist.preprocessor import preprocess
from src.strategy.ret_persist.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("ret-persist")
class RetPersistStrategy(BaseStrategy):
    """Return Persistence Score strategy implementation.

    Extreme simplicity: positive return bar ratio as sole signal.
    """

    def __init__(self, config: RetPersistConfig | None = None) -> None:
        self._config = config or RetPersistConfig()

    @property
    def name(self) -> str:
        return "ret-persist"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> RetPersistConfig:
        return self._config

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return preprocess(df, self._config)

    def generate_signals(self, df: pd.DataFrame) -> StrategySignals:
        return generate_signals(df, self._config)

    def warmup_periods(self) -> int:
        return self._config.warmup_periods()

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
    def from_params(cls, **params: Any) -> RetPersistStrategy:
        config = RetPersistConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "persist_window": f"{cfg.persist_window} bars",
            "long_threshold": f"{cfg.long_threshold:.0%}",
            "short_threshold": f"{cfg.short_threshold:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
