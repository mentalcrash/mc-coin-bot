"""CMF Trend Persistence strategy.

Chaikin Money Flow sign persistence over N bars for institutional flow detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.cmf_persist.config import CmfPersistConfig, ShortMode
from src.strategy.cmf_persist.preprocessor import preprocess
from src.strategy.cmf_persist.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("cmf-persist")
class CmfPersistStrategy(BaseStrategy):
    """CMF Trend Persistence strategy implementation.

    Uses CMF sign persistence ratio to detect institutional accumulation/distribution.
    """

    def __init__(self, config: CmfPersistConfig | None = None) -> None:
        self._config = config or CmfPersistConfig()

    @property
    def name(self) -> str:
        return "cmf-persist"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> CmfPersistConfig:
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
    def from_params(cls, **params: Any) -> CmfPersistStrategy:
        config = CmfPersistConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "cmf_period": f"{cfg.cmf_period} bars",
            "persist_window": f"{cfg.persist_window} bars",
            "persist_threshold": f"{cfg.persist_threshold:.0%}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
