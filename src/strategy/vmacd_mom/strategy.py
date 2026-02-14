"""Volume MACD Momentum strategy.

Volume MACD as primary signal with price momentum confirmation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.registry import register
from src.strategy.vmacd_mom.config import ShortMode, VmacdMomConfig
from src.strategy.vmacd_mom.preprocessor import preprocess
from src.strategy.vmacd_mom.signal import generate_signals

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("vmacd-mom")
class VmacdMomStrategy(BaseStrategy):
    """Volume MACD Momentum strategy implementation.

    Uses Volume MACD as primary trend indicator with price momentum confirmation.
    """

    def __init__(self, config: VmacdMomConfig | None = None) -> None:
        self._config = config or VmacdMomConfig()

    @property
    def name(self) -> str:
        return "vmacd-mom"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> VmacdMomConfig:
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
    def from_params(cls, **params: Any) -> VmacdMomStrategy:
        config = VmacdMomConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "vmacd": f"{cfg.vmacd_fast}/{cfg.vmacd_slow}/{cfg.vmacd_signal}",
            "mom_lookback": f"{cfg.mom_lookback} bars",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
