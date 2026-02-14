"""OBV Acceleration Momentum strategy.

OBV 2nd derivative (acceleration) for smart money activity intensity detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.obv_accel.config import ObvAccelConfig, ShortMode
from src.strategy.obv_accel.preprocessor import preprocess
from src.strategy.obv_accel.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("obv-accel")
class ObvAccelStrategy(BaseStrategy):
    """OBV Acceleration Momentum strategy implementation.

    Uses OBV 2nd derivative to detect changes in smart money activity intensity.
    """

    def __init__(self, config: ObvAccelConfig | None = None) -> None:
        self._config = config or ObvAccelConfig()

    @property
    def name(self) -> str:
        return "obv-accel"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> ObvAccelConfig:
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
    def from_params(cls, **params: Any) -> ObvAccelStrategy:
        config = ObvAccelConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "obv_smooth": f"{cfg.obv_smooth} bars",
            "accel_window": f"{cfg.accel_window} bars",
            "accel_threshold": f"{cfg.accel_threshold:.1f}z",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
