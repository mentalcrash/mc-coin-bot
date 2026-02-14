"""Dual Volatility Trend strategy.

Yang-Zhang vs Parkinson vol ratio for trend/noise regime detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.strategy.base import BaseStrategy
from src.strategy.dual_vol.config import DualVolConfig, ShortMode
from src.strategy.dual_vol.preprocessor import preprocess
from src.strategy.dual_vol.signal import generate_signals
from src.strategy.registry import register

if TYPE_CHECKING:
    import pandas as pd

    from src.strategy.types import StrategySignals


@register("dual-vol")
class DualVolStrategy(BaseStrategy):
    """Dual Volatility Trend strategy implementation.

    Uses YZ/Parkinson vol ratio to detect information arrival vs noise.
    """

    def __init__(self, config: DualVolConfig | None = None) -> None:
        self._config = config or DualVolConfig()

    @property
    def name(self) -> str:
        return "dual-vol"

    @property
    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close", "volume"]

    @property
    def config(self) -> DualVolConfig:
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
    def from_params(cls, **params: Any) -> DualVolStrategy:
        config = DualVolConfig(**params)
        return cls(config=config)

    def get_startup_info(self) -> dict[str, str]:
        cfg = self._config
        mode_map = {
            ShortMode.DISABLED: "Long-Only",
            ShortMode.HEDGE_ONLY: f"Hedge-Short (<={cfg.hedge_threshold:.0%})",
            ShortMode.FULL: "Long/Short",
        }
        return {
            "vol_estimator_window": f"{cfg.vol_estimator_window} bars",
            "ratio_upper": f"{cfg.ratio_upper:.2f}",
            "ratio_lower": f"{cfg.ratio_lower:.2f}",
            "vol_target": f"{cfg.vol_target:.0%}",
            "mode": mode_map.get(cfg.short_mode, "Unknown"),
        }
